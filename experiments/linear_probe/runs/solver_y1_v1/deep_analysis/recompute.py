"""Recompute the frozen pilot ledgers on CPU from sealed source records."""
import csv
import hashlib
import json
import sys
from pathlib import Path
from statistics import mean, median

OUT = Path(__file__).resolve().parent
PILOT = OUT.parent
RUNS = PILOT.parent
ROOT = RUNS.parents[2]
sys.path.insert(0, str(ROOT))
from experiments.linear_probe.common import allocate_words, project_credits, verify
from experiment_state import digest
from math_answer_eval import evaluate_answer, evaluator_info
from run_planner import parse_credits, parse_steps

ARMS = ('vanilla', 'teacher', 'uniform', 'position', 'ridge')
METHODS = dict(ridge='probe', uniform='uniform', position='train_position_mean')
FIELDS = ('prompt_tokens', 'completion_tokens', 'latency_seconds')


def load(path):
    return verify(json.loads(path.read_text()))


def check_hashes(mapping):
    for path, sha in mapping.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha, path


def ledger(attempts):
    values = [a.get('usage', {}) for a in attempts]
    fields = {k: [v.get(k) for v in values] for k in FIELDS}
    fields['total_tokens'] = [
        v[FIELDS[0]] + v[FIELDS[1]] if all(v.get(k) is not None for k in FIELDS[:2]) else None
        for v in values
    ]
    return dict(attempts=len(values), **{
        k: dict(known_sum=round(sum(v for v in vs if v is not None), 6),
                missing_attempts=vs.count(None)) for k, vs in fields.items()
    })


def write(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def by_id(rows):
    return {r['id']: r for r in rows}


def main():
    audit = load(PILOT / 'audit_221040.json')
    prior = load(PILOT / 'analysis_221051_221793.json')
    check_hashes(audit['protected_files'])
    check_hashes(prior['input_sha256'])
    check_hashes({prior['report_path']: prior['report_sha256']})
    assert audit['passed'] and prior['passed']
    data = by_id(load(RUNS / 'pilot_v1/data.json')['splits']['evaluation'])
    recovered = load(RUNS / 'pilot_v1/evaluation_recovered.json')
    ancestry = recovered['recovery_v2']
    check_hashes({ancestry['source_path']: ancestry['source_sha256']})
    original = by_id(load(Path(ancestry['source_path']))['records'])
    cap = by_id(recovered['records'])
    assert all(cap[i] == original[i] for i in cap)
    assert not ancestry['accepted_repaired_rows']
    labeled = by_id(load(RUNS / 'credit_a2000_tp4/evaluation_labeled.json')['records'])
    diagnostic = load(RUNS / 'analysis_credit_tp4_220316/extended_220319.json')
    check_hashes(diagnostic['input_sha256'])
    frozen = {arm: by_id(diagnostic['splits']['evaluation'][key]['records'])
              for arm, key in METHODS.items()}
    plan = by_id(audit['solver_requests'])
    assert set(data) == set(cap) == set(labeled) == set(plan)
    assert evaluator_info('original') == audit['evaluator']
    budget_rows, solver_rows, cost_rows, generation_rows = [], [], [], []
    session_info = {}
    for job, mode in ((221051, 'main'), (221043, 'smoke')):
        gate = load(PILOT / f'complete_solver_{mode}_{job}.json')
        check_hashes(gate['protected_outputs'])
        session = load(PILOT / f'responses_solver_{job}_{mode}.json')
        memory = load(PILOT / f'memory_solver_{job}_{mode}.json')
        ready = load(PILOT / f'ready_solver_{job}_{mode}.json')
        assert session['passed'] and gate['passed']
        assert session['audit_fingerprint'] == gate['audit_fingerprint'] == audit['fingerprint']
        attempts = [a for row in session['records'] for a in row['stage']['attempts']]
        assert ledger(attempts) == session['all_attempt_cost']
        assert (PILOT / f'server_solver_{job}_{mode}.log').read_text().count(
            'POST /v1/chat/completions') == len(attempts)
        session_info[mode] = dict(wall=session['wall_seconds'], monitored=memory['wall_seconds'],
                                  startup=ready['server_start_to_ready_seconds'], cost=ledger(attempts))
        for row in session['records']:
            stage = row['stage']
            result = stage['result']
            raw = stage['attempts'][-1]['raw_response']
            choice = raw['choices'][0]
            payload = row['payload']
            source = plan[row['id']]
            arm = row['arm']
            assert row['request_fingerprint'] == digest(payload)
            assert choice['message']['content'] == result['content']
            assert choice['finish_reason'] == result['finish_reason']
            assert all(raw['usage'][k] == result['usage'][k] for k in FIELDS[:2])
            assert raw['usage']['total_tokens'] == sum(raw['usage'][k] for k in FIELDS[:2])
            assert stage['attempts'][-1]['usage'] == result['usage']
            assert row['answer'] == source['answer'] == data[row['id']]['answer']
            assert row['steps_fingerprint'] == source['steps_fingerprint']
            assert row['budgets_words'] == source['budgets_words'][arm]
            assert row['credits'] == source['credits'].get(arm)
            expected_cap = 128 if mode == 'smoke' and row['request_order'] < 2 else 8192
            assert payload == {**source['requests'][arm], 'max_tokens': expected_cap}
            usage = ledger(stage['attempts'])
            budget = row['total_words']
            words = len(result['content'].split())
            status = evaluate_answer(result['content'], row['answer'], result['finish_reason'],
                                     stage['status'], mode='original')
            if mode == 'main' or result['finish_reason'] == 'stop':
                assert status == stage['parsed']['answer_evaluation']
            solver_rows.append(dict(job=job, session=mode, id=row['id'], level=data[row['id']]['level'],
                                    arm=arm, order=row['request_order'], seed=payload['seed'],
                                    max_tokens=payload['max_tokens'], finish_reason=result['finish_reason'],
                                    evaluated_status=status['status'], prediction=status.get('prediction'),
                                    reference=row['answer'], attempts=usage['attempts'],
                                    retries=usage['attempts'] - 1, **{
                                        f'solver_{k}': v['known_sum'] for k, v in usage.items() if k != 'attempts'
                                    }, **{
                                        f'missing_{k}_attempts': v['missing_attempts']
                                        for k, v in usage.items() if k != 'attempts'
                                    }, completion_words=words, total_budget_words=budget,
                                    whole_word_ratio=words / budget if budget else None,
                                    whole_word_compliant=words <= budget if budget and result['finish_reason'] == 'stop' else None,
                                    subtask_compliance=None))
    main_rows = [r for r in solver_rows if r['session'] == 'main']
    assert [(r['id'], r['arm']) for r in main_rows] == [
        (i, arm) for i, p in plan.items() for arm in p['order']]
    write('problem_comparison.csv', solver_rows)

    def component(i, arm, name, scope, usage, attempts=None, **details):
        cost_rows.append(dict(id=i, arm=arm, component=name, scope=scope, attempts=attempts,
                              **usage, **details))

    for i, p in plan.items():
        capture = cap[i]['capture']
        control = capture['capture_control']
        teacher = labeled[i]['teacher_stage']
        teacher_credit = labeled[i]['teacher']['credits']
        raw_teacher = teacher['attempts'][-1]['raw_response']
        teacher_text = raw_teacher['choices'][0]['message']['content']
        assert teacher_text == teacher['result']['content']
        assert all(raw_teacher['usage'][k] == teacher['result']['usage'][k] for k in FIELDS[:2])
        assert teacher['attempts'][-1]['usage'] == teacher['result']['usage']
        assert parse_credits(teacher_text, len(p['steps'])) == teacher['parsed'] == teacher_credit == p['credits']['teacher']
        assert parse_steps(capture['text']) == p['steps']
        assert capture['steps'] == p['steps'] == labeled[i]['capture']['steps']
        assert cap[i]['steps_fingerprint'] == p['steps_fingerprint']
        assert capture == labeled[i]['capture']
        assert control['same_generated_ids'] and capture['generated_ids'] == control['generated_ids']
        capture_attempts = cap[i]['attempts']
        for a in capture_attempts:
            ca = a['capture_control_attempt']
            assert a['usage'] == capture['usage']
            assert len(ca['diagnostics']['generated_ids']) == ca['completion_tokens']
            assert len(capture['generated_ids']) == capture['usage']['completion_tokens']
            generation_rows.append(dict(id=i, capture_status=a['status'], control_status=ca['status'],
                                        capture_prompt=a['usage'].get('prompt_tokens'),
                                        capture_output=a['usage'].get('completion_tokens'),
                                        control_prompt=ca.get('prompt_tokens'),
                                        control_output=ca.get('completion_tokens'),
                                        wall_including_storage=a.get('wall_seconds_including_storage')))
        for arm in ARMS:
            if arm == 'vanilla':
                continue
            pred = frozen[arm][i] if arm in METHODS else None
            timing = pred['probe'] if pred else {}
            credits = project_credits(timing['raw'])['credits'] if pred else teacher_credit
            budgets = allocate_words(credits, p['total_words'], 'constant')
            assert sum(budgets) == p['total_words']
            assert credits == p['credits'][arm] and budgets == p['budgets_words'][arm]
            if pred:
                assert credits == timing['credits'] and budgets == pred['budgets']['probe']
                assert pred['steps'] == p['steps'] and pred['teacher_credits'] == teacher_credit
            for n, step in enumerate(p['steps']):
                budget_rows.append(dict(id=i, level=data[i]['level'], step=n + 1, subtask=step,
                                        arm=arm, raw_prediction=timing['raw'][n] if pred else None,
                                        credit_pp=credits[n], budget_words=budgets[n],
                                        total_budget_words=p['total_words'],
                                        delta_credit_vs_teacher=credits[n] - teacher_credit[n],
                                        delta_words_vs_teacher=budgets[n] - p['budgets_words']['teacher'][n]))
            decomposition = capture if arm == 'ridge' else control
            metrics = decomposition['metrics']
            assert metrics['extra_forward_passes'] == metrics['feature_forward_input_tokens'] == 0
            usage = {k: decomposition['usage'][k] for k in FIELDS[:2]}
            usage['latency_seconds'] = metrics['generation_plus_feature_seconds']
            component(i, arm, 'decomposition', 'historical_HF_1x3090', usage, 1,
                      generation_seconds=metrics['generation_seconds'],
                      hidden_retention_seconds=metrics['hidden_retention_seconds'],
                      alignment_seconds=metrics['alignment_seconds'],
                      feature_transfer_seconds=metrics['feature_transfer_seconds'],
                      extra_forward_passes=metrics['extra_forward_passes'])
            if arm == 'ridge':
                component(i, arm, 'feature_validation', 'historical_capture_path_wall',
                          dict(prompt_tokens=0, completion_tokens=0,
                               latency_seconds=metrics['feature_validation_seconds']))
            if arm == 'teacher':
                t = ledger(teacher['attempts'])
                assert all(t[k]['missing_attempts'] == 0 for k in FIELDS)
                component(i, arm, 'credit_assessment', 'historical_vLLM_4xA2000',
                          {k: t[k]['known_sum'] for k in FIELDS}, t['attempts'])
            else:
                keys = ['inference_seconds', 'projection_seconds']
                if arm == 'ridge':
                    keys.append('feature_read_seconds')
                component(i, arm, 'frozen_credit', 'historical_CPU',
                          dict(prompt_tokens=0, completion_tokens=0,
                               latency_seconds=sum(timing[k] for k in keys)),
                          **{k: timing[k] for k in keys})
    for r in main_rows:
        component(r['id'], r['arm'], 'solver', 'measured_221051_vLLM_4x3090',
                  {k: r[f'solver_{k}'] for k in FIELDS}, r['attempts'])
    for r in list(cost_rows):
        r['total_tokens'] = r['prompt_tokens'] + r['completion_tokens']
    for arm in ARMS:
        rows = [r for r in cost_rows if r['arm'] == arm]
        for name in dict.fromkeys(r['component'] for r in rows):
            parts = [r for r in rows if r['component'] == name]
            component('ALL', arm, name, parts[0]['scope'],
                      {k: sum(r[k] for r in parts) for k in (*FIELDS, 'total_tokens')},
                      sum(r['attempts'] or 0 for r in parts),
                      **{k: sum(r[k] for r in parts) for k in (
                          'generation_seconds', 'hidden_retention_seconds', 'alignment_seconds',
                          'feature_transfer_seconds', 'extra_forward_passes', 'inference_seconds',
                          'projection_seconds', 'feature_read_seconds') if all(k in r for r in parts)})
        for name, parts in [('planner_sum', [r for r in rows if r['component'] != 'solver']),
                            ('online_sum', rows)]:
            usage = {k: sum(r[k] for r in parts) for k in (*FIELDS, 'total_tokens')}
            component('ALL', arm, name, 'mixed_hardware_estimate' if arm != 'vanilla' else 'solver_only',
                      usage)
            if name == 'online_sum':
                for k in FIELDS:
                    assert abs(usage[k] - prior['results']['arms'][arm]['estimated_online_totals'][k]) < 1e-6
        old = prior['results']['arms'][arm]
        sol = [r for r in main_rows if r['arm'] == arm]
        assert sum(r['evaluated_status'] == 'correct' for r in sol) == old['correct']
        for k in (*FIELDS, 'total_tokens'):
            assert abs(sum(r[f'solver_{k}'] for r in sol) - old['all_attempt_cost'][k]['known_sum']) < 1e-6
        eligible = [r['whole_word_compliant'] for r in sol if r['whole_word_compliant'] is not None]
        assert (sum(eligible), len(eligible)) == tuple(
            old['word_budget_compliance'][k] for k in ('compliant_count', 'measured_count'))
    totals = {(r['arm'], r['component']): r for r in cost_rows if r['id'] == 'ALL'}
    for r in totals.values():
        name = r['component']
        if name not in ('planner_sum', 'solver', 'online_sum'):
            continue
        for base in ('teacher', 'vanilla'):
            for k in (*FIELDS, 'total_tokens'):
                baseline = totals[base, name][k]
                r[f'{k}_saving_pct_vs_{base}'] = 100 * (baseline - r[k]) / baseline if baseline else None
    write('cost_breakdown.csv', cost_rows)
    write('budget_comparison.csv', budget_rows)
    write('decomposition_attempts.csv', generation_rows)
    print('PASS: protected hashes, evaluator identity, 25 main + 4 smoke raw responses, frozen projections, budgets, and prior totals.')
    for mode, x in session_info.items():
        print(mode, x)
    for arm in ARMS:
        print(arm, [(name, {k: totals[arm, name][k] for k in (*FIELDS, 'total_tokens')})
                    for name in ('planner_sum', 'solver', 'online_sum')])
        if arm != 'vanilla':
            credit_mae, word_mae = [], []
            for i in plan:
                b = [r for r in budget_rows if r['id'] == i and r['arm'] == arm]
                credit_mae.append(mean(abs(r['delta_credit_vs_teacher']) for r in b))
                word_mae.append(mean(abs(r['delta_words_vs_teacher']) for r in b))
            print('MAE by problem', credit_mae, word_mae, 'macro', mean(credit_mae), mean(word_mae))
    for base in ARMS[:-1]:
        deltas = []
        for i in plan:
            rr = {r['arm']: r for r in main_rows if r['id'] == i}
            deltas.append([rr['ridge'][f'solver_{k}'] - rr[base][f'solver_{k}']
                           for k in (*FIELDS, 'total_tokens')])
        print('ridge minus', base, deltas, 'median', list(map(median, zip(*deltas))))
    for arm in ('ridge', 'teacher', 'position', 'uniform'):
        print('components', arm, [r for r in totals.values() if r['arm'] == arm
                                  and r['component'] not in ('planner_sum', 'online_sum', 'solver')])


if __name__ == '__main__':
    main()
