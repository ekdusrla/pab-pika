"""CPU paired statistics and complete-denominator result export."""
import argparse
import csv
import json
from math import comb, fsum
from pathlib import Path
from statistics import mean

import numpy as np

from .audit import ARMS, OUT
from .runner import experiment, journal, solver_payload
from ..common import digest, file_hash, seal
from ..solver_y1_v1.analysis import budget_usage
from .evaluator import evaluator_info, validate_solver
from experiment_state import atomic_save, attempt_cost, lock

METRICS = ('prompt_tokens', 'completion_tokens', 'total_tokens', 'latency_seconds')


def complete_cost(attempts):
    ledger = attempt_cost([{'attempts': attempts}])
    return {k: ledger[k]['known_sum'] if attempts and not ledger[k]['missing_attempts'] else None
            for k in METRICS}, ledger


def paired_summary(differences):
    if not differences:
        return None
    values = np.asarray(differences)
    n = len(values)
    rng = np.random.default_rng(20261119)
    draws = values[rng.integers(0, n, size=(10000, n))].mean(axis=1)
    return dict(n=n, mean=float(values.mean()), signed_quantiles=np.quantile(
        values, [0, .05, .25, .5, .75, .95, 1]).tolist(),
        negative=int((values < 0).sum()), ties=int((values == 0).sum()),
        positive=int((values > 0).sum()), negative_fraction=float((values < 0).mean()),
        paired_mean_ci95=np.quantile(draws, [.025, .975]).tolist(),
        leave_one_out_mean_range=[float((values.sum()-values.max())/(n-1)),
                                   float((values.sum()-values.min())/(n-1))] if n > 1 else None)


def accuracy_interval(differences):
    """Conservative paired interval via simultaneous exact discordance bounds."""
    n = len(differences)

    def lower(count):
        if not count:
            return 0.
        lo, hi = 0., 1.
        coefficients = [(k, comb(n, k)) for k in range(count, n+1)]
        for _ in range(50):
            p = (lo+hi)/2
            tail = fsum(c*p**k*(1-p)**(n-k) for k, c in coefficients)
            if tail < .0125:
                lo = p
            else:
                hi = p
        return (lo+hi)/2

    def interval(count):
        return lower(count), 1-lower(n-count)

    win = interval(sum(v > 0 for v in differences))
    loss = interval(sum(v < 0 for v in differences))
    return [win[0]-loss[1], win[1]-loss[0]]


def summarize(rows, margin):
    grouped = {a: {r['id']: r for r in rows if r['arm'] == a} for a in ARMS}
    ids = list(grouped['ridge'])
    arms = {}
    contrasts = {}
    for arm, by_id in grouped.items():
        assert set(by_id) == set(ids) and len(by_id) == sum(r['arm'] == arm for r in rows)
        group = list(by_id.values())
        arms[arm] = dict(expected=len(group), correct=sum(r['correct'] for r in group),
                         accuracy=mean(r['correct'] for r in group),
                         outcomes={s:sum(r['answer_status'] == s for r in group)
                                   for s in sorted({r['answer_status'] for r in group})},
                         truncated=sum(r['finish_reason'] == 'length' for r in group))
        for metric in [f'{stage}_{k}' for stage in ('planner','solver','system') for k in METRICS]:
            known = [r[metric] for r in group if r.get(metric) is not None]
            arms[arm][metric] = dict(sum_known=sum(known), n_known=len(known))
    for method in ARMS:
        for base in ('teacher', 'vanilla', 'uniform', 'position'):
            if method == base:
                continue
            truth = grouped[base]
            target = grouped[method]
            diffs = [int(target[i]['correct'])-int(truth[i]['correct']) for i in ids]
            ci = accuracy_interval(diffs)
            entry = dict(accuracy_difference_pp=100*mean(diffs), accuracy_ci95_pp=[100*v for v in ci],
                         accuracy_wins=diffs.count(1), accuracy_losses=diffs.count(-1),
                         accuracy_discordance=(diffs.count(1)+diffs.count(-1))/len(diffs),
                         noninferiority=(100*ci[0] > -margin) if base == 'teacher' and method == 'ridge'
                         and margin is not None else None, metrics={})
            for metric in [f'{stage}_{k}' for stage in ('planner','solver','system') for k in METRICS]:
                pairs = [(target[i].get(metric), truth[i].get(metric)) for i in ids]
                pairs = [(a,b) for a,b in pairs if a is not None and b is not None]
                stats = paired_summary([a-b for a,b in pairs])
                if stats:
                    denominator = sum(b for _,b in pairs)
                    stats['matched_sum_saving_pct'] = -100*sum(a-b for a,b in pairs)/denominator if denominator else None
                entry['metrics'][metric] = stats
            contrasts[f'{method}_vs_{base}'] = entry
    strata = {}
    for level in sorted({r['level'] for r in rows}):
        strata[level] = {}
        for arm in ARMS:
            group = [r for r in rows if r['level'] == level and r['arm'] == arm]
            vals = [r['credit_mae_pp'] for r in group if r.get('credit_mae_pp') is not None]
            last = [r['credits'][-1] for r in group if r.get('credits')]
            budget = [r['budget_l1_words'] for r in group if r.get('budget_l1_words') is not None]
            strata[level][arm] = dict(n=len(group), credit_mae_pp=mean(vals) if vals else None,
                                      last_credit_pp=mean(last) if last else None,
                                      budget_l1_words=mean(budget) if budget else None)
    return dict(arms=arms, paired=contrasts, by_level=strata, margin_pp=margin,
                accuracy_interval='paired exact conservative Bonferroni discordance bounds',
                confirmatory_claim_allowed=False,
                noninferiority_scope='exploratory threshold check on previously used MATH-500',
                other_intervals='paired problem bootstrap, single Solver seed per result group')


def collect(manifest, conf):
    rows = []
    checkpoints = {}
    for row in manifest['rows']:
        path = OUT / 'journal' / manifest['fingerprint'] / f"chunk-{row['chunk']:02d}"
        def get(arm, seed=42, required=False):
            j = journal(path, manifest, conf, row, arm, seed)
            if required and not j.path.is_file():
                raise ValueError(f'Incomplete execution: {row["id"]} {arm} {seed}')
            if j.path.is_file():
                checkpoints[str(j.path)] = file_hash(j.path)
            return j.record
        decomp = get('capture')['stage']
        cap = decomp.get('result', {})
        teacher = get('teacher_credit')['stage']
        probe = get('probe')['stage'].get('result', {})
        truth = teacher.get('parsed', {}).get('credits')
        for seed in conf['solver_seeds']:
            teacher_solver = get('teacher', seed, required=True)
            truth_budget = teacher_solver.get('allocation', {}).get('budgets_words')
            for arm in ARMS:
                result = get(arm, seed, required=True)
                state = result['stage']
                record = state.get('result', {})
                allocation = result.get('allocation', {})
                credits = allocation.get('credits')
                budgets = allocation.get('budgets_words', [])
                if state['status'] in ('pending', 'running', 'interrupted', 'response_received'):
                    raise ValueError('Unfinished request: ' + result['request_id'])
                if state['attempts']:
                    payload = solver_payload(row, cap.get('steps', []), arm, budgets, seed)
                    if state['request'] != payload or any(a['request_fingerprint'] != digest(payload)
                                                          for a in state['attempts']):
                        raise ValueError('Request provenance mismatch: ' + result['request_id'])
                if state['status'] == 'success':
                    latest = state['attempts'][-1]
                    raw = latest['raw_response']
                    if (latest['status'] != 'success' or raw != record['raw_response']
                            or raw['choices'][0]['message']['content'] != record['content']
                            or raw['choices'][0]['finish_reason'] != record['finish_reason']
                            or any(raw['usage'][k] != record['usage'][k] for k in METRICS[:2])
                            or validate_solver(record, row) != state['parsed']):
                        raise ValueError('Raw response does not reproduce the score: ' + result['request_id'])
                reason = record.get('finish_reason')
                ev = state.get('parsed', {}).get('answer_evaluation', {})
                outcome = ev.get('status', state['status'])
                s_cost, solver_ledger = complete_cost(state['attempts'])
                t_cost, t_ledger = complete_cost(teacher['attempts'])
                mode = 'capture' if arm == 'ridge' else 'control'
                history = [a for a in decomp['attempts'] if a['role'] == mode]
                c_cost, c_ledger = complete_cost(history)
                # The Solver consumes shared capture steps. Control token/time attribution
                # is a counterfactual when control generated different steps.
                estimate_aligned = arm == 'vanilla' or (bool(cap) and (arm == 'ridge' or cap['same_generated_ids']))
                planner = dict.fromkeys(METRICS, 0 if arm == 'vanilla' else None)
                if arm != 'vanilla':
                    for k in METRICS:
                        add = t_cost[k] if arm == 'teacher' else 0
                        base = c_cost[k]
                        if k == 'latency_seconds' and base is not None:
                            base = sum(a.get('generation_and_feature_wall_seconds',
                                        a['usage']['latency_seconds']) for a in history)
                            if arm == 'ridge':
                                add = sum(probe[k] for k in ('feature_read_seconds','inference_seconds','projection_seconds')) if probe else None
                        planner[k] = base + add if base is not None and add is not None else None
                combined = {k: planner[k]+s_cost[k] if planner[k] is not None and s_cost[k] is not None
                            and estimate_aligned else None for k in METRICS}
                rows.append(dict(id=row['id'], chunk=row['chunk'], level=row['level'], arm=arm, seed=seed,
                                 request_id=result['request_id'], correct=outcome == 'correct', answer_status=outcome,
                                 prediction=ev.get('prediction'), reference=row['answer'], finish_reason=reason,
                                 max_tokens=conf['solver_max_tokens'], steps=cap.get('steps'), credits=credits,
                                 teacher_credits=truth, budgets_words=budgets, teacher_budgets_words=truth_budget,
                                 credit_mae_pp=mean(abs(a-b) for a,b in zip(credits,truth)) if credits and truth else None,
                                 budget_l1_words=sum(abs(a-b) for a,b in zip(budgets,truth_budget)) if budgets and truth_budget else None,
                                 control_same_generated_ids=cap.get('same_generated_ids'),
                                 end_to_end_seconds=None, system_latency_scope='reconstructed_staged',
                                 capture_hidden_retention_seconds=cap.get('metrics', {}).get('hidden_retention_seconds') if arm=='ridge' else None,
                                 capture_alignment_seconds=cap.get('metrics', {}).get('alignment_seconds') if arm=='ridge' else None,
                                 capture_transfer_seconds=cap.get('metrics', {}).get('feature_transfer_seconds') if arm=='ridge' else None,
                                 probe_seconds=probe.get('inference_seconds') if arm=='ridge' else None,
                                 retries=max(0,solver_ledger['attempts']-1), solver_attempt_cost=solver_ledger,
                                 decomposition_cost=c_ledger if arm!='vanilla' else None,
                                 teacher_credit_cost=t_ledger if arm=='teacher' else None,
                                 **budget_usage(record.get('content') or '',reason,allocation.get('total_words')),
                                 **{f'{name}_{k}':v for name, d in (('decomposition',c_cost if arm!='vanilla' else dict.fromkeys(METRICS)),
                                                    ('teacher_call',t_cost if arm=='teacher' else dict.fromkeys(METRICS)),
                                                    ('planner',planner),('solver',s_cost),('system',combined))
                                    for k,v in d.items()}))
    return rows, checkpoints


def export(dest, seed, selected, hashes, fingerprint, margin):
    report = summarize(selected, margin)
    json_path = dest / f'seed-{seed}.json'
    csv_path = dest / f'seed-{seed}.csv'
    temp = csv_path.with_suffix('.csv.tmp')
    atomic_save(json_path, seal({'summary': report, 'rows': selected,
                                'checkpoint_sha256': hashes, 'manifest': fingerprint}))
    with temp.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        for row in selected:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})
    temp.replace(csv_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    a = p.parse_args()
    doc, conf = experiment(a.manifest)
    if evaluator_info() != doc['evaluator']:
        raise ValueError('Run analysis in the frozen evaluator environment')
    with lock(OUT / 'batch'):
        rows, hashes = collect(doc, conf)
        dest = OUT / 'analysis' / doc['fingerprint']
        for seed in conf['solver_seeds']:
            selected = [r for r in rows if r['seed'] == seed]
            assert len(selected) == 2500
            export(dest, seed, selected, hashes, doc['fingerprint'], conf['accuracy_margin_pp'])


if __name__ == '__main__':
    main()
