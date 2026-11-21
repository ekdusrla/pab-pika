"""Frozen, CPU-prepared 100-question ablation; network is confined to `run`."""
import argparse
import copy
import csv
import io
import json
import random
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
from experiment_state import (atomic_save, digest, file_hash, load_json, lock,
                              read_manifest, run_stage, stage, termination_handler,
                              StopRequested, ServerDead, versions)
from math_answer_eval import evaluate_answer, evaluator_info, extract_boxed, original_parser
from run_solver_compare import allocate_words, messages_for

SOURCE = BASE / 'runs/math500_llama31_8b_v1'
RUN = BASE / 'runs/math500_llama31_8b_ablation_pilot_v1'
MODEL = 'Qwen/QwQ-32B'
REVISION = '976055f8c83f394f35dbd3ab09a285a984907bd0'
SNAPSHOT = Path('/data/dada0228/huggingface/hub/models--Qwen--QwQ-32B/snapshots') / REVISION
METHODS = ('vanilla', 'weighted_pab', 'plan_only', 'equal_credit')
NEW = ('plan_only', 'equal_credit')
COUNTS = {1: 9, 2: 18, 3: 21, 4: 25, 5: 27}
SEED = 20261109
EXPECTED = {
    'planner': '7f44008b64ee63048c492e5885bfe3c36f1f6c0061468f37edc0a0f1229071ef',
    'solver': '6982992a14e4c473cb2efc2ea6ebeefd5b4291cd22d987999244fc0c16fb5685',
}
EVAL = '48f99eebcd002f2a2bba7e64114a749b0b8bb74429be51dad7080fb4cc7b80c6'
IMMUTABLE = ('selection.json', 'requests.json', 'audit_baseline.json', 'protocol.json')


def select(manifest):
    """Selection reads only ID/level, never outcomes, credits, or token usage."""
    rng = random.Random(SEED)
    selected = []
    for level, count in COUNTS.items():
        candidates = sorted(({'id': r['id'], 'level': r['level']}
                             for r in manifest['records'] if r['level'] == level),
                            key=lambda r: int(r['id']))
        selected.extend(rng.sample(candidates, count))
    return {'seed': SEED, 'counts': COUNTS, 'records': selected,
            'ids': [r['id'] for r in selected],
            'rule': 'Levels ascending; numeric ID sort within level; one random.Random(seed); '
                    'random.sample per level; retain draw order. Only ID and level are read.'}


def budget_sentence(n):
    return f' Please only think a little, and directly solve it using up to {n} words.'


def payload(row, method):
    """Edit only the uniquely identified decomposition block of the frozen request."""
    request = copy.deepcopy(row['methods']['weighted_pab']['request'])
    steps, budgets = row['steps'], row['words_per_step']
    old = '\n\n'.join(s + budget_sentence(b) for s, b in zip(steps, budgets))
    if method == 'plan_only':
        new = '\n\n'.join(steps)
    elif method == 'equal_credit':
        equal = allocate_words([1] * len(steps), row['words_budget'], 'constant')
        new = '\n\n'.join(s + budget_sentence(b) for s, b in zip(steps, equal))
    elif method == 'weighted_pab':
        return request
    else:
        raise ValueError(method)
    user = request['messages'][1]
    if len(request['messages']) != 2 or user['role'] != 'user' or user['content'].count(old) != 1:
        raise ValueError('Decomposition block is not uniquely identifiable: ' + row['id'])
    before, after = user['content'].split(old)
    user['content'] = before + new + after
    return request


def sources():
    paths = {'planner': SOURCE / 'planner.json', 'solver': SOURCE / 'solver.json',
             'manifest': BASE / 'math500_manifest.json'}
    hashes = {k: file_hash(v) for k, v in paths.items()}
    for key, expected in EXPECTED.items():
        if hashes[key] != expected:
            raise ValueError('Frozen baseline hash changed: ' + key)
    planner, solver = load_json(paths['planner']), load_json(paths['solver'])
    manifest = read_manifest(paths['manifest'])
    if evaluator_info('original')['fingerprint'] != EVAL or solver['evaluator']['fingerprint'] != EVAL:
        raise ValueError('Original evaluator/dependency fingerprint changed')
    config = solver['config']
    required = {'model': MODEL, 'revision': REVISION, 'temperature': .1, 'top_p': .9,
                'max_tokens': 8192, 'seed': 42, 'max_model_len': 12288, 'decay': 'constant'}
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError('Baseline generation settings differ')
    if digest(planner['config']) != config['planner_config_fingerprint']:
        raise ValueError('Planner config linkage differs')
    plans = {r['id']: r for r in planner['stats']}
    rows = {r['id']: r for r in solver['comparisons']}
    if len(plans) != 500 or len(rows) != 500 or len(solver['comparisons']) != 500:
        raise ValueError('Missing/duplicate baseline rows')
    for original in manifest['records']:
        r, p = rows[original['id']], plans[original['id']]
        for key in ('id', 'query', 'answer', 'level', 'record_fingerprint', 'source_index'):
            if r[key] != original[key] or p[key] != original[key]:
                raise ValueError('Baseline/manifest mapping changed')
        if p['status'] != 'success' or r['steps'] != p['steps'] or r['credits'] != p['credits']:
            raise ValueError('Final plan mismatch')
        if allocate_words(r['credits'], r['words_budget'], 'constant') != r['words_per_step']:
            raise ValueError('Weighted budgets changed')
        for method in ('vanilla', 'weighted_pab'):
            target = r['methods'][method]
            request = target['request']
            if target['status'] != 'success' or len(target['attempts']) != 1:
                raise ValueError('Baseline must have one successful solver attempt')
            if set(request) != {'model', 'messages', 'max_tokens', 'temperature', 'top_p', 'seed'}:
                raise ValueError('Unexpected baseline payload fields')
            if any(request[k] != required[k] for k in request if k != 'messages'):
                raise ValueError('Request generation settings differ')
            expected_messages = messages_for(r, method, r['words_per_step'],
                                             config['instruction'], config['output_format'])
            if request['messages'] != expected_messages or digest(request) != target['attempts'][0]['request_fingerprint']:
                raise ValueError('Frozen prompt/request mismatch')
    return manifest, planner, solver, paths, hashes


def dependencies():
    files = [Path(__file__), Path(__file__).with_name('solver_pilot_y4.sbatch'),
             BASE / 'experiment_state.py', BASE / 'run_solver_compare.py',
             BASE / 'math_answer_eval.py', ROOT / 'utils/qwen_math_parser.py']
    return {str(p.relative_to(ROOT)): file_hash(p) for p in files}


def tokenizer_info():
    """Reads small configs/tokenizer files, never model weights."""
    names = ('config.json', 'generation_config.json', 'tokenizer_config.json',
             'tokenizer.json', 'model.safetensors.index.json')
    result = {n: file_hash(SNAPSHOT / n) for n in names}
    index = load_json(SNAPSHOT / 'model.safetensors.index.json')
    if not all((SNAPSHOT / n).is_file() for n in set(index['weight_map'].values())):
        raise ValueError('Pinned model cache is incomplete; downloads are not allowed')
    return result


def diagnostic(content, official):
    """Observe only. No brace stripping, normalization repair, or alternate score."""
    content = content or ''
    boxed = extract_boxed(content)
    prediction = original_parser().strip_string(original_parser().extract_answer(content, 'math'))
    flags = []
    if boxed is None:
        flags.append('no_balanced_last_boxed')
    if boxed and boxed.startswith('{') and boxed.endswith('}'):
        flags.append('outer_braces_require_manual_review')
    if boxed and any(x in boxed for x in ('[', ']', '(', ')', r'\{', r'\}', r'\cup', r'\in')):
        flags.append('interval_set_or_tuple_do_not_autofix')
    if r'\boxed' in content and boxed is None:
        flags.append('boxed_marker_without_balanced_extraction')
    if official['status'] == 'undetermined':
        flags.append('official_undetermined')
    return {'content_sha256': digest(content), 'last_boxed_raw': boxed,
            'original_extractor_prediction': prediction, 'flags': flags,
            'automatic_transformation': None, 'alternate_accuracy': None,
            'manual_review': None,
            'note': 'Official score is immutable. Flags are not mathematical error labels.'}


def audit_baseline(solver):
    audit, mismatches = [], []
    for row in solver['comparisons']:
        for method in ('vanilla', 'weighted_pab'):
            t = row['methods'][method]
            r = t['result']
            ev = evaluate_answer(r['content'], row['answer'], r['finish_reason'], mode='original')
            if ev != t['evaluation']:
                mismatches.append({'id': row['id'], 'method': method,
                                   'stored': t['evaluation'], 'reevaluated': ev})
            audit.append({'id': row['id'], 'method': method, 'reference': row['answer'],
                          'official_stored': t['evaluation'], 'official_cpu_replay': ev,
                          'diagnostic': diagnostic(r['content'], ev)})
    return {'evaluator': evaluator_info('original'), 'count': len(audit),
            'mismatches': mismatches, 'rows': audit,
            'policy': 'No generated text, reference, or official score is repaired.'}


def make_requests(selection, solver):
    rows = {r['id']: r for r in solver['comparisons']}
    requests = []
    for identifier in selection['ids']:
        r = rows[identifier]
        req = {m: (copy.deepcopy(r['methods'][m]['request']) if m in ('vanilla', 'weighted_pab')
                   else payload(r, m)) for m in METHODS}
        requests.append({'id': identifier, 'level': r['level'], 'payloads': req,
                         'payload_fingerprints': {m: digest(v) for m, v in req.items()},
                         'equal_reuses_weighted': req['equal_credit'] == req['weighted_pab'],
                         'equal_budgets': allocate_words([1] * len(r['steps']), r['words_budget'], 'constant'),
                         'weighted_budgets': r['words_per_step'], 'steps': r['steps']})
    return {'rows': requests, 'new_solver_calls': sum(1 + (not r['equal_reuses_weighted']) for r in requests),
            'equal_reuse_count': sum(r['equal_reuses_weighted'] for r in requests)}


def initialize(selection, requests, solver, config):
    old = {r['id']: r for r in solver['comparisons']}
    rows = []
    for req in requests['rows']:
        r = old[req['id']]
        methods = {}
        for m in METHODS:
            reuse = m if m in ('vanilla', 'weighted_pab') else ('weighted_pab' if m == 'equal_credit' and req['equal_reuses_weighted'] else None)
            if reuse:
                t = copy.deepcopy(r['methods'][reuse])
                t['provenance'] = {'kind': 'baseline_reuse', 'source_id': r['id'],
                                   'source_method': reuse, 'source_solver_sha256': EXPECTED['solver']}
                if t['request'] != req['payloads'][m]:
                    raise ValueError('Reuse requires entire payload equality')
            else:
                t = stage()
                t.update(request=copy.deepcopy(req['payloads'][m]), provenance={'kind': 'new_generation'})
            methods[m] = t
        rows.append({'id': r['id'], 'level': r['level'], 'answer': r['answer'], 'methods': methods})
    return {'schema_version': 2, 'config': config, 'rows': rows}


def exclusive_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def prepare():
    if RUN.exists():
        raise FileExistsError('Prepared run directory already exists; use check/run --resume. No overwrite.')
    manifest, planner, solver, paths, hashes = sources()
    selection = select(manifest)
    requests = make_requests(selection, solver)
    audit = audit_baseline(solver)
    if audit['mismatches']:
        raise ValueError('CPU evaluator differs from frozen scores; preparation stopped')
    protocol = {'source_paths': {k: str(p.relative_to(ROOT)) for k, p in paths.items()},
                'source_sha256': hashes, 'dependencies': dependencies(),
                'tokenizer_files': tokenizer_info(), 'snapshot': str(SNAPSHOT),
                'evaluator': evaluator_info('original'), 'generation_packages': solver['config']['generation_packages'],
                'runtime': {'model': MODEL, 'revision': REVISION, 'seed': 42, 'temperature': .1,
                            'top_p': .9, 'max_tokens': 8192, 'max_model_len': 12288,
                            'tensor_parallel_size': 4, 'dtype': 'float16', 'max_num_seqs': 1,
                            'gpu_memory_utilization': .85},
                'retry_policy': {'retries': 2, 'retry_wait': 2, 'timeout': 600,
                                 'resume_unknown': 'Require explicit --retry-unknown; never silently replay ambiguous requests.'},
                'cost_policy': 'Vanilla=S; Plan-only/Equal=D+S; Weighted=D+A+S. '
                               'Both successful and all-attempt costs. Shared baselines are not new GPU expense.',
                'selection_fingerprint': digest(selection), 'requests_fingerprint': digest(requests)}
    RUN.mkdir(parents=True, exist_ok=False)
    for name, value in [('selection.json', selection), ('requests.json', requests),
                        ('audit_baseline.json', audit), ('protocol.json', protocol)]:
        exclusive_json(RUN / name, value)
    config = {'protocol_sha256': file_hash(RUN / 'protocol.json'),
              'immutable_sha256': {name: file_hash(RUN / name) for name in IMMUTABLE}}
    exclusive_json(RUN / 'checkpoint.json', initialize(selection, requests, solver, config))
    print(json.dumps({'run': str(RUN), 'selected': len(selection['ids']),
                      'equal_reuse': requests['equal_reuse_count'], 'new_solver_calls': requests['new_solver_calls'],
                      'baseline_audit_rows': audit['count'], 'audit_mismatches': 0}, ensure_ascii=False))


def load_prepared():
    _, planner, solver, _, hashes = sources()
    protocol = load_json(RUN / 'protocol.json')
    if protocol['source_sha256'] != hashes or protocol['dependencies'] != dependencies():
        raise ValueError('Frozen sources or execution code changed')
    if protocol['tokenizer_files'] != tokenizer_info():
        raise ValueError('Tokenizer/config snapshot changed')
    current = {k: v for k, v in versions().items() if k not in ('math-verify', 'sympy')}
    if current != protocol['generation_packages']:
        raise ValueError('Generation package versions changed')
    selection, reqs = load_json(RUN / 'selection.json'), load_json(RUN / 'requests.json')
    if digest(selection) != protocol['selection_fingerprint'] or digest(reqs) != protocol['requests_fingerprint']:
        raise ValueError('Selection/request manifest changed')
    if select(read_manifest(BASE / 'math500_manifest.json')) != selection:
        # JSON converts integer dictionary keys to strings.
        if json.loads(json.dumps(select(read_manifest(BASE / 'math500_manifest.json')))) != selection:
            raise ValueError('Pilot selection is not reproducible')
    if make_requests(selection, solver) != reqs:
        raise ValueError('Prompt equivalence/reuse reconstruction failed')
    expected_config = {'protocol_sha256': file_hash(RUN / 'protocol.json'),
                       'immutable_sha256': {name: file_hash(RUN / name) for name in IMMUTABLE}}
    # Never silently recover an uncommitted request whose acceptance is unknown.
    if (RUN / 'checkpoint.json.tmp').exists():
        raise ValueError('Uncommitted checkpoint.json.tmp exists. Inspect/recover it on CPU before execution; '
                         'do not silently replay the older checkpoint.')
    cp = load_json(RUN / 'checkpoint.json')
    if cp['config'] != expected_config:
        raise ValueError('Checkpoint binding changed')
    validate_checkpoint(cp, reqs, solver)
    return cp, reqs, planner, solver


def validate_checkpoint(cp, requests, solver):
    old = {r['id']: r for r in solver['comparisons']}
    if [r['id'] for r in cp['rows']] != [r['id'] for r in requests['rows']]:
        raise ValueError('Checkpoint IDs changed')
    for row, req in zip(cp['rows'], requests['rows']):
        original = old[row['id']]
        if row['answer'] != original['answer'] or row['level'] != original['level'] or set(row['methods']) != set(METHODS):
            raise ValueError('Checkpoint row mapping changed')
        for m, t in row['methods'].items():
            if t['request'] != req['payloads'][m]:
                raise ValueError('Checkpoint request changed')
            expected_reuse = m if m in ('vanilla', 'weighted_pab') else ('weighted_pab' if m == 'equal_credit' and req['equal_reuses_weighted'] else None)
            if expected_reuse:
                comparison = copy.deepcopy(t)
                provenance = comparison.pop('provenance')
                if comparison != original['methods'][expected_reuse] or provenance != {
                        'kind': 'baseline_reuse', 'source_id': row['id'], 'source_method': expected_reuse,
                        'source_solver_sha256': EXPECTED['solver']}:
                    raise ValueError('Reused baseline changed')
            elif t['provenance'] != {'kind': 'new_generation'}:
                raise ValueError('New generation provenance changed')
            for a in t['attempts']:
                if a['request_fingerprint'] != digest(req['payloads'][m]):
                    raise ValueError('Attempt belongs to a different request')
            if t['status'] == 'success':
                raw = t['result']['raw_response']
                if raw['model'] != MODEL or t['attempts'][-1]['raw_response'] != raw:
                    raise ValueError('Successful response provenance changed')
                for k in ('prompt_tokens', 'completion_tokens'):
                    if raw['usage'][k] != t['result']['usage'][k]:
                        raise ValueError('Usage changed')
                if t.get('evaluation_fingerprint', EVAL) != EVAL:
                    raise ValueError('Evaluator changed')


def preflight(cp, requests):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    maximum = 0
    for req in requests['rows']:
        for m, value in req['payloads'].items():
            n = len(tokenizer.apply_chat_template(value['messages'], tokenize=True, add_generation_prompt=True))
            maximum = max(maximum, n + 8192)
            if n + 8192 > 12288:
                raise ValueError('Context limit: ' + req['id'] + '/' + m)
    pending = sum(t['status'] != 'success' for r in cp['rows'] for m, t in r['methods'].items() if m in NEW)
    print(json.dumps({'preflight': 'passed', 'maximum_input_plus_output': maximum,
                      'pending_new_conditions': pending, 'equal_reuse': requests['equal_reuse_count'],
                      'generation_packages': versions()}, ensure_ascii=False))


def needs_unknown_approval(cp):
    return [(r['id'], m) for r in cp['rows'] for m, t in r['methods'].items()
            if m in NEW and t['status'] != 'success' and any(
                a.get('cost_unknown') or a['status'] in ('in_flight', 'interrupted') for a in t['attempts'])]


def check_resume(cp, resume, retry_unknown):
    history = any(t['attempts'] for r in cp['rows'] for m, t in r['methods'].items()
                  if m in NEW and t['provenance']['kind'] == 'new_generation')
    if history and not resume:
        raise ValueError('Checkpoint has execution history; --resume is required')
    if needs_unknown_approval(cp) and not retry_unknown:
        raise ValueError('Ambiguous API attempt: inspect checkpoint and explicitly acknowledge --retry-unknown')


def evaluate_new(row, method):
    t = row['methods'][method]
    result = t['result']
    t['evaluation'] = evaluate_answer(result['content'], row['answer'], result['finish_reason'], mode='original')
    t['evaluation_fingerprint'] = EVAL


def execute(cp, client, save, retry_unknown=False, server_pid=None):
    unknown = needs_unknown_approval(cp)
    if unknown and not retry_unknown:
        raise ValueError('Ambiguous previous API attempts; inspect checkpoint and use --retry-unknown explicitly: ' + str(unknown))
    if unknown:
        cp.setdefault('unknown_retry_acknowledgements', []).append(unknown)
        save()
    for index, row in enumerate(cp['rows']):
        order = NEW if index % 2 == 0 else NEW[::-1]
        for method in order:
            t = row['methods'][method]
            if t['status'] != 'success':
                run_stage(t, t['request'], client, save, lambda result: {},
                          retries=2, retry_wait=2, server_pid=server_pid)
            if t['status'] == 'success' and 'evaluation' not in t:
                evaluate_new(row, method)
                save()
        print(row['id'], {m: row['methods'][m]['status'] for m in NEW}, flush=True)
    # An undetermined evaluation is reported, not mistaken for generation failure.
    return all(t['status'] == 'success' for r in cp['rows'] for t in r['methods'].values())


def sum_usage(stages, all_attempts):
    usages = ([a['usage'] for t in stages for a in t['attempts']] if all_attempts
              else [t['result']['usage'] if t['status'] == 'success' else {} for t in stages])
    result = {'calls': len(usages)}
    for k in ('prompt_tokens', 'completion_tokens', 'latency_seconds'):
        vals = [u.get(k) for u in usages]
        known = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        result[k] = {'known_sum': round(sum(known), 6), 'missing_calls': len(vals) - len(known)}
    pairs = [u['prompt_tokens'] + u['completion_tokens'] for u in usages
             if isinstance(u.get('prompt_tokens'), int) and isinstance(u.get('completion_tokens'), int)]
    result['total_tokens'] = {'known_sum': sum(pairs), 'missing_calls': len(usages) - len(pairs)}
    return result


def planner_components(plan, method):
    return ([] if method == 'vanilla' else [plan['stages']['decomposition']] +
            ([plan['stages']['assessment']] if method == 'weighted_pab' else []))


def analyze(cp, planner):
    plans = {r['id']: r for r in planner['stats']}
    per_problem, audit, summary = [], [], {}
    n = len(cp['rows'])
    for method in METHODS:
        stages, successful = [], []
        transitions = {}
        for row in cp['rows']:
            t = row['methods'][method]
            components = planner_components(plans[row['id']], method)
            stages.extend(components + [t])
            ok = t['status'] == 'success' and 'evaluation' in t
            if ok:
                successful.append(t)
            v = row['methods']['vanilla']
            ev = t.get('evaluation', {'status': 'undetermined', 'reason': 'generation_incomplete'})
            transition = v['evaluation']['status'] + '->' + ev['status']
            transitions[transition] = transitions.get(transition, 0) + 1
            final = sum_usage(components + [t], False)
            attempts = sum_usage(components + [t], True)
            v_final = sum_usage([v], False)
            v_all = sum_usage([v], True)
            metrics = {'id': row['id'], 'level': row['level'], 'method': method,
                       'generation_status': t['status'], 'evaluation': ev['status'],
                       'transition_from_vanilla': transition, 'provenance': t['provenance']['kind']}
            for scope, costs in [('final', final), ('all_attempts', attempts)]:
                for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                    metrics[scope + '_' + k] = costs[k]['known_sum'] if not costs[k]['missing_calls'] else None
                    metrics[scope + '_' + k + '_known'] = costs[k]['known_sum']
                    metrics[scope + '_' + k + '_missing'] = costs[k]['missing_calls']
                baseline = v_final if scope == 'final' else v_all
                metrics[scope + '_paired_completion_delta'] = (
                    costs['completion_tokens']['known_sum'] - baseline['completion_tokens']['known_sum']
                    if not costs['completion_tokens']['missing_calls'] and not baseline['completion_tokens']['missing_calls'] else None)
            metrics['solver_completion_tokens'] = t.get('result', {}).get('usage', {}).get('completion_tokens')
            metrics['solver_paired_completion_delta'] = (metrics['solver_completion_tokens'] - v['result']['usage']['completion_tokens'] if ok else None)
            for name in ('decomposition', 'assessment'):
                included = method != 'vanilla' and (name == 'decomposition' or method == 'weighted_pab')
                ts = [plans[row['id']]['stages'][name]] if included else []
                for scope, all_attempts in [('final', False), ('all_attempts', True)]:
                    metrics[scope + '_' + name + '_completion_tokens'] = sum_usage(ts, all_attempts)['completion_tokens']['known_sum']
            metrics['finish_reason'] = t.get('result', {}).get('finish_reason')
            per_problem.append(metrics)
            if ok:
                audit.append({'id': row['id'], 'method': method, 'official': ev,
                              'diagnostic': diagnostic(t['result']['content'], ev)})
        entry = {'denominator': n, 'api_success': len(successful),
                 'correct': sum(t['evaluation']['status'] == 'correct' for t in successful),
                 'undetermined_or_unfinished': sum(r['methods'][method].get('evaluation', {}).get('status') not in ('correct', 'incorrect') for r in cp['rows']),
                 'cap_reached': sum(t['result']['finish_reason'] == 'length' for t in successful),
                 'transitions_from_vanilla': transitions}
        entry['accuracy_all_selected'] = entry['correct'] / n
        entry['cap_rate_all_selected'] = entry['cap_reached'] / n
        entry['cap_rate_api_success'] = entry['cap_reached'] / len(successful) if successful else None
        for scope, all_attempts in [('final_pipeline', False), ('all_attempt_pipeline', True)]:
            costs = sum_usage(stages, all_attempts)
            for key in ('prompt_tokens', 'completion_tokens', 'total_tokens', 'latency_seconds'):
                costs[key]['denominator'] = n
                costs[key]['mean_known_lower_bound'] = costs[key]['known_sum'] / n
                costs[key]['mean'] = costs[key]['known_sum'] / n if not costs[key]['missing_calls'] else None
            entry[scope] = costs
        entry['new_execution_only'] = sum_usage([r['methods'][method] for r in cp['rows']
                                                if r['methods'][method]['provenance']['kind'] == 'new_generation'], True)
        subset = [r for r in per_problem if r['method'] == method]
        for key in ('solver_completion_tokens', 'final_decomposition_completion_tokens',
                    'final_assessment_completion_tokens', 'all_attempts_decomposition_completion_tokens',
                    'all_attempts_assessment_completion_tokens'):
            values = [r[key] for r in subset]
            entry[key] = {'denominator': n, 'mean': sum(values) / n if all(v is not None for v in values) else None}
        for key in ('final_paired_completion_delta', 'all_attempts_paired_completion_delta', 'solver_paired_completion_delta'):
            values = [r[key] for r in subset if r[key] is not None]
            entry[key] = {'paired_n': len(values), 'sum': sum(values),
                          'mean': sum(values) / len(values) if values else None}
        summary[method] = entry
    return ({'methods': summary, 'primary_metric': 'all_attempt_pipeline.completion_tokens.mean',
             'note': 'Unknown costs are not zero; mean is null when missing. '
                     'Incomplete runs must not be interpreted as completed accuracy. '
                     'Pipeline time is a sum of separate API latencies, not end-to-end wall time.'},
            per_problem, {'evaluator_fingerprint': EVAL, 'rows': audit})


def report(cp, planner):
    summary, rows, audit = analyze(cp, planner)
    cp_hash = file_hash(RUN / 'checkpoint.json')
    summary['checkpoint_sha256'] = cp_hash
    audit['checkpoint_sha256'] = cp_hash
    atomic_save(RUN / 'metrics.json', summary)
    atomic_save(RUN / 'audit_current.json', audit)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    temp = RUN / 'per_problem.csv.tmp'
    temp.write_text(stream.getvalue(), encoding='utf-8')
    temp.replace(RUN / 'per_problem.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'check', 'run', 'analyze'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-unknown', action='store_true')
    parser.add_argument('--api-base', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--server-pid', type=int)
    parser.add_argument('--require-pending', action='store_true', help='check exits 10 when no API work remains')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
        return 0
    if args.mode == 'check':
        cp, reqs, _, _ = load_prepared()
        check_resume(cp, args.resume, args.retry_unknown)
        preflight(cp, reqs)
        if args.require_pending and all(r['methods'][m]['status'] == 'success' for r in cp['rows'] for m in NEW):
            print('All generation already complete; no server is needed.')
            return 10
        return 0
    with lock(RUN / 'checkpoint.json'):
        cp, reqs, planner, _ = load_prepared()
        if args.mode == 'analyze':
            report(cp, planner)
            return 0
        from urllib.parse import urlparse
        url = urlparse(args.api_base)
        if url.scheme != 'http' or url.hostname != '127.0.0.1' or url.path != '/v1' or url.username or url.password or url.query or url.fragment:
            raise ValueError('Only the local pinned server is supported')
        check_resume(cp, args.resume, args.retry_unknown)
        preflight(cp, reqs)
        def save():
            atomic_save(RUN / 'checkpoint.json', cp)
        if all(r['methods'][m]['status'] == 'success' for r in cp['rows'] for m in NEW):
            for row in cp['rows']:
                for method in NEW:
                    if 'evaluation' not in row['methods'][method]:
                        evaluate_new(row, method)
            save()
            report(cp, planner)
            return 0
        from openai import OpenAI
        with termination_handler():
            try:
                with OpenAI(base_url=args.api_base, api_key='DUMMY', timeout=600, max_retries=0) as client:
                    complete = execute(cp, client, save, args.retry_unknown, args.server_pid)
            except StopRequested as exc:
                save()
                return 128 + exc.signum
            except ServerDead as exc:
                cp['last_stop'] = str(exc)
                save()
                return 3
        report(cp, planner)
        return 0 if complete else 2


if __name__ == '__main__':
    raise SystemExit(main())
