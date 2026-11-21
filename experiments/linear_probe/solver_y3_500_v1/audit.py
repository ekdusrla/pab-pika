"""Read-only provenance audit and sealed candidate manifest."""
import argparse
import importlib.metadata
from collections import Counter
from itertools import combinations
from pathlib import Path

from ..common import ROOT, digest, file_hash, problem_hash, read, read_manifest, save_new, seal, verify
from ..pilot_data import relaxed, similarity
from ..pilot import SOLVER, SOLVER_REVISION
from ..common import LEVEL_WORDS, MODEL, REVISION
from math_answer_eval import evaluator_info, evaluate_answer

RUNS = ROOT / 'experiments/linear_probe/runs'
OUT = RUNS / 'solver_y3_500_v1'
LEGACY = ROOT / 'experiments/llama_qwq'
CREDIT = RUNS / 'credit_a2000_tp4'
DATA = RUNS / 'pilot_v1/data.json'
PROBE = CREDIT / 'probe_v1'
PILOT = RUNS / 'solver_y1_v1/audit_221040.json'
HUB = Path('/data/dada0228/huggingface/hub')
ARMS = ('vanilla', 'teacher', 'uniform', 'position', 'ridge')
SEED = 42


def model_identity(model, revision):
    snapshot = HUB / ('models--' + model.replace('/', '--')) / 'snapshots' / revision
    metadata = {}
    for file in sorted(snapshot.iterdir()):
        if file.name in ('config.json', 'generation_config.json', 'model.safetensors.index.json',
                         'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja',
                         'vocab.json', 'merges.txt'):
            metadata[file.name] = file_hash(file)
    index = read(snapshot / 'model.safetensors.index.json')
    shards = {name: {'blob': (snapshot / name).resolve(strict=True).name,
                     'bytes': (snapshot / name).stat().st_size}
              for name in sorted(set(index['weight_map'].values()))}
    return dict(model=model, revision=revision, snapshot=str(snapshot), metadata_sha256=metadata, shards=shards)


def pairs(rows, other=None):
    candidates = combinations(rows, 2) if other is None else ((a, b) for a in rows for b in other)
    matches = []
    maximum = 0
    for a, b in candidates:
        x, y = a['query'], b['query']
        s = similarity(x, y)
        maximum = max(maximum, s)
        exact = problem_hash(x) == problem_hash(y)
        if exact or s >= .90 or relaxed(x) == relaxed(y):
            matches.append(dict(a=a['id'], b=b['id'], normalized_match=exact, jaccard=s))
    return dict(max_jaccard=maximum, matches=matches)


def probe_metadata():
    meta = verify(read(PROBE / 'metadata.json'))
    assert verify(read(PROBE / 'status.json'))['passed'] and meta['selection_frozen']
    assert file_hash(PROBE / 'probe.joblib') == meta['artifact_sha256']
    return meta


def audit():
    candidate = read_manifest(LEGACY / 'math500_manifest.json')
    source = ROOT / 'dataset/MATH-500/decomposed_questions.json'
    assert file_hash(source) == candidate['source_sha256']
    data = verify(read(DATA))
    audit_y1 = verify(read(PILOT))
    meta = probe_metadata()
    pinned = {str(path): file_hash(path) for path in (source, LEGACY / 'math500_manifest.json',
               DATA, PILOT, PROBE / 'metadata.json', PROBE / 'probe.joblib')}
    assert data['fingerprint'] == meta['data_fingerprint']
    for split in ('train', 'validation'):
        path = CREDIT / f'{split}_labeled.json'
        label = verify(read(path))
        assert label['fingerprint'] == meta[f'{split}_fingerprint']
        assert Counter(r['problem_hash'] for r in data['splits'][split]) == Counter(meta[f'{split}_problem_hashes'])
        pinned[str(path)] = file_hash(path)
    assert {r['id'] for r in audit_y1['solver_requests']} == {r['id'] for r in data['splits']['evaluation']}
    rows = candidate['records']
    screens = {name: pairs(rows, group) for name, group in
               {**data['splits'], 'benchmarks': data['benchmarks']}.items()}
    screens['within_math500'] = pairs(rows)
    seen = set()
    history = []
    by_id = {r['id']: problem_hash(r['query']) for r in rows}
    for path in sorted((LEGACY / 'runs').glob('*/solver.json')):
        old = read(path)
        hits = set()
        for r in old['comparisons']:
            identifier = str(r['id'])
            if problem_hash(r['query']) == by_id.get(identifier):
                hits.add(identifier)
        planner = path.with_name('planner.json')
        benchmark_ids = read(planner)['config']['benchmark_ids']
        pinned[str(planner)] = file_hash(planner)
        history.append(dict(path=str(path.relative_to(ROOT)), matched=len(hits),
                            planner_path=str(planner.relative_to(ROOT)), benchmark_ids=benchmark_ids))
        seen.update(hits)
        pinned[str(path)] = file_hash(path)
    path = LEGACY / 'runs/math500_llama31_8b_ablation_pilot_v1/checkpoint.json'
    ablations = read(path)
    hits = {str(r['id']) for r in ablations['rows']} & by_id.keys()
    seen.update(hits)
    history.append(dict(path=str(path.relative_to(ROOT)), matched=len(hits)))
    pinned[str(path)] = file_hash(path)
    assert evaluator_info('original') == audit_y1['evaluator']
    reference_failures = [r['id'] for r in rows if evaluate_answer(
        'Answer: $\\boxed{' + r['answer'] + '}$.', r['answer'], 'stop', 'success')['status'] != 'correct']
    return seal(dict(kind='solver_y3_500_candidate_v1', approved=False,
        blockers=['Prior project Solver results cover this candidate; strict untouched set unavailable',
                  'No user-approved substitute or prior-use scope'],
        source=candidate['dataset'], split='MATH-500 existing evaluation',
        source_sha256=candidate['source_sha256'], source_fingerprint=candidate['fingerprint'],
        input_sha256=pinned, count=len(rows), rows=[dict(r, problem_hash=by_id[r['id']], chunk=i//50+1)
                                                    for i, r in enumerate(rows)],
        levels=dict(Counter(r['level'] for r in rows)), seed=SEED, arms=ARMS, screens=screens,
        previous_experiments=history, previously_evaluated_count=len(seen),
        untouched_count=len(rows)-len(seen), reference_self_check_failures=reference_failures,
        model_identity={'planner': model_identity(MODEL, REVISION), 'solver': model_identity(SOLVER, SOLVER_REVISION)},
        tokenizer_package_versions={name: importlib.metadata.version(name) for name in ('transformers', 'tokenizers')},
        evaluator=audit_y1['evaluator'], probe_fingerprint=meta['fingerprint'],
        probe_settings={'layer':meta['best_layer_idx'],'alpha':meta['alpha'],
                        'dimension':meta['feature_dimension'],'feature_signature':meta['feature_signature']},
        budget_settings={'decay':'constant','level_words':LEVEL_WORDS},
        solver_sampling={k:audit_y1['legacy_solver_config'][k] for k in ('max_tokens','temperature','top_p','seed')},
        prior_margin_proposal_pp=3, approved_margin_pp=None))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=OUT / 'candidate_manifest.json')
    args = p.parse_args()
    if args.output.resolve().parent != OUT:
        p.error('Write only under the new experiment result directory')
    doc = audit()
    save_new(args.output, doc)
    print({k: doc[k] for k in ('approved', 'count', 'levels', 'previous_experiments',
                              'previously_evaluated_count', 'untouched_count', 'reference_self_check_failures')})
    print(doc['screens'])
    print('SHA256', file_hash(args.output))
