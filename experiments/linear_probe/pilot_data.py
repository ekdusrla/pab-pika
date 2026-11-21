"""Pinned external MATH train download and deterministic 24/6/5 pilot selection."""
import argparse
from functools import lru_cache
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

from .common import (MODEL, REVISION, ROOT, read, read_manifest, problem_hash, seal,
                     save_new, verify, validate_data, file_hash, digest, output_path)
from math_answer_eval import extract_boxed

DATASET = 'EleutherAI/hendrycks_math'
DATA_REVISION = '21a5633873b6a120296cce3e2df9d5550074f4a3'
SUBJECTS = ('algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra',
            'number_theory', 'prealgebra', 'precalculus')
SEED = 20261113


def fetch(output):
    """Network only here, CPU only; exclusive files under new experiment runs."""
    output = output_path(output)
    output.mkdir(parents=True, exist_ok=False)
    files = []
    for name in ('README.md', *[f'{s}/train-00000-of-00001.parquet' for s in SUBJECTS]):
        url = f'https://huggingface.co/datasets/{DATASET}/resolve/{DATA_REVISION}/{name}'
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as response, target.open('xb') as stream:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                stream.write(block)
        files.append({'path': name, 'url': url, 'sha256': file_hash(target)})
    save_new(output / 'source.json', seal({'dataset': DATASET, 'revision': DATA_REVISION,
        'split': 'train', 'files': files, 'policy': 'No test split downloaded'}))


def load_rows(source_dir):
    import pyarrow.parquet as pq
    provenance = verify(read(source_dir / 'source.json'))
    if provenance['dataset'] != DATASET or provenance['revision'] != DATA_REVISION or provenance['split'] != 'train':
        raise ValueError('Unexpected external source')
    rows = []
    for item in provenance['files']:
        path = source_dir / item['path']
        if file_hash(path) != item['sha256']:
            raise ValueError('External source changed')
        if path.suffix != '.parquet':
            continue
        for index, item_row in enumerate(pq.read_table(path).to_pylist()):
            match = re.fullmatch(r'Level ([1-5])', item_row['level'])
            level = int(match.group(1)) if match else None
            solution = item_row['solution']
            answer = extract_boxed(solution)
            rows.append({'id': f"{DATASET}@{DATA_REVISION}/{path.parent.name}/train/{index}",
                         'query': item_row['problem'], 'level': level, 'solution': solution,
                         'answer': answer, 'subject': path.parent.name, 'source_row_index': index,
                         'native_level': item_row['level'],
                         'source_file': str(path.relative_to(source_dir)),
                         'problem_hash': problem_hash(item_row['problem'])})
    return rows, provenance


def relaxed(text):
    text = re.sub(r'\\(?:left|right|displaystyle|textstyle)\b', '', text)
    return re.sub(r'[^a-z0-9]', '', text.casefold().replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac'))


@lru_cache(maxsize=20000)
def grams(text):
    text = relaxed(text)
    return frozenset(text[i:i+5] for i in range(max(1,len(text)-4)))


def similarity(a, b):
    aa, bb = grams(a), grams(b)
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def select(rows, final_manifest, source_provenance, seed=SEED):
    """Selection independent of Teacher/Probe/Solver outputs, stratified by level."""
    forbidden = {problem_hash(r['query']) for r in final_manifest['records']}
    seen, clean, rejected = set(), [], []
    for row in sorted(rows, key=lambda r: r['id']):
        h = problem_hash(row['query'])
        reason = ('invalid_native_level' if row.get('level') not in range(1,6) else
                  'math500_normalized_overlap' if h in forbidden else
                  'source_normalized_duplicate' if h in seen else
                  'missing_boxed_answer' if not row.get('answer') else None)
        seen.add(h)
        if reason:
            rejected.append({'id': row['id'], 'reason': reason})
        else:
            clean.append({**row, 'problem_hash': h})
    # Every role gets its own problems. Pilot/benchmarks cover all native levels.
    ranked = sorted(clean, key=lambda r: (digest([seed, r['problem_hash']]), r['id']))
    bins = {level: [r for r in ranked if r['level'] == level] for level in range(1, 6)}
    selected = {'benchmarks': [], 'evaluation': [], 'validation': [], 'train': []}
    near_rejected = []
    used = []
    # 5 benchmarks + 5 pilot + 6 validation + 24 train = 40 independent problems.
    schedule = [('benchmarks', i) for i in range(1,6)] + [('evaluation', i) for i in range(1,6)]
    schedule += [('validation', i) for i in range(1,6)] + [('validation', seed % 5 + 1)]
    schedule += [('train', i % 5 + 1) for i in range(24)]
    for role, level in schedule:
        while bins[level]:
            candidate = bins[level].pop(0)
            comparisons = [*final_manifest['records'], *used]
            # A second conservative screen catches common LaTeX-only changes and near copies.
            hit = next((other for other in comparisons if
                        relaxed(candidate['query']) == relaxed(other['query']) or
                        similarity(candidate['query'], other['query']) >= .90), None)
            if hit:
                near_rejected.append({'id': candidate['id'], 'other_id': hit['id'],
                                      'similarity': similarity(candidate['query'], hit['query'])})
                continue
            used.append(candidate); selected[role].append(candidate)
            break
        else:
            raise ValueError(f'Insufficient independent candidates at level {level}')
    # Observe the closest final-evaluation neighbour for all 40 selected problems.
    neighbours = []
    for row in used:
        score, other_id = max((similarity(row['query'], other['query']), other['id'])
                              for other in final_manifest['records'])
        neighbours.append({'id': row['id'], 'nearest_math500_id': other_id, 'similarity': score})
    doc = seal({'schema_version': 1, 'kind': 'linear_probe_pilot_data',
        'source_provenance': source_provenance, 'split_seed': seed,
        'evaluation_manifest_fingerprint': final_manifest['fingerprint'],
        'final_evaluation_policy': 'MATH-500 untouched, not used for training, validation, pilot or benchmarks',
        'benchmarks': selected.pop('benchmarks'), 'splits': selected,
        'selection_rule': 'Native level stratification, SHA256(seed,normalized question) rank; fixed role order',
        'overlap_policy': 'NFKC+casefold+whitespace removal; relaxed LaTeX equality and character-5gram Jaccard >= .90 screened',
        'audit': {'source_rows': len(rows), 'normalized_eligible_rows': len(clean),
                  'rejected': rejected, 'near_duplicate_rejected': near_rejected,
                  'nearest_final_neighbours': neighbours,
                  'selected_normalized_math500_overlap': 0,
                  'selected_relaxed_or_similarity90_math500_overlap': 0,
                  'semantic_duplicate_guarantee': False}})
    validate_pilot(doc)
    return doc


def validate_pilot(doc):
    validate_data(doc)
    if doc.get('kind') != 'linear_probe_pilot_data' or {k: len(v) for k,v in doc['splits'].items()} != {'train':24,'validation':6,'evaluation':5} or len(doc['benchmarks']) != 5:
        raise ValueError('Require the independent external 24/6/5 pilot and five benchmarks')
    rows = [*doc['benchmarks'], *[r for group in doc['splits'].values() for r in group]]
    if len({r['id'] for r in rows}) != 40 or any(not r.get('answer') for r in rows):
        raise ValueError('Missing reference answers or duplicate source IDs')
    return doc


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('fetch'); f.add_argument('--output', type=Path, required=True)
    s = sub.add_parser('select'); s.add_argument('--source', type=Path, required=True)
    s.add_argument('--eval-manifest', type=Path, required=True); s.add_argument('--output', type=Path, required=True)
    s.add_argument('--seed', type=int, default=SEED)
    s = sub.add_parser('audit'); s.add_argument('--source',type=Path,required=True)
    s.add_argument('--data',type=Path,required=True);s.add_argument('--eval-manifest',type=Path,required=True)
    s.add_argument('--output',type=Path,required=True)
    a = p.parse_args(argv)
    if a.command == 'fetch':
        fetch(a.output)
    elif a.command == 'select':
        rows, provenance = load_rows(a.source)
        save_new(a.output, select(rows, read_manifest(a.eval_manifest), provenance, a.seed))
    else:
        data=validate_pilot(read(a.data));rows,provenance=load_rows(a.source)
        final=read_manifest(a.eval_manifest)
        reproduced=select(rows,final,provenance,data['split_seed'])
        if reproduced != data:raise ValueError('Frozen Pilot is not reproducible from pinned external source')
        save_new(a.output,seal({'kind':'pilot_data_audit','data_fingerprint':data['fingerprint'],
            'final_manifest_fingerprint':final['fingerprint'],'source_rows':len(rows),
            'counts':{k:len(v) for k,v in data['splits'].items()},'benchmark_count':len(data['benchmarks']),
            'rejected_counts':dict(Counter(r['reason'] for r in data['audit']['rejected'])),
            'source_sha256_verified':True,'selection_reproduced':True,
            'selected_normalized_final_overlap':0,'selected_near90_final_overlap':0,
            'max_selected_to_final_char5gram_jaccard':max(r['similarity'] for r in data['audit']['nearest_final_neighbours']),
            'semantic_review_required':True}))


if __name__ == '__main__':
    main()
