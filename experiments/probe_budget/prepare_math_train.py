#!/usr/bin/env python3
"""Task 1 (1-3): load MATH train split, remove MATH-500 overlap, sample 400/level.

Output: data_v2/math_train_sample2000.json  (+ data_v2/prepare_report.json)
"""
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset, concatenate_datasets

REPO = 'EleutherAI/hendrycks_math'
CONFIGS = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra',
           'number_theory', 'prealgebra', 'precalculus']
PER_LEVEL = 400
SEED = 42
OUT = Path('data_v2')


def parse_level(s):
    try:
        return int(str(s).replace('Level', '').strip())
    except ValueError:
        return None


def main():
    OUT.mkdir(exist_ok=True)
    parts = []
    for c in CONFIGS:
        d = load_dataset(REPO, c, split='train')
        print(f'{c:>26}: {len(d)} rows, columns={d.column_names}', flush=True)
        parts.append(d)
    train = concatenate_datasets(parts)
    print(f'TOTAL train rows: {len(train)}')
    print('example row:', {k: (str(v)[:80]) for k, v in train[0].items()})
    raw_levels = Counter(train['level'])
    print('raw level values:', dict(sorted(raw_levels.items())))
    print('type values:', dict(sorted(Counter(train['type']).items())))

    m500 = load_dataset('HuggingFaceH4/MATH-500', split='test')
    m500_exact = set(m500['problem'])
    m500_strip = {' '.join(p.split()) for p in m500['problem']}

    rows, n_exact, n_strip_only, n_badlevel, n_dup_in_train = [], 0, 0, 0, 0
    seen = set()
    for i, r in enumerate(train):
        p = r['problem']
        if p in m500_exact:
            n_exact += 1
            continue
        if ' '.join(p.split()) in m500_strip:
            n_strip_only += 1   # reported; also excluded (conservative)
            continue
        lv = parse_level(r['level'])
        if lv not in (1, 2, 3, 4, 5):
            n_badlevel += 1
            continue
        if p in seen:
            n_dup_in_train += 1
            continue
        seen.add(p)
        rows.append({'train_idx': i, 'problem': p, 'level': lv, 'type': r['type'],
                     'solution': r['solution']})

    print(f'overlap with MATH-500 (exact text): {n_exact}')
    print(f'overlap after whitespace-normalisation only (also excluded): {n_strip_only}')
    print(f'rows with unparseable level (e.g. "Level ?") excluded: {n_badlevel}')
    print(f'duplicate problems inside train excluded: {n_dup_in_train}')
    by_level = {lv: [r for r in rows if r['level'] == lv] for lv in range(1, 6)}
    print('eligible per level:', {lv: len(v) for lv, v in by_level.items()})

    rng = random.Random(SEED)
    sample = []
    for lv in range(1, 6):
        pool = sorted(by_level[lv], key=lambda r: r['train_idx'])
        sample += rng.sample(pool, PER_LEVEL)
    for pid, r in enumerate(sample):
        r['problem_id'] = pid
    print('sampled per level:', dict(sorted(Counter(r['level'] for r in sample).items())))
    print('sampled per type:', dict(sorted(Counter(r['type'] for r in sample).items())))

    (OUT / 'math_train_sample2000.json').write_text(
        json.dumps(sample, ensure_ascii=False, indent=1), encoding='utf-8')
    (OUT / 'prepare_report.json').write_text(json.dumps({
        'repo': REPO, 'configs': CONFIGS, 'n_train_total': len(train),
        'raw_levels': {str(k): v for k, v in raw_levels.items()},
        'overlap_exact': n_exact, 'overlap_ws_only': n_strip_only,
        'bad_level': n_badlevel, 'dup_in_train': n_dup_in_train,
        'eligible_per_level': {lv: len(v) for lv, v in by_level.items()},
        'seed': SEED, 'per_level': PER_LEVEL,
    }, indent=1), encoding='utf-8')
    print(f'saved {len(sample)} problems -> {OUT}/math_train_sample2000.json')


if __name__ == '__main__':
    main()
