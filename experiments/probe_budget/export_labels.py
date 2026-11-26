"""Export the probe training labels and CV splits without the hidden states.

Reads data_v2/gen/problems/*.pt (the 1988 kept problems, ~4.5 GB with hidden states)
and writes small files to data_v2/labels/:
  problems.jsonl  one line per kept problem: ids, MATH level/type, steps, per-step
                  evaluated_levels (the probe target) and credits from PaB call 2
  cv_folds.json   outer GroupKFold(5) problem ids per fold, and for every fold the
                  inner GroupShuffleSplit(test_size=0.2) train/val problem ids for
                  seeds 0, 1, 2 (Ridge used seed 0; MLP used 0, 1, 2)
  versions.json   library versions used to rebuild the splits
The splits are rebuilt exactly as train_probe_mlp.py builds them, and the position-mean
baseline is recomputed on them and checked against data_v2/probe_report/baselines.json.

Run from this folder on a compute node:
  srun --ntasks=1 --cpus-per-task=1 --mem=8G --time=00:30:00 python export_labels.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

import train_probe_mlp as tp

HERE = Path(__file__).resolve().parent
OUT = HERE / 'data_v2' / 'labels'
SEEDS = [0, 1, 2]
KEYS = ['problem_id', 'train_idx', 'level', 'type', 'evaluated_levels', 'credits',
        'problem_evaluated_level', 'label_flags', 'last_step_uses_h_end']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(tp.DATA.glob('*.pt'), key=lambda p: int(p.stem))
    recs = []
    with (OUT / 'problems.jsonl').open('w', encoding='utf-8') as f:
        for p in files:
            r = torch.load(p, weights_only=False)
            row = {k: r[k] for k in KEYS}
            row['n_steps'] = len(r['evaluated_levels'])
            row['steps'] = r['steps']
            f.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
            recs.append({'pid': r['problem_id'], 'level': r['level'],
                         'y': np.array(r['evaluated_levels'], float)})

    pid, k, n, y = tp.flat(recs)
    print(f'{len(recs)} problems, {len(y)} steps')

    # outer folds: identical call to train_probe_mlp.main
    folds = list(GroupKFold(5).split(np.zeros(len(y)), y, pid))
    prob_ids = np.array([r['pid'] for r in recs])
    out = {'outer': [], 'note': 'problem_id lists; outer = GroupKFold(5) over steps grouped by '
                                 'problem_id, problems sorted by id; inner = GroupShuffleSplit('
                                 'n_splits=1, test_size=0.2, random_state=seed) on the outer-train problems'}
    for fi, (tr, te) in enumerate(folds):
        tr_p, te_p = set(np.unique(pid[tr])), set(np.unique(pid[te]))
        tr_r = [i for i, p in enumerate(prob_ids) if p in tr_p]
        recs_tr = [recs[i] for i in tr_r]
        pid_tr = np.concatenate([[r['pid']] * len(r['y']) for r in recs_tr])
        y_tr = np.concatenate([r['y'] for r in recs_tr])
        inner = {}
        for s in SEEDS:
            a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=s)
                        .split(np.zeros(len(y_tr)), y_tr, pid_tr))
            inner[f'seed{s}'] = {'train': sorted(int(x) for x in np.unique(pid_tr[a])),
                                 'val': sorted(int(x) for x in np.unique(pid_tr[b]))}
        out['outer'].append({'fold': fi,
                             'train': sorted(int(x) for x in tr_p),
                             'test': sorted(int(x) for x in te_p),
                             'inner': inner})
    (OUT / 'cv_folds.json').write_text(json.dumps(out, indent=1))
    (OUT / 'versions.json').write_text(json.dumps({
        'python': sys.version.split()[0], 'numpy': np.__version__,
        'scikit-learn': sklearn.__version__, 'torch': torch.__version__}, indent=1))

    # sanity check: the committed baselines must come out the same on these folds
    base = tp.run_baselines(pid, k, n, y, folds)
    ref = json.loads((tp.REPORT / 'baselines.json').read_text())
    for name, m in ref.items():
        for kk, vv in m.items():
            if isinstance(vv, float):
                got = base[name][kk]
                assert (np.isnan(vv) and np.isnan(got)) or abs(got - vv) < 1e-9, (name, kk, got, vv)
    print('baselines.json reproduced on exported folds: OK')


if __name__ == '__main__':
    main()
