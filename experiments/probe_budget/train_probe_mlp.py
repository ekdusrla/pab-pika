#!/usr/bin/env python3
"""Tasks 5-8: label diagnostics, Ridge/MLP difficulty probes, report, save.

Target: per-step evaluated_level (1..5) from data_v2/gen/problems/*.pt
Outer CV: GroupKFold(5) by problem_id. Inside each outer-train fold, a
problem-level inner split (20%) is used for layer + hyper-parameter selection
and MLP early stopping. Outer-test folds are never touched for selection.

Features (layer L):
  F1 h_k     F2 h_k - h_{k-1} (h_0 = h_prompt)     F3 [h_k ; h_end]     F4 [h_k - h_{k-1} ; h_end]

Usage:
  python train_probe_mlp.py --diag-only          # task 5
  python train_probe_mlp.py                      # tasks 6-8
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

HERE = Path(__file__).resolve().parent
import os
DATA = Path(os.environ.get('PROBE_DATA', HERE / 'data_v2' / 'gen' / 'problems'))
OUT = Path(os.environ.get('PROBE_OUT', HERE / 'probe_artifacts'))
REPORT = Path(os.environ.get('PROBE_REPORT', HERE / 'data_v2' / 'probe_report'))
CAND_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32]
FEATURES = ['F1', 'F2', 'F3', 'F4']
RIDGE_ALPHAS = [1e1, 1e2, 1e3, 1e4, 1e5]
MLP_GRID = [dict(hidden=h, dropout=d, wd=w) for h in (256,) for d in (0.1, 0.3) for w in (1e-4, 1e-2)]
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


# ------------------------------------------------------------------ data
def load(layers):
    recs = []
    for p in sorted(DATA.glob('*.pt'), key=lambda p: int(p.stem)):
        r = torch.load(p, weights_only=False)
        li = [r['layers'].index(l) for l in layers]
        recs.append({'pid': r['problem_id'], 'level': r['level'], 'y': np.array(r['evaluated_levels'], float),
                     'H': r['h_steps'][:, li].astype(np.float32),          # [n, L, d]
                     'h_end': r['h_end'][li].astype(np.float32),           # [L, d]
                     'h_prompt': r['h_prompt'][li].astype(np.float32)})
    return recs


def flat(recs):
    pid = np.concatenate([[r['pid']] * len(r['y']) for r in recs])
    k = np.concatenate([np.arange(len(r['y'])) for r in recs])
    n = np.concatenate([[len(r['y'])] * len(r['y']) for r in recs])
    y = np.concatenate([r['y'] for r in recs])
    return pid, k, n, y


def build_X(recs, li, feat):
    rows = []
    for r in recs:
        H = r['H'][:, li]
        prev = np.vstack([r['h_prompt'][li][None], H[:-1]])
        end = np.repeat(r['h_end'][li][None], len(H), 0)
        rows.append({'F1': H, 'F2': H - prev, 'F3': np.hstack([H, end]),
                     'F4': np.hstack([H - prev, end])}[feat])
    return np.vstack(rows)


# ------------------------------------------------------------------ metrics
def shares(pid, d):
    d = np.clip(np.asarray(d, float), 0.1, None)
    w = np.empty_like(d)
    for p in np.unique(pid):
        m = pid == p
        w[m] = d[m] / d[m].sum()
    return w


def metrics(pid, y, pred, level_metrics=True):
    out = {}
    if level_metrics:
        out['r2'] = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out['mae'] = np.abs(y - pred).mean()
    out['share_mae_pp'] = 100 * np.abs(shares(pid, y) - shares(pid, pred)).mean()
    rho, hit, n_var = [], [], 0
    for p in np.unique(pid):
        m = pid == p
        yt, pt = y[m], pred[m]
        if len(yt) < 2 or np.ptp(yt) == 0:
            continue
        n_var += 1
        rho.append(spearmanr(yt, pt).correlation if np.ptp(pt) > 0 else np.nan)
        hit.append(float(np.argmax(pt) in set(np.flatnonzero(yt == yt.max()))))
    rho = np.array(rho, float)
    out['spearman'] = float(np.nanmean(rho)) if np.isfinite(rho).any() else float('nan')
    out['spearman_defined_frac'] = float(np.isfinite(rho).mean()) if len(rho) else 0.0
    out['top1'] = float(np.mean(hit)) if hit else float('nan')
    out['n_problems_nonconst'] = n_var
    return out


# ------------------------------------------------------------------ baselines
def position_mean(tr_idx, te_idx, k, n, y):
    tab = defaultdict(list)
    for i in tr_idx:
        tab[(n[i], k[i])].append(y[i])
    rel = defaultdict(list)          # fallback: relative position bucket
    for i in tr_idx:
        rel[round(k[i] / max(1, n[i] - 1), 1)].append(y[i])
    g = y[tr_idx].mean()
    pr = []
    for i in te_idx:
        v = tab.get((n[i], k[i])) or rel.get(round(k[i] / max(1, n[i] - 1), 1))
        pr.append(np.mean(v) if v else g)
    return np.array(pr)


def run_baselines(pid, k, n, y, folds):
    res = {}
    g = np.empty_like(y); pm = np.empty_like(y)
    for tr, te in folds:
        g[te] = y[tr].mean()
        pm[te] = position_mean(tr, te, k, n, y)
    res['uniform'] = metrics(pid, y, np.ones_like(y), level_metrics=False)
    res['global_mean'] = metrics(pid, y, g)
    res['position_mean'] = metrics(pid, y, pm)
    return res


# ------------------------------------------------------------------ models
class Scaler:
    def fit(self, X):
        self.mu = X.mean(0, keepdims=True); self.sd = X.std(0, keepdims=True) + 1e-6; return self
    def transform(self, X):
        return (X - self.mu) / self.sd


def ridge_fit(X, y, alpha):
    Xt = torch.as_tensor(X, device=DEV, dtype=torch.float64)
    yt = torch.as_tensor(y, device=DEV, dtype=torch.float64)
    ym = yt.mean()
    K = Xt @ Xt.T                                              # dual form (n << d)
    a = torch.linalg.solve(K + alpha * torch.eye(len(K), device=DEV, dtype=torch.float64), yt - ym)
    w = Xt.T @ a
    return {'w': w.float().cpu().numpy(), 'b': float(ym)}


def ridge_pred(m, X):
    return X @ m['w'] + m['b']


class MLP(torch.nn.Module):
    def __init__(self, d, hidden, dropout):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.ReLU(),
                                       torch.nn.Dropout(dropout), torch.nn.Linear(hidden, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


def mlp_fit(Xtr, ytr, Xva, yva, hidden, dropout, wd, seed, lr=1e-3, max_epochs=300, patience=20, bs=256):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr = torch.as_tensor(Xtr, device=DEV); ytr = torch.as_tensor(ytr, device=DEV, dtype=torch.float32)
    Xva = torch.as_tensor(Xva, device=DEV); yva = torch.as_tensor(yva, device=DEV, dtype=torch.float32)
    m = MLP(Xtr.shape[1], hidden, dropout).to(DEV)
    with torch.no_grad():
        m.net[-1].bias.fill_(float(ytr.mean()))
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator(device='cpu').manual_seed(seed)
    best, best_state, bad, best_ep = float('inf'), None, 0, 0
    for ep in range(max_epochs):
        m.train()
        perm = torch.randperm(len(Xtr), generator=g).to(DEV)
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]
            loss = torch.nn.functional.mse_loss(m(Xtr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            v = torch.nn.functional.mse_loss(m(Xva), yva).item()
        if v < best - 1e-5:
            best, bad, best_ep = v, 0, ep
            best_state = {k: t.detach().clone() for k, t in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    m.eval()
    return m, best, best_ep


def mlp_pred(m, X):
    with torch.no_grad():
        return m(torch.as_tensor(X, device=DEV)).cpu().numpy()


# ------------------------------------------------------------------ selection inside a train fold
def select_and_fit(kind, recs_tr, Xcache, feat, seed):
    """Inner problem-level split of the TRAIN fold -> choose layer+HP by inner-val MSE.
    Returns (model, scaler, layer, hp, inner_val_mse)."""
    pid_tr = np.concatenate([[r['pid']] * len(r['y']) for r in recs_tr])
    y_tr = np.concatenate([r['y'] for r in recs_tr])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    a, b = next(gss.split(np.zeros(len(y_tr)), y_tr, pid_tr))
    best = None
    for li, L in enumerate(CAND_LAYERS):
        X = Xcache(li)
        sc = Scaler().fit(X[a])
        Xa, Xb = sc.transform(X[a]), sc.transform(X[b])
        grid = RIDGE_ALPHAS if kind == 'ridge' else MLP_GRID
        for hp in grid:
            if kind == 'ridge':
                m = ridge_fit(Xa, y_tr[a], hp)
                v = float(((ridge_pred(m, Xb) - y_tr[b]) ** 2).mean())
            else:
                m, v, _ = mlp_fit(Xa, y_tr[a], Xb, y_tr[b], seed=seed, **hp)
            if best is None or v < best[0]:
                best = (v, li, hp, m, sc)
    v, li, hp, m, sc = best
    if kind == 'ridge':     # refit on the whole train fold with chosen layer/alpha
        X = Xcache(li)
        sc = Scaler().fit(X)
        m = ridge_fit(sc.transform(X), y_tr, hp)
    # MLP keeps the early-stopped model (needs the inner val for stopping)
    return m, sc, CAND_LAYERS[li], hp, v


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diag-only', action='store_true')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    args = ap.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    recs = load(CAND_LAYERS)
    pid, k, n, y = flat(recs)
    print(f'{len(recs)} problems, {len(y)} steps, device={DEV}, load {time.time()-t0:.0f}s', flush=True)
    print('label dist:', {int(v): int((y == v).sum()) for v in np.unique(y)})
    print('problems with constant labels:', sum(np.ptp(r['y']) == 0 for r in recs))

    folds = list(GroupKFold(5).split(np.zeros(len(y)), y, pid))
    base = run_baselines(pid, k, n, y, folds)
    print('\n=== Task 5: baselines (GroupKFold-5 by problem) ===')
    for name, m in base.items():
        print(f'{name:>14}: ' + '  '.join(f'{kk}={vv:.4f}' if isinstance(vv, float) else f'{kk}={vv}'
                                         for kk, vv in m.items()), flush=True)
    (REPORT / 'baselines.json').write_text(json.dumps(base, indent=1))
    if args.diag_only:
        return

    # problem-index folds (for slicing recs)
    prob_ids = np.array([r['pid'] for r in recs])
    step_off = np.cumsum([0] + [len(r['y']) for r in recs])
    results = defaultdict(list)   # (kind, feat) -> list over seeds of metric dicts
    choices = defaultdict(list)
    for feat in FEATURES:
        full_X = {}
        def Xall(li):
            if li not in full_X:
                full_X[li] = build_X(recs, li, feat)
            return full_X[li]
        for kind in ('ridge', 'mlp'):
            seeds = args.seeds if kind == 'mlp' else args.seeds[:1]
            for seed in seeds:
                pred = np.empty_like(y)
                for f, (tr, te) in enumerate(folds):
                    tr_p = np.unique(pid[tr]); te_p = np.unique(pid[te])
                    tr_r = [i for i, p in enumerate(prob_ids) if p in set(tr_p)]
                    te_r = [i for i, p in enumerate(prob_ids) if p in set(te_p)]
                    tr_rows = np.concatenate([np.arange(step_off[i], step_off[i + 1]) for i in tr_r])
                    te_rows = np.concatenate([np.arange(step_off[i], step_off[i + 1]) for i in te_r])
                    m, sc, L, hp, v = select_and_fit(kind, [recs[i] for i in tr_r],
                                                     lambda li: Xall(li)[tr_rows], feat, seed)
                    Xte = sc.transform(Xall(CAND_LAYERS.index(L))[te_rows])
                    pred[te_rows] = ridge_pred(m, Xte) if kind == 'ridge' else mlp_pred(m, Xte)
                    choices[(kind, feat)].append({'seed': seed, 'fold': f, 'layer': L, 'hp': hp, 'inner_val_mse': v})
                    print(f'  {kind} {feat} seed={seed} fold={f}: layer={L} hp={hp} inner_val_mse={v:.4f} '
                          f'[{time.time()-t0:.0f}s]', flush=True)
                mt = metrics(pid, y, pred)
                results[(kind, feat)].append(mt)
                np.save(REPORT / f'oof_{kind}_{feat}_seed{seed}.npy', pred)
                print(f'{kind} {feat} seed={seed}: ' + '  '.join(f'{kk}={vv:.4f}' for kk, vv in mt.items()
                                                                if isinstance(vv, float)), flush=True)
        full_X.clear()

    # ---------------- Task 7 table
    cols = ['r2', 'mae', 'share_mae_pp', 'spearman', 'top1']
    lines = ['| method | evaluated_level R² | MAE | share MAE (pp) | within-problem Spearman | top-1 hit |',
             '|---|---|---|---|---|---|']
    def cell(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return '—'
        return f'{np.mean(vals):.3f}' if len(vals) == 1 else f'{np.mean(vals):.3f} ± {np.std(vals):.3f}'
    for name in ('uniform', 'global_mean', 'position_mean'):
        lines.append(f'| {name} | ' + ' | '.join(cell([base[name].get(c)]) for c in cols) + ' |')
    for kind in ('ridge', 'mlp'):
        for feat in FEATURES:
            rs = results[(kind, feat)]
            lines.append(f'| {kind} {feat} (n_seeds={len(rs)}) | ' + ' | '.join(cell([r[c] for r in rs]) for c in cols) + ' |')
    table = '\n'.join(lines)
    pm = base['position_mean']
    verdict = []
    for (kind, feat), rs in results.items():
        s = np.mean([r['spearman'] for r in rs]); t = np.mean([r['top1'] for r in rs])
        verdict.append(f'{kind} {feat}: spearman {s:.3f} vs pos {pm["spearman"]:.3f} '
                       f'({"BEATS" if s > pm["spearman"] else "does NOT beat"}), '
                       f'top1 {t:.3f} vs pos {pm["top1"]:.3f} ({"BEATS" if t > pm["top1"] else "does NOT beat"})')
    report = (f'# Difficulty probe report\n\nproblems={len(recs)} steps={len(y)} '
              f'non-constant-label problems={pm["n_problems_nonconst"]}\n\n{table}\n\n'
              f'Spearman/top-1 computed only on problems whose labels are not all equal.\n\n## vs position-mean\n'
              + '\n'.join(f'- {v}' for v in verdict) + '\n')
    (REPORT / 'report.md').write_text(report)
    (REPORT / 'choices.json').write_text(json.dumps({f'{k[0]}_{k[1]}': v for k, v in choices.items()}, indent=1, default=str))
    (REPORT / 'results.json').write_text(json.dumps({f'{k[0]}_{k[1]}': v for k, v in results.items()}, indent=1))
    print('\n' + report, flush=True)

    # ---------------- Task 8: refit best config on ALL problems and save
    best_key = max(results, key=lambda kf: (np.mean([r['spearman'] for r in results[kf]]),
                                            np.mean([r['top1'] for r in results[kf]])))
    kind, feat = best_key
    print(f'best config by within-problem Spearman (tie: top1): {kind} {feat}', flush=True)
    X_cache = {}
    def Xall2(li):
        if li not in X_cache:
            X_cache[li] = build_X(recs, li, feat)
        return X_cache[li]
    m, sc, L, hp, v = select_and_fit(kind, recs, Xall2, feat, args.seeds[0])
    OUT.mkdir(exist_ok=True)
    payload = {'kind': kind, 'feature': feat, 'layer': L, 'hp': hp,
               'scaler_mu': sc.mu, 'scaler_sd': sc.sd}
    if kind == 'ridge':
        payload['ridge'] = m
    else:
        payload['mlp_state'] = {k2: t.cpu() for k2, t in m.state_dict().items()}
        payload['mlp_dims'] = {'d': int(sc.mu.shape[1]), **hp}
    torch.save(payload, OUT / 'probe.pt')
    (OUT / 'metadata.json').write_text(json.dumps({
        'base_model': 'meta-llama/Llama-3.1-8B-Instruct (local fp16)',
        'target': 'per-step evaluated_level (1..5) from official PaB credit prompt (seed-42 benchmarks)',
        'kind': kind, 'feature': feat, 'layer': L, 'hp': hp, 'inner_val_mse': v,
        'capture': 'hidden_state_capture_v2.generate_with_marker_states (no double BOS)',
        'cv_metrics_mean': {c: float(np.mean([r[c] for r in results[best_key]])) for c in cols},
        'cv_metrics_std': {c: float(np.std([r[c] for r in results[best_key]])) for c in cols},
        'position_mean_baseline': {c: pm.get(c) for c in cols},
        'n_problems': len(recs), 'n_steps': int(len(y)),
        'allocation': 'w_k = d_k / sum(d) then polynomial decay p=2 via allocate_tokens (inference/_5_...)',
    }, indent=1, default=float))
    print(f'saved {OUT}/probe.pt and metadata.json', flush=True)


if __name__ == '__main__':
    main()
