#!/usr/bin/env python3
"""Compare vanilla / PaB / probe runs on MATH-500 (run on a compute node, CPU only).

FLOPs (analytic, forward only, KV-cached decoding):
  per sequence with T = prompt + output tokens:
     FLOPs ~= 2 * N * T + L * d_attn * T^2
  (2N per token for the dense matmuls incl. LM head, N = params excluding the input embedding;
   attention scores+values cost 2*L*d_attn*ctx per token, summed over ctx=1..T -> L*d_attn*T^2)
Probe (Ridge F3): 2 * 8192 FLOPs per step (+ standardisation), reported separately.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # repo root
PLAN = ROOT / 'experiments' / 'probe_budget' / 'data_v2' / 'math500_plan'
RUNS = {
    'vanilla': ROOT / 'results' / 'vanilla_math_results_qwq32b_v2.json',
    'pab': ROOT / 'results' / 'planned_local_weighted_math_results_pab_qwq32b_v2.json',
    'probe': ROOT / 'results' / 'planned_local_weighted_math_results_probe_qwq32b_v2.json',
}
LLAMA = dict(N=8.03e9 - 128256 * 4096, L=32, d_attn=32 * 128)
QWQ = dict(N=32.76e9 - 152064 * 5120, L=64, d_attn=40 * 128)


def flops(m, prompt, out):
    T = prompt + out
    return 2 * m['N'] * T + m['L'] * m['d_attn'] * T ** 2


def main():
    plan = {}
    for p in sorted(PLAN.glob('records*.jsonl')):      # sequential run + shard files
        for l in p.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                assert int(r['id']) not in plan, f'duplicate id {r["id"]}'
                plan[int(r['id'])] = r
    print(f'planning records: {len(plan)}')
    assert len(plan) == 500
    rows, per_run = [], {}
    for name, path in RUNS.items():
        if not path.exists():
            print(f'missing {path}'); continue
        data = json.loads(path.read_text())
        head, res = data[0], data[1:]
        n = len(res)
        ok = [r for r in res if not r['failed']]
        acc = 100 * np.mean([r['score'] for r in res])
        q_in = np.array([r['prompt_tokens'] for r in ok]); q_out = np.array([r['completion_tokens'] for r in ok])
        q_fl = np.array([flops(QWQ, r['prompt_tokens'], r['completion_tokens']) for r in ok])
        q_sec = np.array([r['reasoning_seconds'] for r in ok])
        p_in = p_out = p_fl = p_sec = np.zeros(1); probe_fl = 0.0
        if name in ('pab', 'probe'):
            ids = [r['id'] for r in ok]
            pr = [plan[i] for i in ids]
            if name == 'pab':
                p_in = np.array([r['call1']['input_tokens'] + sum(a['input_tokens'] for a in r['call2_attempts']) for r in pr])
                p_out = np.array([r['call1']['output_tokens'] + sum(a['output_tokens'] for a in r['call2_attempts']) for r in pr])
                p_fl = np.array([flops(LLAMA, r['call1']['input_tokens'], r['call1']['output_tokens'])
                                 + sum(flops(LLAMA, a['input_tokens'], a['output_tokens']) for a in r['call2_attempts'])
                                 for r in pr])
            else:
                p_in = np.array([r['call1']['input_tokens'] for r in pr])
                p_out = np.array([r['call1']['output_tokens'] for r in pr])
                probe_fl = np.mean([2 * 8192 * len(r['steps']) for r in pr])
                p_fl = np.array([flops(LLAMA, r['call1']['input_tokens'], r['call1']['output_tokens']) for r in pr]) + probe_fl
            p_sec = np.array([r['planning_seconds'][name] for r in pr])
        rows.append({
            'method': name, 'n': n, 'failed': n - len(ok), 'accuracy': acc,
            'plan_in': p_in.mean(), 'plan_out': p_out.mean(),
            'reason_in': q_in.mean(), 'reason_out': q_out.mean(),
            'plan_TFLOPs': p_fl.mean() / 1e12, 'reason_TFLOPs': q_fl.mean() / 1e12,
            'total_TFLOPs': (p_fl.mean() + q_fl.mean()) / 1e12, 'probe_FLOPs': probe_fl,
            'plan_s': p_sec.mean(), 'reason_s': q_sec.mean(), 'total_s': p_sec.mean() + q_sec.mean(),
            'run_wall_s': head.get('run_wall_seconds_this_invocation'),
            'max_out': int(q_out.max()), 'n_out_ge_8000': int((q_out >= 8000).sum()),
            'reason_s_median': float(np.median(q_sec)),
        })
        per_run[name] = {r['id']: bool(r['score']) for r in res}
    hdr = ('| method | acc (%) | fail | plan in | plan out | reason in | reason out | plan TFLOPs | '
           'reason TFLOPs | total TFLOPs | plan s | reason s | total s/problem | QwQ run wall s |')
    print(hdr); print('|' + '---|' * 14)
    for r in rows:
        print(f"| {r['method']} | {r['accuracy']:.1f} | {r['failed']} | {r['plan_in']:.0f} | {r['plan_out']:.0f} | "
              f"{r['reason_in']:.0f} | {r['reason_out']:.0f} | {r['plan_TFLOPs']:.2f} | {r['reason_TFLOPs']:.1f} | "
              f"{r['total_TFLOPs']:.1f} | {r['plan_s']:.1f} | {r['reason_s']:.1f} | {r['total_s']:.1f} | {r['run_wall_s']} |")
    (ROOT / 'results' / 'summary_v2.json').write_text(json.dumps(rows, indent=1, default=float))
    print('\nmax output / #outputs>=8000 / median reason s:',
          {r['method']: (r['max_out'], r['n_out_ge_8000'], round(r['reason_s_median'], 1)) for r in rows})
    for a, b in (('vanilla', 'pab'), ('vanilla', 'probe'), ('pab', 'probe')):
        if a in per_run and b in per_run:
            ids = per_run[a].keys()
            only_a = sum(per_run[a][i] and not per_run[b][i] for i in ids)
            only_b = sum(per_run[b][i] and not per_run[a][i] for i in ids)
            print(f'discordant {a} vs {b}: only {a} correct={only_a}, only {b} correct={only_b}')
    by_lv = {}
    for name in per_run:
        by_lv[name] = {lv: round(100 * np.mean([per_run[name][i] for i in per_run[name] if plan[i]['level'] == lv]), 1)
                       for lv in range(1, 6)}
    print('accuracy by level:', by_lv)
    if plan:
        fl =[f.split(':')[0] for r in plan.values() for f in r['flags']]
        from collections import Counter
        print('\nplanning flags:', dict(Counter(fl)))


if __name__ == '__main__':
    main()
