#!/usr/bin/env python3
"""Planning stage for the MATH-500 evaluation (LLaMA-3.1-8B-Instruct, HF fp16, 1 GPU, sequential).

Per problem (ids = row index of HuggingFaceH4/MATH-500 test, same as run_inf):
  Call 1  marker decomposition (greedy) with hidden states at layer = probe layer only
          -> official-format steps  (SHARED by PaB and probe runs)
  Probe   d_k = probe(h)  (Ridge F3, layer from probe_artifacts/metadata.json)
  Call 2  official PaB credit prompt (seed-42 benchmarks), greedy; on parse failure retry
          up to 4x with sampling T=0.1, top_p=0.9 (mirrors official generate_response +
          llm_retry); all retries count toward PaB planning time.

Fallbacks (counted, never silent):
  call-1 failure -> official fallback step "1. Directly solve the problem. Hint: None." for BOTH runs
  call-2 failure after retries -> uniform credits for PaB (flag credit_fallback)

Outputs:
  data_v2/math500_plan/records.jsonl        per-problem record (resumable)
  dataset/MATH-500/plan_v2_pab.json       {"stats":[...]} credits = LLaMA credits
  dataset/MATH-500/plan_v2_probe.json     {"stats":[...]} credits = probe d_k
  dataset/MATH-500/plan_v2_pab_level.json credits = LLaMA evaluated_level (optional ablation)
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate_training_data_v2 import (LLAMA, assemble_steps, call_credit, get_benchmark_examples,
                                       load_official_prompts, parse_labels)
from hidden_state_capture_v2 import build_decomp_messages_with_marker, generate_with_marker_states
from probe_inference_v2 import load_probe, predict_difficulties

HERE = Path(__file__).resolve().parent
OUT = HERE / 'data_v2' / 'math500_plan'
DS_DIR = HERE.parents[1] / 'dataset' / 'MATH-500'
FALLBACK_STEP = '1. Directly solve the problem. Hint: None.'


@torch.no_grad()
def call_credit_sampled(model, tok, prompts, bench, problem, steps, seed):
    torch.manual_seed(seed)
    sys_p = prompts['evaluate_question_level_batch']['system'].format(domain='math', benchmarks=bench)
    usr_p = prompts['evaluate_question_level_batch']['user'].format(problem=problem, steps=steps)
    ids = tok.apply_chat_template([{'role': 'system', 'content': sys_p}, {'role': 'user', 'content': usr_p}],
                                  add_generation_prompt=True, return_tensors='pt').to(model.device)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=2048, do_sample=True,
                         temperature=0.1, top_p=0.9, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    gen = out[0, ids.shape[1]:].tolist()
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids('<|eot_id|>')}
    return {'text': tok.decode(gen, skip_special_tokens=True), 'input_tokens': int(ids.shape[1]),
            'output_tokens': len(gen), 'seconds': time.perf_counter() - t0,
            'truncated': not (gen and gen[-1] in eos)}


def load_done():
    done = {}
    for p in sorted(OUT.glob('records*.jsonl')):
        for l in p.read_text().splitlines():
            if l.strip():
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    print(f'WARNING: skipping partial line in {p.name}', flush=True)
                    continue
                done[r['id']] = r
    return done


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard-id', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--merge', action='store_true', help='only write plan_v2_*.json from all shard records')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rec_path = OUT / f'records_shard{args.shard_id}.jsonl'
    done = load_done()

    ds = load_dataset('HuggingFaceH4/MATH-500')['test']
    if args.merge:
        missing = [i for i in range(len(ds)) if str(i) not in done]
        if missing:
            raise SystemExit(f'merge: {len(missing)} problems missing, e.g. {missing[:10]}')
        write_outputs(ds, done)
        return
    prompts = load_official_prompts()
    bench, bench_rec = get_benchmark_examples()
    probe, meta = load_probe()
    L = probe['layer']
    print(f'probe: {meta["kind"]} {meta["feature"]} layer {L}; benchmarks {bench_rec["unique_ids"]}', flush=True)
    print(f'GPU: {torch.cuda.get_device_name(0)}; {len(done)} already done', flush=True)
    tok = AutoTokenizer.from_pretrained(LLAMA)
    model = AutoModelForCausalLM.from_pretrained(LLAMA, torch_dtype=torch.float16, device_map='cuda').eval()

    # warm-up (excluded from timing)
    generate_with_marker_states(model, tok, build_decomp_messages_with_marker('What is 1+1?', 1), [L], 32)

    for i, item in enumerate(ds):
        sid = str(i)
        if sid in done or i % args.num_shards != args.shard_id:
            continue
        problem, level = item['problem'], item['level']
        r = {'id': sid, 'level': level, 'flags': []}

        # ---- call 1 (shared) ----
        cap = generate_with_marker_states(model, tok, build_decomp_messages_with_marker(problem, level), [L], 768)
        r['call1'] = {k: cap[k] for k in ('input_tokens', 'output_tokens', 'seconds')}
        r['decomposition_text'] = cap['text']
        try:
            steps, raw, H, used_end = assemble_steps(cap)
            r['last_step_uses_h_end'] = used_end
        except ValueError as e:
            steps, H = [FALLBACK_STEP], None
            r['flags'].append(f'call1_fallback:{e}')
        r['steps'] = steps

        # ---- probe ----
        t0 = time.perf_counter()
        if H is not None:
            d = predict_difficulties(probe, H[:, 0].astype(np.float32), cap['h_end'][0].astype(np.float32),
                                     cap['h_prompt'][0].astype(np.float32))
        else:
            d = [1.0]
        r['probe'] = {'seconds': time.perf_counter() - t0, 'difficulties': d}

        # ---- call 2 (PaB) ----
        attempts, credits, levels = [], None, None
        if H is None:
            credits, levels = [100.0], [None]
            r['flags'].append('call2_skipped_fallback_step')
        else:
            for a in range(5):
                c2 = (call_credit(model, tok, prompts, bench, problem, steps, 2048) if a == 0 else
                      call_credit_sampled(model, tok, prompts, bench, problem, steps, seed=1000 * i + a))
                att = {k: c2[k] for k in ('input_tokens', 'output_tokens', 'seconds')}
                att['response'] = c2['text']
                try:
                    if c2['truncated']:
                        raise ValueError('truncated')
                    lv, cr, _, fl = parse_labels(c2['text'], len(steps))
                    if any(c is None for c in cr):
                        raise ValueError('credit_missing')
                    credits, levels = cr, lv
                    att['ok'] = True; attempts.append(att)
                    r['flags'] += [f'call2_{f}' for f in fl]
                    break
                except ValueError as e:
                    att['ok'] = False; att['error'] = str(e); attempts.append(att)
            if credits is None:
                credits = [100.0 / len(steps)] * len(steps); levels = [None] * len(steps)
                r['flags'].append('credit_fallback_uniform')
        r['call2_attempts'] = attempts
        r['credits'], r['levels'] = credits, levels
        r['planning_seconds'] = {
            'pab': r['call1']['seconds'] + sum(a['seconds'] for a in attempts),
            'probe': r['call1']['seconds'] + r['probe']['seconds'],
        }
        with rec_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
        done[sid] = r
        print(f'[{sid}] L{level} n={len(steps)} credits={credits} probe_d={[round(x, 2) for x in d]} '
              f'flags={r["flags"]} t_pab={r["planning_seconds"]["pab"]:.1f}s '
              f't_probe={r["planning_seconds"]["probe"]:.1f}s', flush=True)
    print(f'SHARD {args.shard_id} DONE', flush=True)


def write_outputs(ds, done):
    # ---- write run_inf-compatible files ----
    def write(name, credit_key):
        stats = []
        for i, item in enumerate(ds):
            r = done[str(i)]
            cr = {'pab': r['credits'], 'probe': r['probe']['difficulties'],
                  'pab_level': (r['levels'] if None not in r['levels'] else [1.0] * len(r['steps']))}[credit_key]
            stats.append({'id': str(i), 'query': item['problem'], 'steps': r['steps'], 'credits': cr,
                          'solution': item['solution'], 'answer': item['answer'], 'level': item['level'],
                          'reference': '', 'flags': r['flags']})
        tmp = DS_DIR / f'{name}.json.tmp'
        tmp.write_text(json.dumps({'stats': stats}, ensure_ascii=False, indent=1), encoding='utf-8')
        tmp.rename(DS_DIR / f'{name}.json')
    write('plan_v2_pab', 'pab'); write('plan_v2_probe', 'probe'); write('plan_v2_pab_level', 'pab_level')

    flags = [f.split(':')[0] for r in done.values() for f in r['flags']]
    from collections import Counter
    print('flag counts:', dict(Counter(flags)))
    for k in ('pab', 'probe'):
        print(f'mean planning seconds {k}: {np.mean([r["planning_seconds"][k] for r in done.values()]):.2f}')
    print('PLAN DONE', flush=True)


if __name__ == '__main__':
    main()
