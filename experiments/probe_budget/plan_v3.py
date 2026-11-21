#!/usr/bin/env python3
"""Planning (LLaMA-3.1-8B-Instruct, HF fp16, 1 GPU, sequential) for
  --dataset math500 : JOINT only (call1/call2 records already exist in data_v2/math500_plan)
  --dataset travel  : call1 marker + probe + PaB call2 + JOINT, on TravelPlanner validation (180)

JOINT (ablation, 1 LLaMA call): the official break_down prompt (domain formatted as in the
official script) + an instruction to output "Level: k, Credit: c" after each hint
(credits sum to 100). Credits are used exactly like PaB credits (w = c/sum c, then decay).
Retry policy identical to PaB call2: greedy, then up to 4 sampled tries (T=0.1, top_p=0.9),
then fallback (steps ok -> uniform credits; steps not parseable -> official fallback step).
All attempts count toward the joint planning tokens/time.

TravelPlanner: level easy/medium/hard -> 1/3/5 (official level_mapper); domain "travel planning";
PaB call2 benchmarks = "None" (official break_down_question.py does the same for travelplanner).
Call1 marker prompt is IDENTICAL to the one the probe was trained on (MATH wording), as the
official decomposition prompt is also used unchanged (with "math problem" wording) for travel.
"""
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate_training_data_v2 import (LLAMA, STEP_PATTERN, assemble_steps, get_benchmark_examples,
                                       load_official_prompts, parse_labels)
from hidden_state_capture_v2 import build_decomp_messages_with_marker, generate_with_marker_states
from probe_inference_v2 import load_probe, predict_difficulties

HERE = Path(__file__).resolve().parent
FALLBACK_STEP = '1. Directly solve the problem. Hint: None.'
TP_LEVEL = {'easy': 1, 'medium': 3, 'hard': 5}
CFG = {
    'math500': dict(domain='math', out=HERE / 'data_v2' / 'joint_math500', ds_dir=HERE.parents[1] / 'dataset' / 'MATH-500'),
    'travel': dict(domain='travel planning', out=HERE / 'data_v2' / 'travel_plan',
                   ds_dir=HERE.parents[1] / 'dataset' / 'TravelPlanner'),
}

# ---- JOINT prompt: official break_down + credit lines (example gets credit lines too) ----
JOINT_ADDENDUM = (
    "\nIn the same answer, also evaluate the difficulty of each sub-question and assign it a portion of the credits, "
    "assuming the whole problem is worth 100 points. Right after each hint, add one line in the form\n"
    "Level: <integer from 1 to 5>, Credit: <integer from 1 to 100>\n"
    "The credits of all sub-questions MUST sum to 100. Do not add anything else.\n"
)
EXAMPLE_CREDITS = ['Level: 2, Credit: 25', 'Level: 3, Credit: 40', 'Level: 3, Credit: 35']
CREDIT_RE = re.compile(r'Level:\s*(\d+)\s*[,;]?\s*Credit:\s*(\d+(?:\.\d+)?)', re.I)


def joint_messages(prompts, domain, problem, level):
    sys_p = prompts['break_down']['system'].format(domain=domain)
    # insert credit line after each example hint (the three "Hint:" lines of the official example)
    lines, k, out = sys_p.split('\n'), 0, []
    for ln in lines:
        out.append(ln)
        if ln.startswith('Hint:') and k < len(EXAMPLE_CREDITS):
            out.append(EXAMPLE_CREDITS[k]); k += 1
    assert k == 3, k
    sys_p = '\n'.join(out) + JOINT_ADDENDUM
    usr_p = prompts['break_down']['user'].format(problem=problem, level=level)
    return [{'role': 'system', 'content': sys_p}, {'role': 'user', 'content': usr_p}]


def parse_joint(text):
    """-> (steps, levels, credits, flags) or raise ValueError."""
    matches = re.findall(STEP_PATTERN, text, re.DOTALL)
    if not matches:
        raise ValueError('no_steps')
    steps, levels, credits, flags = [], [], [], []
    # credit line for step i = first Level/Credit line between end of hint i and start of step i+1
    blocks = re.split(r'\n(?=\d+\.\s)', text.strip())
    blocks = [b for b in blocks if re.match(r'\d+\.', b.strip())]
    for q, h in matches:
        q2 = CREDIT_RE.sub('', q).strip(); h2 = CREDIT_RE.sub('', h).strip().rstrip(',;')
        steps.append(f'{q2} Hint: {h2}')
    if len(blocks) != len(steps):
        raise ValueError(f'block_count:{len(blocks)}!={len(steps)}')
    for b in blocks:
        m = CREDIT_RE.findall(b)
        if len(m) != 1:
            raise ValueError(f'credit_lines:{len(m)}')
        lv, c = int(m[0][0]), float(m[0][1])
        if not 1 <= lv <= 5 or c <= 0:
            raise ValueError(f'bad_level_or_credit:{lv},{c}')
        levels.append(lv); credits.append(c)
    if abs(sum(credits) - 100) > 1e-6:
        flags.append(f'credit_sum:{sum(credits)}')
    return steps, levels, credits, flags


@torch.no_grad()
def hf_generate(model, tok, messages, max_new_tokens, sample=False, seed=0):
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt').to(model.device)
    kw = dict(do_sample=True, temperature=0.1, top_p=0.9) if sample else dict(do_sample=False, temperature=None, top_p=None)
    if sample:
        torch.manual_seed(seed)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                         pad_token_id=tok.eos_token_id, **kw)
    torch.cuda.synchronize()
    gen = out[0, ids.shape[1]:].tolist()
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids('<|eot_id|>')}
    return {'text': tok.decode(gen, skip_special_tokens=True), 'input_tokens': int(ids.shape[1]),
            'output_tokens': len(gen), 'seconds': time.perf_counter() - t0,
            'truncated': not (gen and gen[-1] in eos)}


def run_joint(model, tok, prompts, domain, problem, level, i):
    attempts, res = [], None
    for a in range(5):
        g = hf_generate(model, tok, joint_messages(prompts, domain, problem, level), 1024,
                        sample=a > 0, seed=1000 * i + a)
        att = {k: g[k] for k in ('input_tokens', 'output_tokens', 'seconds')}; att['response'] = g['text']
        try:
            if g['truncated']:
                raise ValueError('truncated')
            res = parse_joint(g['text']); att['ok'] = True; attempts.append(att); break
        except ValueError as e:
            att['ok'] = False; att['error'] = str(e); attempts.append(att)
    flags = []
    if res is None:
        try:   # steps only (credit lines broken) from the last attempt
            st = [f'{q.strip()} Hint: {h.strip()}' for q, h in re.findall(STEP_PATTERN, attempts[-1]['response'], re.DOTALL)]
        except Exception:
            st = []
        st = [CREDIT_RE.sub('', s).strip() for s in st] or [FALLBACK_STEP]
        res = (st, [None] * len(st), [100.0 / len(st)] * len(st), [])
        flags.append('joint_fallback_uniform' if st != [FALLBACK_STEP] else 'joint_fallback_step')
    steps, levels, credits, fl = res
    return {'steps': steps, 'levels': levels, 'credits': credits, 'flags': flags + [f'joint_{f}' for f in fl],
            'attempts': attempts, 'seconds': sum(a['seconds'] for a in attempts)}


def run_pab_call2(model, tok, prompts, domain, bench, problem, steps, i):
    sys_p = prompts['evaluate_question_level_batch']['system'].format(domain=domain, benchmarks=bench)
    usr_p = prompts['evaluate_question_level_batch']['user'].format(problem=problem, steps=steps)
    msgs = [{'role': 'system', 'content': sys_p}, {'role': 'user', 'content': usr_p}]
    attempts, credits, levels, flags = [], None, None, []
    for a in range(5):
        g = hf_generate(model, tok, msgs, 2048, sample=a > 0, seed=1000 * i + a)
        att = {k: g[k] for k in ('input_tokens', 'output_tokens', 'seconds')}; att['response'] = g['text']
        try:
            if g['truncated']:
                raise ValueError('truncated')
            lv, cr, _, fl = parse_labels(g['text'], len(steps))
            if any(c is None for c in cr):
                raise ValueError('credit_missing')
            credits, levels = cr, lv; att['ok'] = True; attempts.append(att)
            flags += [f'call2_{f}' for f in fl]; break
        except ValueError as e:
            att['ok'] = False; att['error'] = str(e); attempts.append(att)
    if credits is None:
        credits = [100.0 / len(steps)] * len(steps); levels = [None] * len(steps)
        flags.append('credit_fallback_uniform')
    return credits, levels, flags, attempts


def load_records(out):
    done = {}
    for p in sorted(out.glob('records*.jsonl')):
        for l in p.read_text().splitlines():
            if l.strip():
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    print(f'WARNING: skipping partial line in {p.name}', flush=True); continue
                done[r['id']] = r
    return done


def load_items(dataset):
    if dataset == 'math500':
        ds = load_dataset('HuggingFaceH4/MATH-500')['test']
        return [dict(problem=x['problem'], level=x['level'], solution=x['solution'], answer=x['answer'],
                     reference='') for x in ds]
    ds = load_dataset('osunlp/TravelPlanner', 'validation')['validation']
    return [dict(problem=x['query'], level=TP_LEVEL[x['level']], solution='', answer='',
                 reference=x['reference_information']) for x in ds]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=list(CFG), required=True)
    ap.add_argument('--shard-id', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--smoke', type=int, default=0, help='process only N items of this shard into out/smoke')
    args = ap.parse_args()
    cfg = CFG[args.dataset]; out = cfg['out'] / 'smoke' if args.smoke else cfg['out']
    out.mkdir(parents=True, exist_ok=True)
    items = load_items(args.dataset)
    done = load_records(out)
    if args.merge:
        missing = [i for i in range(len(items)) if str(i) not in done]
        if missing:
            raise SystemExit(f'merge: {len(missing)} missing, e.g. {missing[:10]}')
        write_outputs(args.dataset, items, done, cfg['ds_dir']); return

    prompts = load_official_prompts()
    travel = args.dataset == 'travel'
    if travel:
        probe, meta = load_probe(); L = probe['layer']
        bench = 'None'
        print(f'probe {meta["kind"]} {meta["feature"]} layer {L}', flush=True)
    tok = AutoTokenizer.from_pretrained(LLAMA)
    model = AutoModelForCausalLM.from_pretrained(LLAMA, torch_dtype=torch.float16, device_map='cuda').eval()
    print(f'GPU: {torch.cuda.get_device_name(0)}; {len(done)} already done', flush=True)
    hf_generate(model, tok, [{'role': 'user', 'content': 'What is 1+1?'}], 8)   # warm-up
    if travel:
        generate_with_marker_states(model, tok, build_decomp_messages_with_marker('What is 1+1?', 1), [L], 32)

    rec_path = out / f'records_shard{args.shard_id}.jsonl'
    n_proc = 0
    for i, it in enumerate(items):
        sid = str(i)
        if sid in done or i % args.num_shards != args.shard_id:
            continue
        if args.smoke and n_proc >= args.smoke:
            break
        n_proc += 1
        r = {'id': sid, 'level': it['level'], 'flags': []}
        if travel:
            cap = generate_with_marker_states(model, tok, build_decomp_messages_with_marker(it['problem'], it['level']), [L], 768)
            r['call1'] = {k: cap[k] for k in ('input_tokens', 'output_tokens', 'seconds')}
            r['decomposition_text'] = cap['text']
            try:
                steps, _, H, used_end = assemble_steps(cap); r['last_step_uses_h_end'] = used_end
            except ValueError as e:
                steps, H = [FALLBACK_STEP], None; r['flags'].append(f'call1_fallback:{e}')
            r['steps'] = steps
            t0 = time.perf_counter()
            d = (predict_difficulties(probe, H[:, 0].astype(np.float32), cap['h_end'][0].astype(np.float32),
                                      cap['h_prompt'][0].astype(np.float32)) if H is not None else [1.0])
            r['probe'] = {'seconds': time.perf_counter() - t0, 'difficulties': d}
            if H is None:
                cr, lv, fl, att = [100.0], [None], ['call2_skipped_fallback_step'], []
            else:
                cr, lv, fl, att = run_pab_call2(model, tok, prompts, cfg['domain'], bench, it['problem'], steps, i)
            r['credits'], r['levels'], r['call2_attempts'] = cr, lv, att; r['flags'] += fl
        j = run_joint(model, tok, prompts, cfg['domain'], it['problem'], it['level'], i)
        r['joint'] = j; r['flags'] += j['flags']
        r['planning_seconds'] = {'joint': j['seconds']}
        if travel:
            r['planning_seconds']['pab'] = r['call1']['seconds'] + sum(a['seconds'] for a in r['call2_attempts'])
            r['planning_seconds']['probe'] = r['call1']['seconds'] + r['probe']['seconds']
        with rec_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
        done[sid] = r
        print(f'[{sid}] L{it["level"]} joint n={len(j["steps"])} cr={j["credits"]} '
              + (f'| pab n={len(r["steps"])} cr={r["credits"]} probe={[round(x, 2) for x in r["probe"]["difficulties"]]} '
                 if travel else '') + f'flags={r["flags"]} t={r["planning_seconds"]}', flush=True)
    print(f'SHARD {args.shard_id} DONE', flush=True)


def write_outputs(dataset, items, done, ds_dir):
    def write(name, get):
        stats = []
        for i, it in enumerate(items):
            r = done[str(i)]; steps, credits = get(r)
            stats.append({'id': str(i), 'query': it['problem'], 'steps': steps, 'credits': credits,
                          'solution': it['solution'], 'answer': it['answer'], 'level': it['level'],
                          'reference': it['reference'], 'flags': r['flags']})
        tmp = ds_dir / f'{name}.json.tmp'
        tmp.write_text(json.dumps({'stats': stats}, ensure_ascii=False, indent=1), encoding='utf-8')
        tmp.rename(ds_dir / f'{name}.json')
        print('wrote', ds_dir / f'{name}.json')
    write('plan_v3_joint', lambda r: (r['joint']['steps'], r['joint']['credits']))
    if dataset == 'travel':
        write('plan_v3_pab', lambda r: (r['steps'], r['credits']))
        write('plan_v3_probe', lambda r: (r['steps'], r['probe']['difficulties']))
    print('flag counts:', dict(Counter(f.split(':')[0] for r in done.values() for f in r['flags'])))
    for k in done['0']['planning_seconds']:
        print(f'mean planning seconds {k}: {np.mean([r["planning_seconds"][k] for r in done.values()]):.2f}')
    print('PLAN DONE', flush=True)


if __name__ == '__main__':
    main()
