#!/usr/bin/env python3
"""v2 training-data generation on MATH train (data_v2/math_train_sample2000.json).

Per problem, two LLaMA-3.1-8B-Instruct calls (transformers, fp16, greedy):
  Call 1: marker decomposition; hidden states captured only at marker-completing
          tokens, the last prompt token (h_prompt) and the last pre-EOS token (h_end).
  Call 2: the OFFICIAL PaB credit/level prompt (PROMPTS["evaluate_question_level_batch"]
          read verbatim from dataset/break_down_question.py) with the seed-42
          benchmark examples, steps passed in the official format
          ("N. question Hint: hint", python-list repr), markers removed.

Labels: per-step evaluated_level (int 1..5, main) and credit (aux, stored only).
Problems that fail any check are written to the exclusion list with a reason;
no fallback labels are ever produced.

Output (resumable, shard-agnostic):
  data_v2/gen/problems/{problem_id}.pt   one dict per kept problem
  data_v2/gen/excluded_shard{S}.jsonl    {"problem_id", "reason", ...}
"""
import argparse
import json
import os
import random
import re
import sys
import textwrap  # noqa: F401  (needed by exec of the official PROMPTS block)
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hidden_state_capture_v2 import STEP_MARKER, build_decomp_messages_with_marker, generate_with_marker_states

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]  # repo root (this file: experiments/probe_budget/)
DATA = HERE / 'data_v2'
LLAMA = os.environ.get('LLAMA_PATH', 'meta-llama/Llama-3.1-8B-Instruct')
LAYERS = list(range(33))          # embeddings + 32 blocks; fp16 -> see size estimate
STEP_PATTERN = r"(\d+\..*?)(?=\nHint:)(?:\nHint:\s*(.*?))(?:\n|$)"   # official regex


# ---------------------------------------------------------------- official prompt
def load_official_prompts():
    src = (REPO / 'dataset' / 'break_down_question.py').read_text(encoding='utf-8')
    a = src.index('PROMPTS = {}')
    b = src.index('async def break_down_question')
    ns = {'textwrap': textwrap}
    exec(src[a:b], ns)
    return ns['PROMPTS']


def get_benchmark_examples():
    """Reproduce break_down_question_measured.py: random.seed(42) then
    MathDatasetLoader(decomposed=False).create_benchmark_questions()."""
    path = DATA / 'benchmark_examples_seed42.json'
    from datasets import load_dataset
    ds = load_dataset('HuggingFaceH4/MATH-500')['test']
    random.seed(42)
    bins = [[] for _ in range(5)]
    for d in ds:
        bins[d['level'] - 1].append(d)
    qs = [random.choice(b) for b in bins if b]
    text = "\n".join(f"\nLevel {q['level']} Example:\nQ: {q['problem']}\nA: {q['solution']}\n" for q in qs)
    rec = {'seed': 42, 'unique_ids': [q['unique_id'] for q in qs],
           'levels': [q['level'] for q in qs], 'benchmark_examples': text}
    if path.exists():
        old = json.loads(path.read_text(encoding='utf-8'))
        assert old['benchmark_examples'] == text, 'benchmark examples changed!'
    else:
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding='utf-8')
    return text, rec


# ---------------------------------------------------------------- official parser
def extract_json_objects(text, decoder=json.JSONDecoder()):
    pos, results = 0, []
    while True:
        match = text.find("{", pos)
        if match == -1:
            break
        try:
            result, index = decoder.raw_decode(text[match:])
            results.append(result)
            pos = match + index
        except ValueError:
            pos = match + 1
    return results


def maybe_load_json(text):
    try:
        return json.loads(text)
    except Exception:
        res = extract_json_objects(text)
        return res[0] if len(res) else res


def _as_int(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str) and v.strip().isdigit():
        return int(v.strip())
    return None


def parse_labels(response, n_steps):
    """Returns (levels, credits, problem_level, flags) or raises ValueError(reason)."""
    obj = maybe_load_json(response)
    if not isinstance(obj, dict) or not obj:
        raise ValueError('json_unparseable')
    levels, credits, flags = [], [], []
    for i in range(1, n_steps + 1):
        e = obj.get(str(i))
        if not isinstance(e, dict):
            raise ValueError(f'step_missing:{i}')
        lv = _as_int(e.get('evaluated_level'))
        if lv is None or not 1 <= lv <= 5:
            raise ValueError(f'bad_level:{i}:{e.get("evaluated_level")!r}')
        levels.append(lv)
        c = e.get('credit')
        credits.append(float(c) if isinstance(c, (int, float)) and not isinstance(c, bool) else None)
    extra = [k for k in obj if k.isdigit() and int(k) > n_steps]
    if extra:
        flags.append(f'extra_step_keys:{extra}')
    if any(c is None for c in credits):
        flags.append('credit_missing_or_nonnumeric')
    elif abs(sum(credits) - 100) > 1e-6:
        flags.append(f'credit_sum:{sum(credits)}')
    pl = obj.get('problem', {}).get('evaluated_level') if isinstance(obj.get('problem'), dict) else None
    return levels, credits, pl, flags


# ---------------------------------------------------------------- step assembly
def assemble_steps(cap):
    """Map marker segments -> (official-format step strings, raw segments, states).
    Raises ValueError(reason) on any inconsistency."""
    if not cap['stopped_by_eos']:
        raise ValueError('decomp_truncated')
    parts = cap['text'].split(STEP_MARKER)
    n_mark = len(parts) - 1
    if n_mark != len(cap['marker_states']):
        raise ValueError(f'marker_state_count:{n_mark}!={len(cap["marker_states"])}')
    raw = [p.strip() for p in parts[:-1]]
    if any(not r for r in raw):
        raise ValueError('empty_segment')
    states = list(cap['marker_states'])
    trailing = parts[-1].strip()
    used_h_end_for_last = False
    if trailing:
        raw.append(trailing)
        states.append(cap['h_end'])
        used_h_end_for_last = True
    if not raw:
        raise ValueError('no_steps')
    steps = []
    for r in raw:
        m = re.findall(STEP_PATTERN, r + '\n', re.DOTALL)
        if len(m) != 1:
            raise ValueError(f'step_format:{len(m)}_matches')
        steps.append(f"{m[0][0]} Hint: {m[0][1].strip()}")
    return steps, raw, np.stack(states), used_h_end_for_last


# ---------------------------------------------------------------- call 2
@torch.no_grad()
def call_credit(model, tok, prompts, bench, problem, steps, max_new_tokens):
    sys_p = prompts['evaluate_question_level_batch']['system'].format(domain='math', benchmarks=bench)
    usr_p = prompts['evaluate_question_level_batch']['user'].format(problem=problem, steps=steps)
    msgs = [{'role': 'system', 'content': sys_p}, {'role': 'user', 'content': usr_p}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt').to(model.device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                         do_sample=False, temperature=None, top_p=None,
                         pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    gen = out[0, ids.shape[1]:].tolist()
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids('<|eot_id|>')}
    return {'text': tok.decode(gen, skip_special_tokens=True), 'input_tokens': int(ids.shape[1]),
            'output_tokens': len(gen), 'seconds': sec,
            'truncated': not (gen and gen[-1] in eos)}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard-id', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--per-level', type=int, default=0, help='smoke test: first N per level (0=all)')
    ap.add_argument('--out', default=str(DATA / 'gen'))
    ap.add_argument('--decomp-max-new', type=int, default=768)
    ap.add_argument('--credit-max-new', type=int, default=2048)
    args = ap.parse_args()

    out = Path(args.out)
    (out / 'problems').mkdir(parents=True, exist_ok=True)
    problems = json.loads((DATA / 'math_train_sample2000.json').read_text(encoding='utf-8'))
    if args.per_level:
        problems = [p for lv in range(1, 6) for p in [q for q in problems if q['level'] == lv][:args.per_level]]
    problems = [p for i, p in enumerate(problems) if i % args.num_shards == args.shard_id]

    excl_path = out / f'excluded_shard{args.shard_id}.jsonl'
    done = {int(p.stem) for p in (out / 'problems').glob('*.pt')}
    if excl_path.exists():
        done |= {json.loads(l)['problem_id'] for l in excl_path.read_text().splitlines() if l.strip()}
    todo = [p for p in problems if p['problem_id'] not in done]
    print(f'shard {args.shard_id}/{args.num_shards}: {len(problems)} assigned, {len(todo)} to do', flush=True)

    prompts = load_official_prompts()
    bench, bench_rec = get_benchmark_examples()
    print(f'benchmark unique_ids (seed 42): {bench_rec["unique_ids"]}', flush=True)

    print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
    tok = AutoTokenizer.from_pretrained(LLAMA)
    model = AutoModelForCausalLM.from_pretrained(LLAMA, torch_dtype=torch.float16, device_map='cuda').eval()

    for n, p in enumerate(todo):
        pid = p['problem_id']
        rec_ex = {'problem_id': pid, 'level': p['level'], 'type': p['type']}
        cap = c2 = call1 = call2 = None
        try:
            cap = generate_with_marker_states(
                model, tok, build_decomp_messages_with_marker(p['problem'], p['level']),
                LAYERS, max_new_tokens=args.decomp_max_new)
            call1 = {k: cap[k] for k in ('input_tokens', 'output_tokens', 'seconds')}
            try:
                steps, raw, H, used_end = assemble_steps(cap)
            except ValueError as e:
                raise ValueError(f'call1:{e}') from None
            c2 = call_credit(model, tok, prompts, bench, p['problem'], steps, args.credit_max_new)
            call2 = {k: c2[k] for k in ('input_tokens', 'output_tokens', 'seconds')}
            if c2['truncated']:
                raise ValueError('call2:truncated')
            try:
                levels, credits, plevel, flags = parse_labels(c2['text'], len(steps))
            except ValueError as e:
                raise ValueError(f'call2:{e}') from None
        except ValueError as e:
            rec_ex.update({'reason': str(e),
                           'decomposition_text': cap['text'] if cap else None,
                           'credit_response': c2['text'] if c2 else None,
                           'call1': call1, 'call2': call2})
            with excl_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(rec_ex, ensure_ascii=False) + '\n')
            print(f'[{pid}] EXCLUDED: {e}', flush=True)
            continue

        torch.save({
            'problem_id': pid, 'train_idx': p['train_idx'], 'level': p['level'], 'type': p['type'],
            'problem': p['problem'],
            'decomposition_text': cap['text'], 'gen_ids': cap['gen_ids'],
            'marker_tok_idx': cap['marker_tok_idx'],
            'raw_segments': raw, 'steps': steps, 'last_step_uses_h_end': used_end,
            'layers': LAYERS,
            'h_steps': H,                       # [n_steps, L, d] fp16
            'h_end': cap['h_end'],              # [L, d] fp16
            'h_prompt': cap['h_prompt'],        # [L, d] fp16
            'credit_response': c2['text'],
            'evaluated_levels': levels, 'credits': credits, 'problem_evaluated_level': plevel,
            'label_flags': flags,
            'call1': call1, 'call2': call2,
        }, out / 'problems' / f'{pid}.pt')
        print(f'[{pid}] ok L{p["level"]} steps={len(steps)} levels={levels} credits={credits} '
              f'flags={flags} t1={call1["seconds"]:.1f}s t2={call2["seconds"]:.1f}s', flush=True)
        if (n + 1) % 25 == 0:
            print(f'  progress {n+1}/{len(todo)}', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    sys.exit(main())
