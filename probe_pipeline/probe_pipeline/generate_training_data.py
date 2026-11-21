#!/usr/bin/env python3
"""Build (hidden_state, credit) training pairs for the probe.

This script LOADS the model and drives the loop; hidden_state_capture.py
only supplies the decoding function. Two LLM calls per problem happen here,
same as the original PAB pipeline:

  STEP 1: generate_with_step_hidden_states()  -- decomposition + hidden
          states captured for free during this single generation call.
  STEP 2: a normal credit-assessment call      -- this call's OUTPUT (the
          credit numbers) becomes the probe's training LABEL. It is only
          needed while building the training set; once the probe is
          trained, STEP 2 is never called again at inference time.

Output: probe_training_data.pt
    A flat list of dicts, one per sub-question:
    {"problem_id": int, "step_idx": int, "step_text": str,
     "hidden_states": {layer_idx: np.ndarray}, "credit": float}
"""
import json
import os
import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from hidden_state_capture import (
    build_decomp_messages_with_marker,
    generate_with_step_hidden_states,
    parse_steps_from_marked_text,
)

import os
LLAMA = os.environ.get('LLAMA_PATH', '/data/yunse1203/hf_cache/llama-3.1-8b-instruct')
OUT_PATH = Path('probe_training_data.pt')

# Original P-and-B paper (Appendix H) credit-evaluation prompt, verbatim.
# NOTE: the paper's template includes a <benchmarks> placeholder meant to be
# filled with worked reference examples of each difficulty level; we do not
# have that benchmark content, so it is left empty below. If you obtain the
# paper's actual benchmark examples, insert them where marked.
CREDIT_SYSTEM = (
    "You are an experienced expert in math and exam question designer. Your task is to evaluate "
    "the difficulty level of a given exam problem and its sub-questions by comparing it "
    "against a set of benchmark questions of known levels.\n"
    "Based on their levels, you will need to assign each subquestion a portion of the credits "
    "(assuming the total credit points is 100 for the whole problem).\n"
    "Each level reflects increasing complexity from 1 (easiest) to 5 (most challenging).\n"
    "Evaluate based on the conceptual depth, steps involved in solving, required knowledge, and "
    "potential for misdirection.\n"
    "Use the following benchmark examples as references:\n"
    "<benchmarks>\n"  # TODO: paste the paper's actual benchmark examples here if available
    "1. You will be provided a question and its subquestions. You will evaluate the difficulty "
    "level of the problem and its sub-questions. Assuming the whole problem is worth 100 "
    "points, you assign each sub-question a portion of the score points.\n"
    "- Adhere to the given subquestions, and DO NOT make new subquestions.\n"
    "- Sum of each subquestion's credits MUST EQUAL to 100.\n"
    "2. You must return the result in a structured JSON format:\n"
    "{\n"
    '  "problem": {"reason": "...", "evaluated_level": level_q},\n'
    '  "1": {"reason": "...", "evaluated_level": level_1, "credit": credit_1},\n'
    '  "2": {"reason": "...", "evaluated_level": level_2, "credit": credit_2},\n'
    "  ...\n"
    "}\n"
    "where\n"
    '- "reason": a short explanation (up to 50 words) of your level assessment.\n'
    '- "evaluated_level": an integer from 1 to 5 indicating your judgment.\n'
    '- "credit": an integer between 1 to 100 indicating when the question is solved correctly, '
    "how many credit can be given.\n"
)


def build_credit_messages(problem: str, steps: list[str]) -> list[dict]:
    steps_text = '\n'.join(f'{i+1}. {s}' for i, s in enumerate(steps))
    return [
        {'role': 'system', 'content': CREDIT_SYSTEM},
        {'role': 'user', 'content': (
            f"Evaluate the level of the following question:\n"
            f"Problem: {problem}\nSub-questions: {steps_text}\nOutput:"
        )},
    ]


def _parse_credit_json(text: str, n_steps: int) -> list[float]:
    """Extract per-sub-question 'credit' from the original paper's nested
    JSON schema: {"problem": {...}, "1": {"reason":..., "credit": N}, ...}.
    The 'reason' fields make responses much longer than a flat format, so
    this is robust to: the model writing prose before the JSON (take the
    LAST {...} block), and truncated/malformed JSON (regex salvage)."""
    b = text.rfind('}')
    if b < 0:
        a = text.find('{')
        raw = text[a:] if a >= 0 else ''
    else:
        depth, a = 0, b
        for i in range(b, -1, -1):
            if text[i] == '}':
                depth += 1
            elif text[i] == '{':
                depth -= 1
                if depth == 0:
                    a = i
                    break
        raw = text[a:b + 1]
    raw = re.sub(r',\s*([\]}])', r'\1', raw)  # trailing commas

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        salvaged = re.findall(
            r'"(\d+)"\s*:\s*\{[^{}]*"credit"\s*:\s*(\d+(?:\.\d+)?)[^{}]*\}', raw, re.DOTALL
        )
        if not salvaged:
            print('  [assess_credits] WARNING: unparseable/truncated JSON, equal split', flush=True)
            return [100.0 / n_steps] * n_steps
        obj = {k: {'credit': float(v)} for k, v in salvaged}

    credits = []
    for idx in range(1, n_steps + 1):
        entry = obj.get(str(idx))
        if not isinstance(entry, dict) or 'credit' not in entry:
            print(f'  [assess_credits] WARNING: step {idx} missing, equal credit', flush=True)
            credits.append(100.0 / n_steps)
            continue
        try:
            credits.append(float(entry['credit']))
        except (TypeError, ValueError):
            credits.append(100.0 / n_steps)
    return credits


@torch.no_grad()
def assess_credits(model, tok, problem: str, steps: list[str]) -> list[float]:
    """STEP 2: plain generation (no hidden states needed here -- only used
    to produce the training label). max_new_tokens is generous because the
    original prompt's JSON includes a 'reason' field per sub-question, which
    runs much longer than a flat {"1": 40, ...} response would."""
    messages = build_credit_messages(problem, steps)
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors='pt').to(model.device)
    out_ids = model.generate(
        **inputs, max_new_tokens=1024, do_sample=False, pad_token_id=tok.eos_token_id,
    )
    text = tok.decode(out_ids[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    return _parse_credit_json(text, len(steps))


def main():
    print(f'Loading {LLAMA} ...', flush=True)
    tok = AutoTokenizer.from_pretrained(LLAMA)
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA, torch_dtype=torch.float16, device_map='auto'
    ).eval()

    ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
    print(f'{len(ds)} problems loaded.', flush=True)

    training_pairs = []

    for i, item in enumerate(ds):
        problem, level = item['problem'], item['level']

        # ---- STEP 1: decomposition + hidden states, ONE call ----
        messages = build_decomp_messages_with_marker(problem, level)
        text, hidden_states_per_step = generate_with_step_hidden_states(model, tok, messages)
        steps = parse_steps_from_marked_text(text)

        if not steps or len(steps) != len(hidden_states_per_step):
            print(f'[{i}] marker/step count mismatch '
                  f'(steps={len(steps)}, markers={len(hidden_states_per_step)}), skipping',
                  flush=True)
            continue

        # ---- STEP 2: credit assessment, used ONLY to create the label ----
        credits = assess_credits(model, tok, problem, steps)

        for step_idx, (step_text, hs, credit) in enumerate(
            zip(steps, hidden_states_per_step, credits)
        ):
            training_pairs.append({
                'problem_id': i, 'step_idx': step_idx, 'step_text': step_text,
                'hidden_states': hs, 'credit': credit,
            })

        if (i + 1) % 25 == 0:
            print(f'  processed {i+1}/{len(ds)} problems, '
                  f'{len(training_pairs)} training pairs so far', flush=True)

    torch.save(training_pairs, OUT_PATH)
    print(f'Saved {len(training_pairs)} (hidden_state, credit) pairs to {OUT_PATH}', flush=True)


if __name__ == '__main__':
    main()
