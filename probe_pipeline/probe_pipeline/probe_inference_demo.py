#!/usr/bin/env python3
"""End-to-end demo: after the probe is trained, this is the ENTIRE planning
stage for a new problem -- exactly one LLaMA call (STEP 1), no STEP 2.

Compare this to the original PAB pipeline (two LLaMA calls per problem) and
to generate_training_data.py (which still needs STEP 2, but only to build
the training label, once).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hidden_state_capture import build_decomp_messages_with_marker, generate_with_step_hidden_states

import os
LLAMA = os.environ.get('LLAMA_PATH', '/data/yunse1203/hf_cache/llama-3.1-8b-instruct')
PROBE_DIR = Path('probe_artifacts')


def load_probe():
    pipeline = joblib.load(PROBE_DIR / 'best_probe.joblib')
    meta = json.loads((PROBE_DIR / 'probe_metadata.json').read_text())
    return pipeline, meta['best_layer_idx']


def plan_with_probe(model, tok, pipeline, best_layer_idx, problem: str, level: int):
    """Returns (steps, predicted_credits) using exactly ONE LLaMA call."""
    messages = build_decomp_messages_with_marker(problem, level)
    text, hidden_states_per_step = generate_with_step_hidden_states(model, tok, messages)

    from hidden_state_capture import parse_steps_from_marked_text
    steps = parse_steps_from_marked_text(text)

    X = np.stack([hs[best_layer_idx] for hs in hidden_states_per_step])
    X_scaled = pipeline['scaler'].transform(X)
    predicted_credits = pipeline['regressor'].predict(X_scaled)

    # Normalise to sum to 100, matching the original credit convention.
    total = predicted_credits.sum()
    predicted_credits = (predicted_credits / total * 100).tolist()

    return steps, predicted_credits


if __name__ == '__main__':
    tok = AutoTokenizer.from_pretrained(LLAMA)
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA, torch_dtype=torch.float16, device_map='auto'
    ).eval()
    pipeline, best_layer_idx = load_probe()

    example_problem = "Find the remainder when 9 * 99 * 999 * ... (999 nines) is divided by 1000."
    steps, credits = plan_with_probe(model, tok, pipeline, best_layer_idx, example_problem, level=3)

    for s, c in zip(steps, credits):
        print(f'[{c:5.1f} credit] {s}')
