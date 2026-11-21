"""Task 8 inference: one LLaMA call (marker decomposition) -> per-step difficulty
d_k from the saved probe -> token allocation with the SAME formula as
inference/_5_planned_local_weighted_model.allocate_tokens (w_k = d_k/sum d,
times polynomial decay (N-i)^2, renormalised, floor + remainder).

    from probe_inference_v2 import load_probe, plan_with_probe, allocate_tokens, polynomial_decay
"""
import json
from pathlib import Path

import numpy as np
import torch

from hidden_state_capture_v2 import build_decomp_messages_with_marker, generate_with_marker_states
from generate_training_data_v2 import assemble_steps

PROBE_DIR = Path(__file__).resolve().parent / 'probe_artifacts'


# ---- copied verbatim from inference/_5_planned_local_weighted_model.py ----
def allocate_tokens(credits, total_tokens, schedule_fn):
    N = len(credits)
    sched_weights = np.array([schedule_fn(i, N) for i in range(N)], dtype=float)
    combined = np.array(credits) * sched_weights
    combined /= combined.sum()
    tokens = np.floor(combined * total_tokens).astype(int)
    diff = total_tokens - tokens.sum()
    for i in np.argsort(-combined)[:diff]:
        tokens[i] += 1
    return tokens


def polynomial_decay(i, N, power=2):
    return (N - i)**power
# ---------------------------------------------------------------------------


class _MLP(torch.nn.Module):
    def __init__(self, d, hidden, dropout, **_):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.ReLU(),
                                       torch.nn.Dropout(dropout), torch.nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_probe(probe_dir=PROBE_DIR):
    p = torch.load(Path(probe_dir) / 'probe.pt', weights_only=False)
    meta = json.loads((Path(probe_dir) / 'metadata.json').read_text())
    if p['kind'] == 'mlp':
        m = _MLP(**p['mlp_dims']).eval()
        m.load_state_dict(p['mlp_state'])
        p['model'] = m
    return p, meta


def _features(p, H, h_end, h_prompt):
    # H: [n, d] at the probe layer
    prev = np.vstack([h_prompt[None], H[:-1]])
    end = np.repeat(h_end[None], len(H), 0)
    return {'F1': H, 'F2': H - prev, 'F3': np.hstack([H, end]), 'F4': np.hstack([H - prev, end])}[p['feature']]


def predict_difficulties(p, H, h_end, h_prompt):
    X = (_features(p, H, h_end, h_prompt).astype(np.float32) - p['scaler_mu']) / p['scaler_sd']
    if p['kind'] == 'ridge':
        d = X @ p['ridge']['w'] + p['ridge']['b']
    else:
        with torch.no_grad():
            d = p['model'](torch.as_tensor(X, dtype=torch.float32)).numpy()
    return np.clip(d, 0.1, None).tolist()      # same clip as used in evaluation shares


def plan_with_probe(model, tok, p, problem, level, max_new_tokens=768):
    """ONE LLaMA generation. Returns dict with steps (official format), d_k list,
    and call-1 token/time stats. Raises ValueError on marker/format failure."""
    L = p['layer']
    cap = generate_with_marker_states(model, tok, build_decomp_messages_with_marker(problem, level),
                                      [L], max_new_tokens=max_new_tokens)
    steps, raw, H, used_end = assemble_steps(cap)
    d = predict_difficulties(p, H[:, 0].astype(np.float32), cap['h_end'][0].astype(np.float32),
                             cap['h_prompt'][0].astype(np.float32))
    return {'steps': steps, 'difficulties': d, 'last_step_uses_h_end': used_end,
            'call1': {k: cap[k] for k in ('input_tokens', 'output_tokens', 'seconds')}}
