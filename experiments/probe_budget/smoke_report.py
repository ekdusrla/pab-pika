#!/usr/bin/env python3
"""Task 3 report for a generate_training_data_v2.py output dir (CPU only)."""
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from hidden_state_capture_v2 import STEP_MARKER

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else 'data_v2/gen_smoke')
N_TOTAL = 2000
tok = AutoTokenizer.from_pretrained(os.environ.get('LLAMA_PATH', 'meta-llama/Llama-3.1-8B-Instruct'))

kept = [torch.load(p, weights_only=False) for p in sorted((OUT / 'problems').glob('*.pt'), key=lambda p: int(p.stem))]
excl = [json.loads(l) for f in OUT.glob('excluded_shard*.jsonl') for l in f.read_text().splitlines() if l.strip()]
n = len(kept) + len(excl)
print(f'problems: {n}  kept={len(kept)}  excluded={len(excl)}')
reasons = Counter(e['reason'].split(':')[0] + ':' + e['reason'].split(':')[1] if e['reason'].count(':') else e['reason'] for e in excl)
print('exclusion reasons:', dict(reasons))
for e in excl:
    print(f"  pid={e['problem_id']} L{e['level']} reason={e['reason']}")
call1_fail = sum(e['reason'].startswith('call1') for e in excl)
call2_fail = sum(e['reason'].startswith('call2') for e in excl)
marker_mis = sum(('marker' in e['reason'] or 'step_format' in e['reason'] or 'empty_segment' in e['reason']) for e in excl)
print(f'call-1 failures (marker/format/truncation): {call1_fail}/{n} = {call1_fail/n:.1%}'
      f'   of which marker/step-format: {marker_mis}')
print(f'call-2 parse failures: {call2_fail}/{n - call1_fail} reaching call 2 = {call2_fail/max(1, n-call1_fail):.1%}')
print(f'kept problems whose last step had no marker (used h_end): '
      f'{sum(k["last_step_uses_h_end"] for k in kept)}/{len(kept)}')
print('label flags:', Counter(f.split(':')[0] for k in kept for f in k['label_flags']))
print('steps/problem:', dict(sorted(Counter(len(k['steps']) for k in kept).items())))
print('evaluated_level dist:', dict(sorted(Counter(l for k in kept for l in k['evaluated_levels']).items())))

def stat(xs):
    return f'mean={np.mean(xs):.2f} median={np.median(xs):.2f} max={np.max(xs):.2f}'
allc1 = [k['call1'] for k in kept] + [e['call1'] for e in excl if e.get('call1')]
allc2 = [k['call2'] for k in kept] + [e['call2'] for e in excl if e.get('call2')]
for name, cs in [('call1', allc1), ('call2', allc2)]:
    print(f'{name}: sec {stat([c["seconds"] for c in cs])} | in_tok {stat([c["input_tokens"] for c in cs])} '
          f'| out_tok {stat([c["output_tokens"] for c in cs])}')
per_prob = np.mean([c['seconds'] for c in allc1]) + np.mean([c['seconds'] for c in allc2])
print(f'avg sec/problem (call1+call2) = {per_prob:.1f}s  -> {N_TOTAL} problems on 1 GPU = '
      f'{per_prob*N_TOTAL/3600:.1f} h')

sizes = [p.stat().st_size for p in (OUT / 'problems').glob('*.pt')]
print(f'storage: {sum(sizes)/1e6:.1f} MB for {len(sizes)} kept; mean {np.mean(sizes)/1e6:.2f} MB/problem '
      f'-> ~{np.mean(sizes)*N_TOTAL/1e9:.2f} GB for {N_TOTAL}')
k0 = kept[0]
print(f'h_steps shape {k0["h_steps"].shape} dtype {k0["h_steps"].dtype}; h_end {k0["h_end"].shape}')

# ---- capture-position check on 3 problems
print('\n=== capture positions (token at captured index shown in [[ ]]) ===')
picks = [k for k in kept if k['last_step_uses_h_end']][:1] + [k for k in kept if not k['last_step_uses_h_end']][:2]
for k in picks[:3]:
    ids = k['gen_ids']
    print(f'\n--- pid {k["problem_id"]} (L{k["level"]}) steps={len(k["steps"])} '
          f'markers={len(k["marker_tok_idx"])} last_step_uses_h_end={k["last_step_uses_h_end"]}')
    for j in k['marker_tok_idx']:
        before = tok.decode(ids[max(0, j - 6):j])
        at = tok.convert_ids_to_tokens(ids[j])
        after = tok.decode(ids[j + 1:j + 3])
        print(f'  marker@{j}: ...{before!r} [[{at!r}]] {after!r}')
    j = len(ids) - 2   # last pre-EOS token
    print(f'  h_end@{j}: ...{tok.decode(ids[max(0, j-6):j])!r} [[{tok.convert_ids_to_tokens(ids[j])!r}]] '
          f'then EOS={tok.convert_ids_to_tokens(ids[-1])!r}')
    print('  official-format steps sent to call 2:')
    for s in k['steps']:
        print('   ', s[:160].replace('\n', '\\n'))
    print('  evaluated_levels:', k['evaluated_levels'], ' credits:', k['credits'])
