"""v2 capture: manual greedy decoding with KV cache that copies hidden states to
CPU ONLY at (a) the last prompt token, (b) each token that completes a
<<<STEP>>> marker, (c) the final token before EOS (h_end).

Capture convention ("marker-completing token"):
  When generated token t_j completes a marker (the number of STEP_MARKER
  occurrences in the decoded text increases), we capture the hidden state AT
  POSITION j, i.e. the representation of t_j itself. This is produced by the
  NEXT forward pass (the one that takes t_j as input), so the capture is
  deferred by one iteration. Hence the representation has "read" the full
  marker.

  h_end: hidden state at the position of the last token before EOS (the
  forward pass whose logits emitted EOS). It has seen the whole plan.
  If generation stops by max_new_tokens instead of EOS, `stopped_by_eos` is
  False and the caller should treat the sample as truncated.

  h_prompt: hidden state at the last prompt token (needed for F2, k=1).

The decomposition prompt is unchanged from hidden_state_capture.py.
"""
import time

import numpy as np
import torch

from hidden_state_capture import STEP_MARKER, build_decomp_messages_with_marker  # noqa: F401


def _grab(hidden_states, layers):
    # hidden_states: tuple(n_layers+1) of [1, seq, d] -> np.float16 [len(layers), d]
    return torch.stack([hidden_states[l][0, -1, :] for l in layers]).to(torch.float16).cpu().numpy()


@torch.no_grad()
def generate_with_marker_states(model, tok, messages, layers, max_new_tokens=768):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tok(prompt, return_tensors='pt', add_special_tokens=False).input_ids.to(model.device)
    eos_ids = set([tok.eos_token_id] + ([tok.convert_tokens_to_ids('<|eot_id|>')] or []))

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    past = None
    next_input = input_ids
    gen_ids, marker_states, marker_tok_idx = [], [], []
    h_prompt = h_end = None
    pending_capture = None      # index j of a marker-completing token awaiting its state
    n_markers_seen = 0
    stopped_by_eos = False

    for step in range(max_new_tokens + 1):
        out = model(input_ids=next_input, past_key_values=past, use_cache=True,
                    output_hidden_states=True)
        past = out.past_key_values
        if step == 0:
            h_prompt = _grab(out.hidden_states, layers)
        if pending_capture is not None:
            # this forward's input was the marker-completing token -> its state
            marker_states.append(_grab(out.hidden_states, layers))
            marker_tok_idx.append(pending_capture)
            pending_capture = None
        if step == max_new_tokens:
            break

        nxt = int(out.logits[0, -1, :].argmax())
        if nxt in eos_ids:
            # state at the last pre-EOS position (input of this forward)
            h_end = _grab(out.hidden_states, layers)
            gen_ids.append(nxt)
            stopped_by_eos = True
            break
        gen_ids.append(nxt)

        text = tok.decode(gen_ids, skip_special_tokens=True)
        c = text.count(STEP_MARKER)
        if c > n_markers_seen:
            if c - n_markers_seen > 1:
                raise RuntimeError('two markers completed by one token')
            n_markers_seen = c
            pending_capture = len(gen_ids) - 1
        next_input = torch.tensor([[nxt]], device=model.device)

    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    full_text = tok.decode(gen_ids, skip_special_tokens=True)
    return {
        'text': full_text,
        'gen_ids': gen_ids,
        'marker_states': marker_states,          # list of [L, d] fp16
        'marker_tok_idx': marker_tok_idx,        # index into gen_ids
        'h_prompt': h_prompt,
        'h_end': h_end,
        'stopped_by_eos': stopped_by_eos,
        'input_tokens': int(input_ids.shape[1]),
        'output_tokens': len(gen_ids),           # includes the EOS token if emitted
        'seconds': seconds,
    }
