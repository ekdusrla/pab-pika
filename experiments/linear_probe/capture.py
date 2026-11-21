"""Unmodified decomposition prompt; hidden states reused from generation forwards.

No imports of torch/transformers at module import. Model loading is CLI-explicit.
Feature h[t] is the output for generated token t after it has been processed,
not the preceding position that predicted t. EOS completes the last token's
processing, so normal completion needs no additional forward.
"""
import re
import time

from .common import FEATURE, digest, parse_steps


def step_positions(text, prefixes):
    """Map step spans to the last wholly contained, processed token.

    Prefixes are exact decode(ids[:t+1]), including tokenizer boundary effects.
    Reject unstable text offsets rather than silently assigning another step.
    """
    steps = parse_steps(text)
    matches = list(re.finditer(r'(?m)^\s*\d+\.\s+(.+?)(?=^\s*\d+\.\s+|\Z)', text, re.DOTALL))
    positions = []
    for match in matches:
        start = match.start(1)
        end = len(text[:match.end()].rstrip())
        candidates = [i for i, prefix in enumerate(prefixes)
                      if start < len(prefix) <= end and text.startswith(prefix)]
        if not candidates:
            raise ValueError('No stable processed token inside step')
        positions.append(candidates[-1])
    if len(positions) != len(steps) or len(set(positions)) != len(steps):
        raise ValueError('Step/token alignment mismatch')
    return steps, positions


def generate(model, tok, messages, layers=None, max_new_tokens=1024,
             temperature=0.1, top_p=0.9, seed=42, capture=True, max_context=8192,
             diagnostics=None, memory_check=None):
    import torch
    if temperature <= 0 or not 0 < top_p <= 1 or max_new_tokens < 1:
        raise ValueError('Invalid decoding settings')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = next(model.parameters()).device
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    if not eos:
        raise ValueError('Missing model EOS configuration')
    selected = list(range(model.config.num_hidden_layers + 1)) if layers is None else sorted(set(layers))
    if not selected or any(i < 0 or i > model.config.num_hidden_layers for i in selected):
        raise ValueError('Layer index out of range')
    # Tokenize through the chat template once: no accidental duplicate BOS.
    inputs = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt').to(device)
    if diagnostics is not None:
        diagnostics.update(prompt_tokens=inputs.shape[1], generated_ids=[], forward_calls=0)
    if inputs.shape[1] + max_new_tokens > min(max_context, model.config.max_position_embeddings):
        raise ValueError('Model context exceeded')
    ids, retained, past = [], [], None
    if diagnostics is not None:
        diagnostics['generated_ids'] = ids  # Preserve actual tokens even on truncation/parse failure.
    forward_seconds = sampling_seconds = retention_seconds = 0.0
    next_input = inputs
    if memory_check is not None:
        memory_check('before_generation')
    sync()
    started = time.perf_counter()
    with torch.inference_mode():
        for index in range(max_new_tokens):
            sync(); begin = time.perf_counter()
            out = model(input_ids=next_input, past_key_values=past, use_cache=True, output_hidden_states=capture)
            if diagnostics is not None:
                diagnostics['forward_calls'] += 1
            sync(); forward_seconds += time.perf_counter() - begin
            if diagnostics is not None:
                diagnostics.update(phase='model_forward', forward_seconds=forward_seconds)
            if memory_check is not None:
                memory_check('model_forward')
            past = out.past_key_values
            if capture and index > 0:
                begin = time.perf_counter()
                retained.append({layer: out.hidden_states[layer][0, -1].detach().clone() for layer in selected})
                sync(); retention_seconds += time.perf_counter() - begin
                if diagnostics is not None:
                    diagnostics.update(phase='hidden_retention', hidden_retention_seconds=retention_seconds)
                if memory_check is not None:
                    memory_check('hidden_retention')
            begin = time.perf_counter()
            logits = out.logits[0, -1].float() / temperature
            sorted_logits, indices = torch.sort(logits, descending=True)
            probabilities = torch.softmax(sorted_logits, dim=-1)
            remove = probabilities.cumsum(-1) - probabilities > top_p
            sorted_logits[remove] = -float('inf')
            token = indices[torch.multinomial(torch.softmax(sorted_logits, -1), 1)]
            token_id = token.item()
            sync(); sampling_seconds += time.perf_counter() - begin
            ids.append(token_id)
            if diagnostics is not None:
                diagnostics.update(phase='sampling', sampling_seconds=sampling_seconds,
                                   generation_seconds=time.perf_counter()-started)
            if memory_check is not None:
                memory_check('sampling')
            if token_id in eos:
                break
            next_input = token.reshape(1, 1)
        sync()
        generation_seconds = time.perf_counter() - started
        if diagnostics is not None:
            diagnostics.update(generation_seconds=generation_seconds, phase='step_alignment')
        begin = time.perf_counter()
        text = tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        prefixes = [tok.decode(ids[:i+1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    for i in range(len(ids) - 1)]
        if ids[-1] not in eos:
            raise ValueError('Decomposition truncated: no features/labels accepted')
        steps, positions = step_positions(text, prefixes)
        alignment_seconds = time.perf_counter() - begin
        if diagnostics is not None:
            diagnostics.update(parse_success=True, n_steps=len(steps), alignment_seconds=alignment_seconds,
                               phase='feature_transfer')
        begin = time.perf_counter()
        features = [{str(layer): retained[position][layer].float().cpu().tolist() for layer in selected}
                    for position in positions] if capture else None
        sync()
        transfer_seconds = time.perf_counter() - begin
        if diagnostics is not None:
            diagnostics.update(feature_transfer_seconds=transfer_seconds, phase='complete')
        if memory_check is not None:
            memory_check('feature_transfer')
    return {'text': text, 'steps': steps, 'features': features, 'feature_positions': positions,
            'feature_definition': FEATURE, 'prompt_ids_sha256': digest(inputs[0].tolist()),
            'generated_ids': ids, 'finish_reason': 'stop',
            'usage': {'prompt_tokens': inputs.shape[1], 'completion_tokens': len(ids),
                      'latency_seconds': generation_seconds},
            'metrics': {'logical_generation_calls': 1, 'forward_calls': len(ids),
                        'extra_forward_passes': 0, 'feature_forward_input_tokens': 0,
                        'forward_seconds': forward_seconds, 'sampling_seconds': sampling_seconds,
                        'hidden_retention_seconds': retention_seconds,
                        'alignment_seconds': alignment_seconds, 'feature_transfer_seconds': transfer_seconds,
                        'generation_seconds': generation_seconds,
                        'generation_plus_feature_seconds': generation_seconds + alignment_seconds + transfer_seconds,
                        'capture_enabled': capture, 'captured_layers': selected if capture else [],
                        'timing_policy': 'synchronize each measured GPU interval; includes instrumentation overhead',
                        'hidden_state_forward_overhead_seconds': None}}


def capture_pair(model, tok, messages, **kwargs):
    """Optional matched no-capture control; explicitly costs another generation.

    The two runs share RNG seed. Token equality is required before reporting a
    timing delta, which remains noisy/order-dependent, not a causal exact cost.
    """
    control = generate(model, tok, messages, capture=False, **kwargs)
    result = generate(model, tok, messages, capture=True, **kwargs)
    same = control['generated_ids'] == result['generated_ids']
    result['capture_control'] = {k: control[k] for k in ('usage', 'metrics', 'generated_ids')}
    result['capture_control']['same_generated_ids'] = same
    result['capture_control']['capture_minus_control_seconds'] = (
        result['metrics']['generation_plus_feature_seconds'] - control['metrics']['generation_plus_feature_seconds']) if same else None
    return result
