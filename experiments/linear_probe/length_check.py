"""Synthetic length ramp: CPU plan, then ONE fresh GPU process per condition."""
import argparse
import time
from pathlib import Path

from .common import (REVISION, MODEL, decomp_messages, prompt_fingerprint, runtime,
                     digest, read, verify, seal, save_new, output_path)


def synthetic_row(repeats):
    return {'id': f'synthetic-labels-{repeats}', 'level': 1, 'query':
        'A toy robot has 3 boxes with 4 blue blocks in each box. It gives away 2 blocks. '
        'The following catalogue labels are identifiers only and do not change the block counts:\n'
        + 'blue amber crate label; ' * repeats + '\nHow many blocks remain?'}


def condition(tok, target):
    def count(n):
        return len(tok.apply_chat_template(decomp_messages(synthetic_row(n)), tokenize=True, add_generation_prompt=True))
    lo, hi = 0, target
    if count(0) > target:
        raise ValueError('Target shorter than original prompt')
    while lo + 1 < hi:
        middle = (lo + hi) // 2
        if count(middle) <= target:
            lo = middle
        else:
            hi = middle
    repeats = min((lo, hi), key=lambda n: abs(count(n)-target))
    actual = count(repeats)
    if abs(actual-target) > 16:
        raise ValueError('Cannot reach target tolerance')
    return {'target_tokens': target, 'prompt_tokens': actual, 'row': synthetic_row(repeats),
            'messages': decomp_messages(synthetic_row(repeats))}


def previous_gate(plan, target, previous, threshold):
    if target == 1024:
        return
    required = {2048:1024, 4096:2048}[target]
    if previous is None:
        raise ValueError('Sequential ramp requires the previous condition report')
    doc = verify(read(previous))
    if doc['plan_fingerprint'] != plan['fingerprint'] or doc['target_tokens'] != required:
        raise ValueError('Previous report belongs to another plan/condition')
    if doc['status'] != 'success' or not doc['safe_to_extend'] or doc['peak_reserved_bytes']/2**30 > threshold:
        raise ValueError('Previous condition failed or exceeded memory guard; do not extend automatically')


def failure_kind(exc):
    text = str(exc).lower()
    if 'insufficient free gpu' in text:
        return 'resource_busy'
    if type(exc).__name__ == 'OutOfMemoryError' or 'out of memory' in text:
        return 'oom'
    if 'context exceeded' in text:
        return 'context_exceeded'
    if 'truncated' in text:
        return 'generation_truncated'
    if any(x in text for x in ('subquestions', 'processed token', 'alignment mismatch')):
        return 'parse_failure'
    return 'runtime_error'


def run(a):
    plan = verify(read(a.plan))
    path = output_path(a.output)
    if path.exists():
        raise FileExistsError(path)
    previous_gate(plan, a.target, a.previous, a.max_reserved_gib)
    item = next(r for r in plan['conditions'] if r['target_tokens'] == a.target)
    if plan['prompt_fingerprint'] != prompt_fingerprint() or plan['revision'] != REVISION:
        raise ValueError('Prompt/model changed since CPU preparation')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .capture import generate
    doc = {'kind': 'synthetic_length_result', 'target_tokens': a.target,
        'plan_fingerprint': plan['fingerprint'], 'settings': plan['settings'],
        'prompt_tokens': item['prompt_tokens'], 'completion_tokens': 0,
        'status': 'running', 'parse_success': False, 'safe_to_extend': False,
        'phase': 'preflight', 'runtime': runtime()}
    trace = {}
    started = time.perf_counter()
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError('Require exactly one visible GPU')
        doc['gpu'] = torch.cuda.get_device_name(0)
        if '3090' not in doc['gpu']:
            raise ValueError('This validation requests an RTX 3090')
        free, total = torch.cuda.mem_get_info()
        doc.update(initial_free_bytes=free, device_total_bytes=total)
        if free < 22 * 2**30:
            raise RuntimeError('Insufficient free GPU memory: require 22 GiB before model load')
        tok = AutoTokenizer.from_pretrained(a.model_path, local_files_only=True)
        actual = len(tok.apply_chat_template(item['messages'], tokenize=True, add_generation_prompt=True))
        if actual != item['prompt_tokens']:
            raise ValueError('Tokenizer differs from prepared length plan')
        torch.cuda.reset_peak_memory_stats()
        doc['phase'] = 'model_load'; begin = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(a.model_path, local_files_only=True,
                    torch_dtype=torch.float16, attn_implementation='eager').to('cuda').eval()
        torch.cuda.synchronize()
        doc.update(model_load_seconds=time.perf_counter()-begin,
                   model_load_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   model_load_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        torch.cuda.reset_peak_memory_stats()
        doc['phase'] = 'generation'
        begin = time.perf_counter()
        cap = generate(model, tok, item['messages'], layers=list(range(33)),
            max_new_tokens=plan['settings']['max_new_tokens'], max_context=8192, diagnostics=trace)
        import numpy as np
        if any(np.asarray(v).shape != (4096,) or not np.isfinite(v).all()
               for feature in cap['features'] for v in feature.values()):
            raise ValueError('Invalid feature values/dimension')
        doc.update(status='success', parse_success=True, capture=cap,
                   completion_tokens=cap['usage']['completion_tokens'])
    except Exception as exc:
        doc.update(status=failure_kind(exc), error={'type':type(exc).__name__, 'message':str(exc)[:1500]},
                   completion_tokens=len(trace.get('generated_ids', [])), diagnostics=trace)
    finally:
        if torch.cuda.is_available():
            doc.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved())
        doc['wall_seconds_including_load'] = time.perf_counter()-started
        doc['generation_wall_seconds'] = time.perf_counter()-begin if doc['phase']=='generation' else None
        doc['safe_to_extend'] = doc['status']=='success' and doc.get('peak_reserved_bytes', 2**60) <= a.max_reserved_gib*2**30
        doc['memory_guard_gib'] = a.max_reserved_gib
        save_new(path, seal(doc))
    if doc['status'] != 'success' or not doc['safe_to_extend']:
        raise RuntimeError(f"{doc['status']}; report preserved at {path}; stop ramp")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('plan'); a.add_argument('--model-path',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True)
    a = sub.add_parser('run'); a.add_argument('--model-path',type=Path,required=True)
    a.add_argument('--plan',type=Path,required=True); a.add_argument('--target',type=int,choices=(1024,2048,4096),required=True)
    a.add_argument('--previous',type=Path); a.add_argument('--max-reserved-gib',type=float,default=21.)
    a.add_argument('--output',type=Path,required=True)
    args = p.parse_args(argv)
    if args.model_path.resolve().name != REVISION:
        raise ValueError('Use the pinned local Llama snapshot')
    if args.command == 'plan':
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_path,local_files_only=True)
        save_new(args.output,seal({'kind':'synthetic_length_plan','model':MODEL,'revision':REVISION,
            'prompt_fingerprint':prompt_fingerprint(), 'conditions':[condition(tok,n) for n in (1024,2048,4096)],
            'settings':{'dtype':'float16','attention':'eager','layers':list(range(33)),
                        'max_new_tokens':512,'max_context':8192,'temperature':.1,'top_p':.9,'seed':42},
            'reference_smoke':{'job':219600,'prompt_tokens':650,'completion_tokens':127,
                'steps':4,'peak_allocated_gib':15.47,'peak_reserved_gib':15.58,'generation_seconds':5.89,
                'source':'user-reported PASS; not rerun by Codex'}}))
    else:
        run(args)


if __name__ == '__main__':
    main()
