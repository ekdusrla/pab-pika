"""Supplemental TP=4 policy audit; reuse, do not rerun, credit audit 220305."""
import argparse
import copy
import importlib.metadata
import os
from pathlib import Path

from ..common import ROOT,read,verify,seal,save_new,output_path,file_hash
from ..credit_v1 import audit as base
from experiment_state import atomic_save,lock

SOURCE=base.SOURCE
OUT=ROOT/'experiments/linear_probe/runs/credit_a2000_tp4'
BASE_AUDIT=base.OUT/'audit_220305.json'
SERVER=copy.deepcopy(base.SERVER)
SERVER['server_settings'].update(tensor_parallel_size=4,distributed_executor_backend='mp',
    gpu_profile='rtx_a2000_12gb_x4')
POLICY={'profile':'rtx_a2000_12gb_x4','gpu_count':4,'tensor_parallel_size':4,
    'gpu_name':'NVIDIA RTX A2000 12GB','minimum_total_gib':11,'maximum_total_gib':13,
    'minimum_free_gib_per_gpu':10.,'maximum_observed_used_gib_per_gpu':11.,
    'cpus_per_gpu':6,'ram_gib_per_gpu':12,'executor':'mp','vllm_version':'0.6.6.post1',
    'single_node':True,'gpu_memory_utilization':.85}
require_compute=base.require_compute
capture_projection=base.capture_projection


def code_hashes():
    return {str(p.relative_to(ROOT)):file_hash(p) for pattern in ('*.py','*.sbatch')
            for p in Path(__file__).parent.glob(pattern)}


def check_architecture(config):
    dims={k:config[k] for k in ('hidden_size','num_attention_heads','num_key_value_heads','intermediate_size','vocab_size')}
    if any(v%4 for v in dims.values()) or dims['num_key_value_heads']<4:
        raise ValueError('Llama dimensions do not support the frozen TP=4 partition')
    return dims


def approve(path):
    doc=verify(read(path))
    if doc.get('kind')!='a2000_tp4_supplemental_audit' or not doc.get('passed'):raise ValueError('Supplemental TP4 audit PASS required')
    old=base.approve(BASE_AUDIT)
    if old['fingerprint']!=doc['base_audit_fingerprint'] or file_hash(BASE_AUDIT)!=doc['base_audit_sha256']:
        raise ValueError('Original accepted audit changed')
    if doc['code_sha256']!=code_hashes() or doc['gpu_policy']!=POLICY or doc['teacher_settings']!=SERVER:
        raise ValueError('TP4 policy/code changed; do not mix TP1/TP4 labels')
    for p,h in doc['runtime_source_sha256'].items():
        if file_hash(p)!=h:raise ValueError('vLLM runtime source changed')
    return doc


def run(output):
    require_compute();output=output_path(output)
    with lock(output):
        if Path(str(output)+'.tmp').exists():raise FileExistsError('Uncommitted supplemental audit')
        doc={'kind':'a2000_tp4_supplemental_audit','passed':False};save_new(output,seal(doc))
        try:
            original=base.approve(BASE_AUDIT)  # Seal/hash verification only, no rerun of the full audit/tests.
            if importlib.metadata.version('vllm')!=POLICY['vllm_version']:raise ValueError('Require vLLM 0.6.6.post1')
            model=Path(original['model_path']);dims=check_architecture(read(model/'config.json'))
            distribution=importlib.metadata.distribution('vllm')
            names=('vllm/engine/arg_utils.py','vllm/config.py','vllm/model_executor/models/llama.py',
                   'vllm/model_executor/guided_decoding/__init__.py','vllm/entrypoints/openai/protocol.py')
            paths=[Path(distribution.locate_file(name)) for name in names]
            arg_source=paths[0].read_text();llama_source=paths[2].read_text();guided_source=paths[3].read_text()
            if not all(x in arg_source for x in ('--tensor-parallel-size','--guided-decoding-backend','--distributed-executor-backend')):
                raise ValueError('Required server CLI not available')
            if 'QKVParallelLinear' not in llama_source or 'outlines' not in guided_source:
                raise ValueError('Llama TP/guided decoding source unavailable')
            if 'guided_json' not in paths[4].read_text():raise ValueError('OpenAI guided_json protocol unavailable')
            doc.update({k:copy.deepcopy(v) for k,v in original.items() if k not in
                ('kind','passed','fingerprint','code_sha256','teacher_settings')})
            doc.update(base_audit_fingerprint=original['fingerprint'],base_audit_sha256=file_hash(BASE_AUDIT),
                base_audit_path=str(BASE_AUDIT),base_audit_reexecuted=False,teacher_settings=SERVER,
                gpu_policy=POLICY,code_sha256=code_hashes(),llama_tp_dimensions=dims,
                runtime_source_sha256={str(p):file_hash(p) for p in paths},
                runtime_version=importlib.metadata.version('vllm'),
                compatibility={'static_cli_and_shape_checks':True,'actual_tp4_nccl_and_guided_request_tested':False},
                scientific_note='Same PAB request/schema and FP16 weights; TP4 reduction order/hardware can change outputs and latency vs TP1.')
            doc['passed']=True
        except Exception as exc:doc['error']={'type':type(exc).__name__,'message':str(exc)[:2000]}
        atomic_save(output,seal(doc))
    if not doc['passed']:raise RuntimeError('TP4 supplemental audit failed')
    print('A2000 TP4 SUPPLEMENTAL AUDIT PASS: original 220305 reused; GPU runtime still requires Smoke',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
