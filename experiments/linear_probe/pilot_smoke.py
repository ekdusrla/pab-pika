"""One real longest Pilot input, same generation/guard as collection. GPU-explicit CLI."""
import argparse
import os
import subprocess
import time
from pathlib import Path

from .common import (MODEL,REVISION,read,verify,seal,save_new,output_path,runtime,
                     decomp_messages,file_hash,digest,prompt_fingerprint)
from .pilot_data import validate_pilot
from .safety import MemoryGuard,require_device,guarded_generate,failure_kind
from .resources import parse_nodes,validate_allocation
from experiment_state import lock,atomic_save,StopRequested,termination_handler


def longest_input(data,tok):
    candidates=[]
    for split in ('train','validation','evaluation'):
        for row in data['splits'][split]:
            messages=decomp_messages(row)
            count=len(tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))
            candidates.append({'split':split,'row':row,'messages':messages,'prompt_tokens':count})
    if not candidates:raise ValueError('No Pilot inputs')
    return min(candidates,key=lambda x:(-x['prompt_tokens'],str(x['row']['id']),x['split']))


def allocation():
    node=os.environ.get('SLURMD_NODENAME')
    if not node:raise RuntimeError('Smoke must run inside a SLURM compute allocation')
    raw=subprocess.check_output(['scontrol','show','node',node,'-o'],text=True)
    nodes=parse_nodes(raw)
    if len(nodes)!=1:raise ValueError('Expected one allocated node')
    cpus=int(os.environ['SLURM_CPUS_PER_TASK']);mem=int(os.environ['SLURM_MEM_PER_NODE'])
    if cpus!=12 or mem!=40*1024:raise ValueError('Smoke allocation must be 12 CPU and 40 GiB RAM')
    caps=validate_allocation(nodes[0],1,cpus,mem)
    return {'node':nodes[0],'limits':caps,'requested_cpus':cpus,'requested_memory_mib':mem}


def run(a):
    path=output_path(a.output)
    with lock(path):
        for existing in (path,Path(str(path)+'.tmp'),Path(str(path)+'.features.npz')):
            if existing.exists():raise FileExistsError(existing)
        doc={'kind':'pilot_longest_input_gpu_smoke','status':'running','passed':False,
             'safe_observed_case':False,'phase':'preflight','model':MODEL,'revision':REVISION,
             'settings':{'dtype':'float16','attention':'eager','layers':list(range(33)),
                'max_new_tokens':1024,'max_context':8192,'temperature':.1,'top_p':.9,'seed':42},
             'reserved_peak_limit_gib':21.,'runtime':runtime(),
             'worst_case_1024_tokens_verified':False,'parse_success':False,'features_valid':False,
             'completion_tokens':0,'finish_reason':None,'n_steps':None,'extra_forward_passes':0,
             'peak_allocated_bytes':None,'peak_reserved_bytes':None,'timings':{}}
        save_new(path,seal(doc))  # Reserve a NEW output before validation/loading.
        started=time.perf_counter();guard=None;torch=None;load_started=None
        try:
            snapshot=a.model_path.resolve()
            if snapshot.name!=REVISION:raise ValueError('Pinned Llama snapshot required')
            data=validate_pilot(read(a.data));context=verify(read(a.context_plan))
            if context['data_fingerprint']!=data['fingerprint']:raise ValueError('Context plan/data mismatch')
            if context['tokenizer_sha256']!=file_hash(snapshot/'tokenizer.json'):
                raise ValueError('Tokenizer differs from CPU context plan')
            doc.update(data_fingerprint=data['fingerprint'],context_plan_fingerprint=context['fingerprint'],
                       prompt_fingerprint=prompt_fingerprint(),allocation=allocation())
            import torch
            from transformers import AutoTokenizer,AutoModelForCausalLM
            doc['gpu']=require_device(torch)
            doc['device_total_bytes']=torch.cuda.get_device_properties(0).total_memory
            guard=MemoryGuard(torch.cuda,21.)
            begin=time.perf_counter()
            tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
            item=longest_input(data,tok)
            doc['tokenization_and_selection_seconds']=time.perf_counter()-begin
            if item['prompt_tokens']!=context['max_decomposition_prompt_tokens']:
                raise ValueError('Longest input differs from CPU context plan')
            doc.update(selected_problem={'id':item['row']['id'],'problem_hash':item['row']['problem_hash'],
                'split':item['split'],'prompt_tokens':item['prompt_tokens']},messages=item['messages'],
                prompt_tokens=item['prompt_tokens'])
            if item['prompt_tokens']+1024>8192:raise ValueError('Model context exceeded')
            guard.begin();doc['phase']='model_load'
            atomic_save(path,seal(doc))
            begin=time.perf_counter();load_started=begin
            model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,
                torch_dtype=torch.float16,attn_implementation='eager').to('cuda').eval()
            torch.cuda.synchronize()
            doc['model_load_seconds']=time.perf_counter()-begin
            doc['model_load_memory']=guard.snapshot()
            guard.check('model_load')
            if any(p.dtype!=torch.float16 for p in model.parameters() if p.is_floating_point()):
                raise ValueError('Model parameters are not all FP16')
            doc.update(actual_model_config=model.config.to_dict(),generation_config=model.generation_config.to_dict())
            doc['phase']='generation'
            atomic_save(path,seal(doc))
            outcome=guarded_generate(model,tok,item['messages'],guard,list(range(33)),max_new_tokens=1024)
            doc['outcome']={k:v for k,v in outcome.items() if k!='result'}
            doc.update(status=outcome['status'],passed=outcome['safe'],safe_observed_case=outcome['safe'],
                       parse_success=outcome['parse_success'],features_valid=outcome['features_valid'],
                       completion_tokens=outcome['completion_tokens'],finish_reason=outcome['finish_reason'],
                       peak_allocated_bytes=outcome['memory']['peak_allocated_bytes'],
                       peak_reserved_bytes=outcome['memory']['peak_reserved_bytes'],error=outcome['error'])
            trace=outcome['diagnostics']
            doc['n_steps']=trace.get('n_steps')
            doc['timings']={'model_load_seconds':doc['model_load_seconds'],
                'generation_seconds':trace.get('generation_seconds'),
                'hidden_retention_seconds':trace.get('hidden_retention_seconds'),
                'alignment_seconds':trace.get('alignment_seconds'),
                'feature_transfer_seconds':trace.get('feature_transfer_seconds'),
                'feature_validation_seconds':outcome['feature_validation_seconds'],
                'generation_and_feature_wall_seconds':outcome['generation_and_feature_wall_seconds'],
                'partial_on_failure':outcome['status']!='success'}
            if outcome['status']=='success':
                import numpy as np
                doc['phase']='feature_storage'
                cap=outcome['result'];features=cap.pop('features')
                target=output_path(Path(str(path)+'.features.npz'))
                storage_started=time.perf_counter()
                with target.open('xb') as stream:
                    np.savez_compressed(stream,**{str(layer):np.asarray([f[str(layer)] for f in features],dtype=np.float32)
                                                 for layer in range(33)})
                cap['features']={'path':str(target),'sha256':file_hash(target),'n_steps':len(features),
                                 'shape':[len(features),33,4096]}
                storage_seconds=time.perf_counter()-storage_started
                doc.update(capture=cap,n_steps=len(cap['steps']),timings={
                    'model_load_seconds':doc['model_load_seconds'],
                    'generation_seconds':cap['metrics']['generation_seconds'],
                    'hidden_retention_seconds':cap['metrics']['hidden_retention_seconds'],
                    'alignment_seconds':cap['metrics']['alignment_seconds'],
                    'feature_transfer_seconds':cap['metrics']['feature_transfer_seconds'],
                    'feature_validation_seconds':outcome['feature_validation_seconds'],
                    'feature_storage_seconds':storage_seconds,
                    'generation_plus_feature_seconds':cap['metrics']['generation_plus_feature_seconds']})
                doc['phase']='complete'
        except (Exception,StopRequested) as exc:
            doc.update(status=failure_kind(exc),passed=False,safe_observed_case=False,
                       error={'type':type(exc).__name__,'message':str(exc)[:1500],'phase':doc['phase']})
            if guard is not None:
                memory=guard.snapshot()
                doc.update(failure_memory=memory,peak_allocated_bytes=memory.get('peak_allocated_bytes'),
                           peak_reserved_bytes=memory.get('peak_reserved_bytes'))
        finally:
            if load_started is not None and 'model_load_seconds' not in doc:
                doc['model_load_elapsed_seconds_on_failure']=time.perf_counter()-load_started
                doc['timings']['model_load_elapsed_seconds_on_failure']=doc['model_load_elapsed_seconds_on_failure']
            doc['wall_seconds_including_load_and_storage']=time.perf_counter()-started
            doc['coverage_note']='PASS covers this observed EOS-terminated trajectory only; short output does not validate 1024-token peak memory.'
            atomic_save(path,seal(doc))
        if not doc['passed']:raise RuntimeError(f"{doc['status']}; diagnostics saved to {path}")
        print('Observed-case PASS:',path,flush=True)
        print('Actual completion tokens:',doc['completion_tokens'],'; 1024-token worst case NOT verified',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--context-plan',type=Path,required=True)
    p.add_argument('--model-path',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    with termination_handler():run(a)


if __name__=='__main__':main()
