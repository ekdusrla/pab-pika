"""Unchanged model/sampling/cache limits, explicit single-node mp executor."""
import argparse
import importlib.metadata
import os
import sys
import time
from ..common import save_new,seal,runtime
from .plan import PROFILES,CACHE,approve,require_compute
from .hardware import path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',choices=PROFILES,required=True)
    p.add_argument('--session',required=True);p.add_argument('--audit',required=True);p.add_argument('--port',type=int,required=True)
    a=p.parse_args();require_compute();plan=approve(a.audit);s=PROFILES[a.role]
    if importlib.metadata.version('vllm')!='0.6.6.post1':raise RuntimeError('Require original vLLM')
    from vllm.engine.arg_utils import EngineArgs
    if EngineArgs(model=s['model']).enable_prefix_caching is not False:raise ValueError('Prefix cache runtime default differs from audit')
    args=[sys.executable,'-m','vllm.entrypoints.openai.api_server','--model',s['model'],'--revision',s['revision'],
          '--tokenizer-revision',s['revision'],'--tensor-parallel-size','4','--distributed-executor-backend','mp',
          '--dtype','float16','--max-model-len',str(s['context']),'--max-num-seqs','1','--gpu-memory-utilization','.85',
          '--seed','42','--host','127.0.0.1','--port',str(a.port)]
    if a.role=='teacher':args+=['--guided-decoding-backend','outlines']
    save_new(path('server',a.role,a.session),seal({'profile':s,'cache_policy':CACHE,'argv':args,
        'audit_fingerprint':plan['fingerprint'],'runtime':{**runtime(),'engine':'vLLM 0.6.6.post1 / mp / TP4'},
        'started_unix':time.time(),'slurm_job_id':os.environ['SLURM_JOB_ID'],
        'server_restart':a.session.startswith('restart_'),'disk_caches_cleared':False}))
    os.execv(sys.executable,args)
