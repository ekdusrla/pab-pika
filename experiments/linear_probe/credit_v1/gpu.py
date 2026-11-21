"""Separate-process launch/memory gates; nvidia-smi device usage is sampled, not allocator peak."""
import argparse
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path

from ..common import read,verify,seal,save_new,output_path,file_hash
from ..pilot_smoke import allocation
from ..cli import teacher_rows
from .audit import SOURCE,OUT,approve,require_compute
from .teacher import expected_signature,check_server_config,frozen_probe
from experiment_state import atomic_save


def query(args):
    return subprocess.check_output(['nvidia-smi',*args],text=True,timeout=15)


def preflight(a):
    plan=approve(a.audit);path=output_path(OUT/f'launch_{os.environ["SLURM_JOB_ID"]}.json')
    doc={'kind':'teacher_launch_gate','passed':False,'audit_fingerprint':plan['fingerprint'],
         'environment':{k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER',
         'SLURM_JOB_GPUS','SLURM_STEP_GPUS','SLURMD_NODENAME')},'cuda_initialized':False,'tiny_allocation_success':False}
    save_new(path,seal(doc))
    try:
        doc['allocation']=allocation()
        doc['inventory']=query(['--query-gpu=index,uuid,name,memory.total,memory.free','--format=csv,noheader'])
        doc['processes']=query(['--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'])
        import torch
        doc['device_count']=torch.cuda.device_count()
        if doc['device_count']!=1:raise ValueError('Exactly one assigned GPU required')
        torch.cuda.init();torch.cuda.set_device(0);doc['cuda_initialized']=True
        props=torch.cuda.get_device_properties(0)
        doc.update(gpu_name=props.name,gpu_uuid=str(getattr(props,'uuid','unavailable')),total_bytes=props.total_memory)
        if props.name!='NVIDIA GeForce RTX 3090' or not 23*2**30<=props.total_memory<=25*2**30:
            raise ValueError('Require RTX 3090 24 GiB class')
        if doc['gpu_uuid']=='unavailable':raise ValueError('Cannot monitor assigned GPU without UUID')
        free,_=torch.cuda.mem_get_info(0);doc['free_bytes']=free
        if free<21*2**30:raise ValueError('Assigned GPU has less than 21 GiB free')
        test=torch.empty(1,device='cuda:0');torch.cuda.synchronize();doc['tiny_allocation_success']=True;del test
        torch.cuda.empty_cache();torch.cuda.synchronize()
        free,_=torch.cuda.mem_get_info(0);doc['free_after_probe_bytes']=free
        if free<21*2**30:raise ValueError('Assigned GPU lost 21 GiB free headroom')
        doc['passed']=True
    except Exception as exc:doc['error']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
    atomic_save(path,seal(doc))
    if not doc['passed']:raise RuntimeError('Teacher launch gate failed')
    print('TEACHER LAUNCH PASS',flush=True)


def monitor(a):
    launch=verify(read(OUT/f'launch_{os.environ["SLURM_JOB_ID"]}.json'))
    if not launch['passed']:raise ValueError('Launch gate failed')
    uuid=launch['gpu_uuid'];uuid=uuid if uuid.startswith('GPU-') else 'GPU-'+uuid
    path=output_path(OUT/f'memory_{os.environ["SLURM_JOB_ID"]}.json')
    doc={'kind':'sampled_device_memory','passed':False,'gpu_uuid':uuid,'samples':0,
         'peak_device_used_bytes':0,'limit_bytes':21*2**30,'sampling_interval_seconds':.2,
         'exact_torch_allocated_reserved_peak_available':False,
         'note':'Whole assigned GPU usage, including vLLM CUDA graph/KV/context; polling may miss transients.'}
    save_new(path,seal(doc));stopping=False;healthy=True;start=time.perf_counter()
    def stop(*unused):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        while not stopping:
            used=int(query(['-i',uuid,'--query-gpu=memory.used','--format=csv,noheader,nounits']).strip())*2**20
            doc['samples']+=1;doc['peak_device_used_bytes']=max(doc['peak_device_used_bytes'],used)
            if used>doc['limit_bytes']:raise RuntimeError('Assigned GPU device usage exceeded 21 GiB')
            atomic_save(path,seal(doc));time.sleep(.2)
    except Exception as exc:
        healthy=False;doc['error']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
        try:os.kill(a.server_pid,signal.SIGTERM)  # Only the server launched by this job.
        except ProcessLookupError:pass
    finally:
        doc.update(passed=healthy and doc['samples']>0,wall_seconds=time.perf_counter()-start)
        atomic_save(path,seal(doc))
    if not doc['passed']:raise RuntimeError('Memory monitor failed; own server stopped')


def check_complete(path,plan,mode):
    doc=verify(read(path))
    if not doc.get('passed') or doc['mode']!=mode or doc['audit_fingerprint']!=plan['fingerprint']:
        raise ValueError('Successful '+mode+' gate required')
    if doc['teacher_settings']!=expected_signature():raise ValueError('Teacher signature changed')
    for name,h in doc['protected_outputs'].items():
        if file_hash(name)!=h:raise ValueError('Accepted output changed: '+name)
    return doc


def finish(a):
    plan=approve(a.audit);job=os.environ['SLURM_JOB_ID'];path=output_path(OUT/f'complete_{a.mode}_{job}.json')
    doc={'kind':'teacher_gpu_complete','passed':False,'mode':a.mode,'audit_fingerprint':plan['fingerprint'],
         'actual_engine':'vLLM OpenAI API / guided_json / outlines',
         'teacher_settings':expected_signature(),'protected_outputs':{}}
    save_new(path,seal(doc))
    try:
        check_server_config(a.server_config)
        memory_path=OUT/f'memory_{job}.json';memory=verify(read(memory_path))
        if not memory['passed'] or memory['peak_device_used_bytes']>21*2**30:raise ValueError('GPU memory gate failed')
        files=[memory_path,a.server_config,OUT/f'launch_{job}.json']
        if a.mode=='smoke':
            smoke_path=OUT/f'smoke_{job}.json';smoke=verify(read(smoke_path))
            if not smoke['passed'] or smoke['audit_fingerprint']!=plan['fingerprint']:raise ValueError('Smoke response failed')
            files.append(smoke_path);doc['worst_case_2048_tokens_verified']=False
        else:
            data=verify(read(SOURCE/'data.json'))
            for split in (('evaluation',) if a.mode=='evaluation' else ('train','validation')):
                label_path=OUT/(split+'_labeled.json');labels=verify(read(label_path))
                teacher_rows(labels,data,split);files.append(label_path)
        doc['protected_outputs']={str(p):file_hash(p) for p in files};doc['passed']=True
    except Exception as exc:doc['error']={'type':type(exc).__name__,'message':str(exc)}
    atomic_save(path,seal(doc))
    if not doc['passed']:raise RuntimeError('Teacher GPU stage FAIL; inspect '+str(path))
    label='LABELS' if a.mode=='teacher' else a.mode.upper()
    print('TEACHER '+label+' FINAL PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('preflight','monitor','check-smoke','check-probe','finish'))
    p.add_argument('--audit',type=Path);p.add_argument('--server-pid',type=int)
    p.add_argument('--server-config',type=Path);p.add_argument('--mode',choices=('smoke','teacher','evaluation'))
    p.add_argument('--smoke-job');a=p.parse_args();require_compute()
    if a.command=='preflight':preflight(a)
    elif a.command=='monitor':monitor(a)
    elif a.command=='finish':finish(a)
    elif a.command=='check-probe':frozen_probe(OUT/'probe_v1',approve(a.audit))
    else:check_complete(OUT/f'complete_smoke_{a.smoke_job}.json',approve(a.audit),'smoke')
