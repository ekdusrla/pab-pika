"""Separate-process launch/memory gates; nvidia-smi device usage is sampled, not allocator peak."""
import argparse
import importlib.metadata
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path

from ..common import read,verify,seal,save_new,output_path,file_hash
from ..resources import parse_nodes,validate_allocation
from ..cli import teacher_rows
from .audit import SOURCE,OUT,approve,require_compute,POLICY
from .teacher import expected_signature,check_server_config,frozen_probe
from experiment_state import atomic_save


def query(args):
    return subprocess.check_output(['nvidia-smi',*args],text=True,timeout=15)


def allocation():
    if int(os.environ.get('SLURM_JOB_NUM_NODES',os.environ.get('SLURM_NNODES','0')))!=1:
        raise ValueError('All four GPUs must be on one node')
    raw=subprocess.check_output(['scontrol','show','node',os.environ['SLURMD_NODENAME'],'-o'],text=True)
    nodes=parse_nodes(raw)
    if len(nodes)!=1:raise ValueError('One node expected')
    cpus=int(os.environ['SLURM_CPUS_PER_TASK']);mem=int(os.environ['SLURM_MEM_PER_NODE'])
    if cpus!=4*POLICY['cpus_per_gpu'] or mem!=4*POLICY['ram_gib_per_gpu']*1024:
        raise ValueError('Require 24 CPUs/48 GiB for this four-GPU policy')
    caps=validate_allocation(nodes[0],4,cpus,mem)
    return {'node':nodes[0],'limits':caps,'requested_gpus':4,'requested_cpus':cpus,'requested_memory_mib':mem}


def check_device(name,total_bytes,free_bytes):
    if name!=POLICY['gpu_name'] or not POLICY['minimum_total_gib']*2**30<=total_bytes<=POLICY['maximum_total_gib']*2**30:
        raise ValueError('Require four RTX A2000 12 GiB-class cards; 6 GiB is not this audited profile')
    if free_bytes<POLICY['minimum_free_gib_per_gpu']*2**30:
        raise ValueError('Assigned A2000 has less than 10 GiB free; no model loading')


def preflight(a):
    plan=approve(a.audit);path=output_path(OUT/f'launch_{os.environ["SLURM_JOB_ID"]}.json')
    doc={'kind':'a2000_tp4_launch_gate','passed':False,'audit_fingerprint':plan['fingerprint'],'gpu_policy':POLICY,
         'environment':{k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER',
         'SLURM_JOB_GPUS','SLURM_STEP_GPUS','SLURMD_NODENAME')},'devices':[]}
    save_new(path,seal(doc))
    try:
        doc['allocation']=allocation()
        if importlib.metadata.version('vllm')!=POLICY['vllm_version']:raise ValueError('Require vLLM 0.6.6.post1')
        doc['inventory']=query(['--query-gpu=index,uuid,name,memory.total,memory.free','--format=csv,noheader'])
        doc['processes']=query(['--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'])
        import torch
        doc['device_count']=torch.cuda.device_count()
        if doc['device_count']!=4:raise ValueError('Exactly four SLURM-visible GPUs required')
        torch.cuda.init()
        for i in range(4):
            device={'logical_device':i,'cuda_initialized':False,'tiny_allocation_success':False};doc['devices'].append(device)
            # Logical IDs only, always within the unchanged SLURM allocation.
            torch.cuda.set_device(i);props=torch.cuda.get_device_properties(i)
            free,_=torch.cuda.mem_get_info(i)
            device.update(cuda_initialized=True,gpu_name=props.name,gpu_uuid=str(getattr(props,'uuid','unavailable')),
                total_bytes=props.total_memory,free_bytes=free,compute_capability=[props.major,props.minor])
            check_device(props.name,props.total_memory,free)
            if device['gpu_uuid']=='unavailable':raise ValueError('Assigned GPU UUID required for per-GPU monitoring')
            test=torch.empty(1,device=f'cuda:{i}');torch.cuda.synchronize(i);device['tiny_allocation_success']=True;del test
            torch.cuda.empty_cache();torch.cuda.synchronize(i)
            free,_=torch.cuda.mem_get_info(i);device['free_after_probe_bytes']=free
            check_device(props.name,props.total_memory,free)
        if len({d['gpu_uuid'] for d in doc['devices']})!=4:raise ValueError('Duplicate GPU UUID mapping')
        doc['peer_access']=[[True if i==j else torch.cuda.can_device_access_peer(i,j) for j in range(4)] for i in range(4)]
        doc['nccl_tp4_verified']=False  # The following real vLLM/guided request Smoke must verify it.
        doc['passed']=True
    except Exception as exc:doc['error']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
    atomic_save(path,seal(doc))
    if not doc['passed']:raise RuntimeError('Teacher launch gate failed')
    print('A2000 TP4 LAUNCH PASS: four unique GPUs, CUDA allocation and per-GPU free memory verified',flush=True)


def monitor(a):
    launch=verify(read(OUT/f'launch_{os.environ["SLURM_JOB_ID"]}.json'))
    if not launch['passed']:raise ValueError('Launch gate failed')
    uuids=[d['gpu_uuid'] if d['gpu_uuid'].startswith('GPU-') else 'GPU-'+d['gpu_uuid'] for d in launch['devices']]
    path=output_path(OUT/f'memory_{os.environ["SLURM_JOB_ID"]}.json')
    doc={'kind':'sampled_four_gpu_memory','passed':False,'gpu_uuids':uuids,'samples':0,
         'peak_per_gpu_used_bytes':{uuid:0 for uuid in uuids},'peak_device_used_bytes':0,
         'limit_bytes':POLICY['maximum_observed_used_gib_per_gpu']*2**30,'sampling_interval_seconds':.2,
         'exact_torch_allocated_reserved_peak_available':False,
         'note':'Whole assigned GPU usage, including vLLM CUDA graph/KV/context; polling may miss transients.'}
    save_new(path,seal(doc));stopping=False;healthy=True;start=time.perf_counter()
    def stop(*unused):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        while not stopping:
            for uuid in uuids:
                used=int(query(['-i',uuid,'--query-gpu=memory.used','--format=csv,noheader,nounits']).strip())*2**20
                doc['peak_per_gpu_used_bytes'][uuid]=max(doc['peak_per_gpu_used_bytes'][uuid],used)
                doc['peak_device_used_bytes']=max(doc['peak_device_used_bytes'],used)
                if used>doc['limit_bytes']:raise RuntimeError('An assigned A2000 exceeded 11 GiB device usage: '+uuid)
            doc['samples']+=1
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
        if not memory['passed'] or len(memory['gpu_uuids'])!=4 or memory['peak_device_used_bytes']>POLICY['maximum_observed_used_gib_per_gpu']*2**30:
            raise ValueError('Per-GPU A2000 memory gate failed')
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
