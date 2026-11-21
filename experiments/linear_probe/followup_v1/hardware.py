"""Compute-node-only four-GPU guards, per-session memory monitor and completion gates."""
import argparse
import importlib.metadata
import os
import signal
import subprocess
import time
import traceback

from ..common import read,verify,seal,save_new,file_hash
from ..resources import parse_nodes,validate_allocation
from .plan import OUT,PROFILES,CACHE,approve,require_compute
from experiment_state import atomic_save


def path(kind,role,session):return OUT/f'{kind}_{role}_{os.environ["SLURM_JOB_ID"]}_{session}.json'
def query(args):return subprocess.check_output(['nvidia-smi',*args],text=True,timeout=15)


def preflight(a):
    plan=approve(a.audit);profile=PROFILES[a.role]
    p=path('launch',a.role,a.session)
    doc={'passed':False,'role':a.role,'session':a.session,'profile':profile,'audit_fingerprint':plan['fingerprint'],
         'environment':{k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','SLURM_JOB_GPUS',
           'SLURM_STEP_GPUS','SLURM_JOB_NUM_NODES','SLURMD_NODENAME','SLURM_CPUS_PER_TASK','SLURM_MEM_PER_NODE')},'devices':[]}
    save_new(p,seal(doc))
    try:
        if importlib.metadata.version('vllm')!='0.6.6.post1':raise ValueError('Require original vLLM 0.6.6.post1')
        doc['generation_packages']={k:importlib.metadata.version(k) for k in plan['legacy_solver_config']['generation_packages']}
        if doc['generation_packages']!=plan['legacy_solver_config']['generation_packages']:
            raise ValueError('GPU generation environment differs from the audited original runtime')
        if int(os.environ.get('SLURM_JOB_NUM_NODES',os.environ.get('SLURM_NNODES','0')))!=1:raise ValueError('One node required')
        raw=subprocess.check_output(['scontrol','show','node',os.environ['SLURMD_NODENAME'],'-o'],text=True)
        node=parse_nodes(raw)[0];cpus=int(os.environ['SLURM_CPUS_PER_TASK']);mem=int(os.environ['SLURM_MEM_PER_NODE'])
        if cpus!=profile['cpus'] or mem!=profile['ram_gib']*1024:raise ValueError('Frozen per-role CPU/RAM allocation required')
        doc['allocation']={'node':node,'limits':validate_allocation(node,4,cpus,mem),'cpus':cpus,'memory_mib':mem}
        doc['inventory']=query(['--query-gpu=index,uuid,name,memory.total,memory.free','--format=csv,noheader'])
        doc['processes']=query(['--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'])
        import torch
        doc['device_count']=torch.cuda.device_count()
        if doc['device_count']!=4:raise ValueError('Require four assigned visible GPUs, never override CUDA_VISIBLE_DEVICES')
        torch.cuda.init()
        for i in range(4):
            d={'logical_device':i,'cuda_allocation_passed':False};doc['devices'].append(d)
            torch.cuda.set_device(i);props=torch.cuda.get_device_properties(i);free,_=torch.cuda.mem_get_info(i)
            d.update(name=props.name,uuid=str(getattr(props,'uuid','unavailable')),total_bytes=props.total_memory,free_before_bytes=free)
            if props.name!=profile['gpu_name'] or not profile['min_total_gib']*2**30<=props.total_memory<=profile['max_total_gib']*2**30:
                raise ValueError('Assigned model/VRAM does not match frozen role profile')
            if free<profile['min_free_gib']*2**30 or d['uuid']=='unavailable':raise ValueError('Free VRAM/UUID precheck failed')
            x=torch.empty(1,device=f'cuda:{i}');torch.cuda.synchronize(i);del x
            torch.cuda.empty_cache();torch.cuda.synchronize(i)
            d['cuda_allocation_passed']=True;d['free_after_bytes']=torch.cuda.mem_get_info(i)[0]
            if d['free_after_bytes']<profile['min_free_gib']*2**30:raise ValueError('Lost free VRAM headroom')
        if len({d['uuid'] for d in doc['devices']})!=4:raise ValueError('Duplicate assigned UUIDs')
        doc['passed']=True
    except Exception as exc:doc['error']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
    atomic_save(p,seal(doc))
    if not doc['passed']:raise RuntimeError('GPU launch guard failed')


def monitor(a):
    profile=PROFILES[a.role];launch=verify(read(path('launch',a.role,a.session)))
    if not launch['passed']:raise ValueError('Launch failed')
    uuids=[d['uuid'] if d['uuid'].startswith('GPU-') else 'GPU-'+d['uuid'] for d in launch['devices']]
    p=path('memory',a.role,a.session);stop=False;healthy=True
    doc={'passed':False,'role':a.role,'session':a.session,'samples':0,'uuids':uuids,
         'peak_used_bytes':{u:0 for u in uuids},'limit_bytes':profile['used_limit_gib']*2**30,
         'poll_interval_after_round_seconds':.2,'exact_allocator_peak':False}
    save_new(p,seal(doc));start=time.perf_counter()
    def end(*unused):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,end);signal.signal(signal.SIGINT,end)
    try:
        while not stop:
            for u in uuids:
                used=int(query(['-i',u,'--query-gpu=memory.used','--format=csv,noheader,nounits']).strip())*2**20
                doc['peak_used_bytes'][u]=max(doc['peak_used_bytes'][u],used)
                if used>doc['limit_bytes']:raise RuntimeError('Assigned-device memory limit exceeded: '+u)
            doc['samples']+=1;atomic_save(p,seal(doc));time.sleep(.2)
    except Exception as exc:
        healthy=False;doc['error']={'type':type(exc).__name__,'message':str(exc)}
        try:os.kill(a.server_pid,signal.SIGTERM)
        except ProcessLookupError:pass
    finally:
        doc.update(passed=healthy and doc['samples']>0,wall_seconds=time.perf_counter()-start);atomic_save(p,seal(doc))
    if not doc['passed']:raise RuntimeError('Memory gate failed; stopped only this job server')


def complete_path(role,mode,job):return OUT/f'complete_{role}_{mode}_{job}.json'


def abort(a):
    p=complete_path(a.role,a.mode,os.environ['SLURM_JOB_ID'])
    if not p.exists():
        save_new(p,seal({'passed':False,'role':a.role,'mode':a.mode,'profile':PROFILES[a.role],
            'error':{'message':'Launcher failed; inspect per-session launch/memory/responses JSON and server log',
                     'exit_code':a.exit_code},'audit_path':a.audit}))


def check_complete(role,mode,job,plan):
    doc=verify(read(complete_path(role,mode,job)))
    if not doc.get('passed') or doc['audit_fingerprint']!=plan['fingerprint'] or doc['profile']!=PROFILES[role]:
        raise ValueError('Matching role completion gate required')
    for name,h in doc['protected_outputs'].items():
        if file_hash(name)!=h:raise ValueError('Accepted output changed')
    return doc


def finish(a):
    plan=approve(a.audit);sessions=['smoke'] if a.mode=='smoke' else (list(plan['teacher_schedule']) if a.role=='teacher' else ['main'])
    p=complete_path(a.role,a.mode,os.environ['SLURM_JOB_ID'])
    doc={'passed':False,'role':a.role,'mode':a.mode,'profile':PROFILES[a.role],
         'cache_policy':CACHE,'audit_fingerprint':plan['fingerprint'],'protected_outputs':{}}
    save_new(p,seal(doc))
    try:
        files=[];count=0
        for session in sessions:
            run_path=path('responses',a.role,session);run=verify(read(run_path))
            memory=verify(read(path('memory',a.role,session)))
            if not run['passed'] or not memory['passed']:raise ValueError('Session run/memory failed')
            count+=len(run['records'])
            files += [run_path,path('launch',a.role,session),path('memory',a.role,session),path('server',a.role,session)]
        expected=(1 if a.role=='teacher' else 2) if a.mode=='smoke' else (45 if a.role=='teacher' else 25)
        if count!=expected:raise ValueError('Incomplete request coverage')
        doc.update(passed=True,logical_requests=count,protected_outputs={str(f):file_hash(f) for f in files},
                   observed_case_only=a.mode=='smoke')
        approve(a.audit)
    except Exception as exc:doc['passed']=False;doc['error']={'type':type(exc).__name__,'message':str(exc)}
    atomic_save(p,seal(doc))
    if not doc['passed']:raise RuntimeError('Follow-up stage incomplete')
    print(a.role.upper()+' '+a.mode.upper()+' FINAL PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('preflight','monitor','finish','check-smoke','abort'))
    p.add_argument('--audit',required=True);p.add_argument('--role',choices=PROFILES,required=True)
    p.add_argument('--session');p.add_argument('--mode',choices=('smoke','main'));p.add_argument('--server-pid',type=int);p.add_argument('--smoke-job')
    p.add_argument('--exit-code',type=int)
    a=p.parse_args();require_compute()
    if a.command=='preflight':preflight(a)
    elif a.command=='monitor':monitor(a)
    elif a.command=='finish':finish(a)
    elif a.command=='abort':abort(a)
    else:check_complete(a.role,'smoke',a.smoke_job,approve(a.audit))
