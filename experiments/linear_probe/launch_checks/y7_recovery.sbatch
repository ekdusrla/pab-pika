#!/bin/bash
#SBATCH --job-name=probe-v2-y7-guarded
#SBATCH --partition=batch_ce_ugrad
#SBATCH --account=ugrad_ce
#SBATCH --qos=qos_dada0228_2026_2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=moana-y7
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/dada0228/plan-budget/P-and-B/experiments/linear_probe/runs/recovery_v2_guarded_%j.log

# External launch guard: does not change any audited recovery_v2 file or policy.
# User submits. No GPU tests or Python are run on the master node.
set -euo pipefail
bash -n "$0"
cd /data/dada0228/plan-budget/P-and-B
: "${AUDIT_JOB:?successful unchanged V2 CPU audit job ID required}"
: "${SLURMD_NODENAME:?SLURM compute node required}"
HF_PY=${HF_PY:-/data/dada0228/conda_envs/vllm_server/bin/python}
export HF_HOME=/data/dada0228/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
# Preserve SLURM's CUDA_VISIBLE_DEVICES and its logical-to-physical mapping.
"$HF_PY" -B - <<'PY'
import json,os,subprocess,time,traceback
from pathlib import Path
from experiments.linear_probe.recovery_v2.resume import approval
from experiments.linear_probe.pilot_smoke import allocation

job=os.environ['SLURM_JOB_ID'];node=os.environ['SLURMD_NODENAME']
path=Path('experiments/linear_probe/runs/pilot_v1')/f'gpu_launch_{job}.json'
report={'kind':'external_gpu_launch_guard','passed':False,'node':node,'job_id':job,
        'minimum_free_gib':21.,'environment':{k:os.environ.get(k) for k in (
            'CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','SLURM_JOB_GPUS','SLURM_STEP_GPUS',
            'SLURM_CPUS_PER_TASK','SLURM_MEM_PER_NODE')},
        'note':'Launch-only memory headroom gate. Not a generation or 1024-token worst-case test.'}
started=time.perf_counter()
with path.open('x',encoding='utf-8') as stream:
    try:
        report['phase']='frozen_source_audit'
        checked=approval(Path('experiments/linear_probe/runs/pilot_v1')/f'audit_v2_{os.environ["AUDIT_JOB"]}.json')
        report['audit_fingerprint']=checked['fingerprint'];report['allocation']=allocation()
        if node!='moana-y7':raise RuntimeError('This recovery launcher requires the validated y7 node')
        report['phase']='nvidia_smi_inventory'
        for key,fields,kind in (
            ('gpu_inventory','index,uuid,name,memory.total,memory.free','--query-gpu='),
            ('compute_processes','gpu_uuid,pid,used_gpu_memory','--query-compute-apps=')):
            result=subprocess.run(['nvidia-smi',kind+fields,'--format=csv,noheader'],
                                  text=True,capture_output=True,timeout=15,check=False)
            report[key]={'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
            if result.returncode:raise RuntimeError('GPU inventory query failed: '+key)
        report['phase']='cuda_initialization'
        import torch
        report['torch']=torch.__version__;report['torch_cuda']=torch.version.cuda
        report['device_count']=torch.cuda.device_count()
        if report['device_count']!=1:raise RuntimeError('Exactly one SLURM-visible GPU required')
        torch.cuda.init();torch.cuda.set_device(0)
        props=torch.cuda.get_device_properties(0)
        report['logical_device']=0;report['gpu_name']=props.name
        report['gpu_uuid']=str(getattr(props,'uuid','unavailable'))
        report['device_total_bytes']=props.total_memory
        if '3090' not in props.name:raise RuntimeError('Require RTX 3090 before weights')
        free,total=torch.cuda.mem_get_info(0)
        report.update(free_before_probe_bytes=free,total_bytes=total)
        if free<21*2**30:raise RuntimeError('Less than 21 GiB free on assigned GPU; refuse model loading')
        report['phase']='tiny_cuda_allocation'
        probe=torch.empty(1,device='cuda:0');torch.cuda.synchronize()
        del probe
        torch.cuda.empty_cache();torch.cuda.synchronize()
        free,total=torch.cuda.mem_get_info(0);report['free_after_probe_bytes']=free
        if free<21*2**30:raise RuntimeError('Assigned GPU lost required free headroom')
        report['passed']=True;report['phase']='complete'
    except Exception as exc:
        report['error']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
    finally:
        report['wall_seconds']=time.perf_counter()-started
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
if not report['passed']:raise RuntimeError('GPU launch guard failed; inspect '+str(path))
print('GPU LAUNCH GUARD PASS: RTX 3090, one visible device, >=21 GiB free; '+str(path),flush=True)
PY
# The guard process exits and releases its tiny CUDA context before model loading.
# Current model-load OOM requires explicit ACK_SAFETY_STOP=1, supplied by the user.
exec bash experiments/linear_probe/recovery_v2/resume.sbatch
