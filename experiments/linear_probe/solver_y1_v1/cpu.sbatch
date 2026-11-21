#!/bin/bash
#SBATCH --job-name=qwq-y1-cpu
#SBATCH --partition=batch_ce_ugrad
#SBATCH --account=ugrad_ce
#SBATCH --qos=qos_dada0228_2026_2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/data/dada0228/plan-budget/P-and-B/experiments/linear_probe/runs/solver_y1_v1/cpu_%j.log
set -euo pipefail
cd /data/dada0228/plan-budget/P-and-B
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/data/dada0228/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${VLLM_PY:-/data/dada0228/conda_envs/vllm_server/bin/python}
TASK=${TASK:-audit}
if [ "$TASK" = audit ];then
    "$PY" -B - <<'PY'
import ast,re,subprocess
from pathlib import Path
d=Path('experiments/linear_probe/solver_y1_v1')
for p in d.glob('*.py'):compile(ast.parse(p.read_text()),str(p),'exec')
for pattern in ('*.sh','*.sbatch'):
    for p in d.glob(pattern):
        subprocess.run(['bash','-n',str(p)],check=True)
        for block in re.finditer(r"<<'PY'\n(.*?)\nPY(?:\n|$)",p.read_text(),re.S):
            compile(ast.parse(block.group(1)),str(p)+':heredoc','exec')
PY
    "$PY" -B -m unittest -v experiments.linear_probe.solver_y1_v1.tests_mapping experiments.linear_probe.solver_y1_v1.tests_identity
    exec "$PY" -B -m experiments.linear_probe.solver_y1_v1.plan \
       --output "experiments/linear_probe/runs/solver_y1_v1/audit_${SLURM_JOB_ID}.json"
elif [ "$TASK" = metadata ];then
    exec "$PY" -B -m experiments.linear_probe.solver_y1_v1.metadata
elif [ "$TASK" = analysis ];then
    : "${AUDIT_JOB:?y1 CPU audit required}"
    : "${MAIN_JOB:?y1 Pilot GPU job required}"
    exec "$PY" -B -m experiments.linear_probe.solver_y1_v1.analysis \
       --audit "experiments/linear_probe/runs/solver_y1_v1/audit_${AUDIT_JOB}.json" --main-job "$MAIN_JOB"
else
    echo 'Invalid TASK';exit 2
fi
