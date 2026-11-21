#!/bin/bash
#SBATCH --job-name=credit-cpu-audit
#SBATCH --partition=batch_ce_ugrad
#SBATCH --account=ugrad_ce
#SBATCH --qos=qos_dada0228_2026_2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/data/dada0228/plan-budget/P-and-B/experiments/linear_probe/runs/credit_v1/audit_%j.log
set -euo pipefail
cd /data/dada0228/plan-budget/P-and-B
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/data/dada0228/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
PROBE_PY=${PROBE_PY:-/data/dada0228/conda_envs/shortcut-xai/bin/python}
HF_PY=${HF_PY:-/data/dada0228/conda_envs/vllm_server/bin/python}
"$PROBE_PY" -B - <<'PY'
import ast,subprocess
from pathlib import Path
for name in ('','recovery_v1','recovery_v2','credit_v1'):
    for path in (Path('experiments/linear_probe')/name).glob('*.py'):
        compile(ast.parse(path.read_text()),str(path),'exec')
for name in ('audit.sbatch','gpu.sbatch','train.sbatch'):
    subprocess.run(['bash','-n','experiments/linear_probe/credit_v1/'+name],check=True)
PY
"$PROBE_PY" -B -m unittest -v experiments.linear_probe.test_probe \
  experiments.linear_probe.test_pilot experiments.linear_probe.test_safety \
  experiments.linear_probe.recovery_v1.tests experiments.linear_probe.recovery_v2.tests \
  experiments.linear_probe.credit_v1.tests
exec "$HF_PY" -B -m experiments.linear_probe.credit_v1.audit \
  --output "experiments/linear_probe/runs/credit_v1/audit_${SLURM_JOB_ID}.json"
