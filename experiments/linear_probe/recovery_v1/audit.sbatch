#!/bin/bash
#SBATCH --job-name=probe-capture-audit
#SBATCH --partition=batch_ce_ugrad
#SBATCH --account=ugrad_ce
#SBATCH --qos=qos_dada0228_2026_2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/data/dada0228/plan-budget/P-and-B/experiments/linear_probe/runs/audit_%j.log

# CPU only. User submits; decodes recorded token IDs, never loads model weights.
set -euo pipefail
cd /data/dada0228/plan-budget/P-and-B
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/data/dada0228/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
PROBE_PY=${PROBE_PY:-/data/dada0228/conda_envs/shortcut-xai/bin/python}
HF_PY=${HF_PY:-/data/dada0228/conda_envs/vllm_server/bin/python}
"$PROBE_PY" -B - <<'PY'
import ast,subprocess
from pathlib import Path
for directory in (Path('experiments/linear_probe'),Path('experiments/linear_probe/recovery_v1')):
    for path in directory.glob('*.py'):compile(ast.parse(path.read_text()),str(path),'exec')
for name in ('audit.sbatch','resume.sbatch'):
    subprocess.run(['bash','-n','experiments/linear_probe/recovery_v1/'+name],check=True)
PY
"$PROBE_PY" -B -m unittest -v experiments.linear_probe.test_probe \
  experiments.linear_probe.test_pilot experiments.linear_probe.test_safety \
  experiments.linear_probe.recovery_v1.tests
exec "$HF_PY" -B -m experiments.linear_probe.recovery_v1.audit \
  --source experiments/linear_probe/runs/pilot_v1/train_capture.json \
  --data experiments/linear_probe/runs/pilot_v1/data.json \
  --output "experiments/linear_probe/runs/pilot_v1/audit_${SLURM_JOB_ID}.json"
