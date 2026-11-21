#!/bin/bash
#SBATCH --job-name=teacher-stability-analysis
#SBATCH --partition=batch_ce_ugrad
#SBATCH --account=ugrad_ce
#SBATCH --qos=qos_dada0228_2026_2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/data/dada0228/plan-budget/P-and-B/experiments/linear_probe/runs/stability_analysis_v1/cpu_%j.log
set -euo pipefail
cd /data/dada0228/plan-budget/P-and-B
: "${AUDIT_JOB:?accepted followup audit required}"
: "${MAIN_JOB:?completed Teacher main required}"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PROBE_PY:-/data/dada0228/conda_envs/shortcut-xai/bin/python}
"$PY" -B - <<'PY'
import ast
from pathlib import Path
for path in Path('experiments/linear_probe/stability_analysis_v1').glob('*.py'):
    compile(ast.parse(path.read_text()),str(path),'exec')
PY
exec "$PY" -B -m experiments.linear_probe.stability_analysis_v1.analyze \
  --audit "experiments/linear_probe/runs/followup_v1/audit_${AUDIT_JOB}.json" --main-job "$MAIN_JOB"
