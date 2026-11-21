#!/bin/bash
#SBATCH --job-name=probe-safety-tests
#SBATCH --partition=batch_ce_ugrad
#SBATCH --account=ugrad_ce
#SBATCH --qos=qos_dada0228_2026_2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/data/dada0228/plan-budget/P-and-B/experiments/linear_probe/runs/safety_tests_%j.log

# CPU allocation only. User submits; no pretrained models or actual research fit.
set -euo pipefail
cd /data/dada0228/plan-budget/P-and-B
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PROBE_PY=${PROBE_PY:-/data/dada0228/conda_envs/shortcut-xai/bin/python}
"$PROBE_PY" -B - <<'PY'
import ast
import subprocess
from pathlib import Path
root=Path('experiments/linear_probe')
for path in root.glob('*.py'):
    compile(ast.parse(path.read_text()),str(path),'exec')
for name in ('pilot_smoke.sbatch','safety_cpu_tests.sbatch','pilot_gpu.sbatch'):
    subprocess.run(['bash','-n',str(root/name)],check=True)
print('Python and shell static checks passed',flush=True)
PY
exec "$PROBE_PY" -B -m unittest -v \
  experiments.linear_probe.test_probe \
  experiments.linear_probe.test_pilot \
  experiments.linear_probe.test_safety
