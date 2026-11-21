#!/bin/bash
#SBATCH --job-name=a2000-tp4-audit
#SBATCH --partition=batch_ce_ugrad
#SBATCH --account=ugrad_ce
#SBATCH --qos=qos_dada0228_2026_2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/data/dada0228/plan-budget/P-and-B/experiments/linear_probe/runs/credit_a2000_tp4/audit_%j.log
set -euo pipefail
cd /data/dada0228/plan-budget/P-and-B
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/data/dada0228/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PROBE_PY=${PROBE_PY:-/data/dada0228/conda_envs/shortcut-xai/bin/python}
HF_PY=${HF_PY:-/data/dada0228/conda_envs/vllm_server/bin/python}
# Only new TP4 compilation/tests. The accepted 58-test base audit is NOT rerun.
"$PROBE_PY" -B - <<'PY'
import ast,re,subprocess
from pathlib import Path
directory=Path('experiments/linear_probe/credit_a2000_tp4')
for path in directory.glob('*.py'):compile(ast.parse(path.read_text()),str(path),'exec')
for path in directory.glob('*.sbatch'):
    subprocess.run(['bash','-n',str(path)],check=True)
    for i,match in enumerate(re.finditer(r"<<'PY'\n(.*?)\nPY(?:\n|$)",path.read_text(),re.S)):
        compile(ast.parse(match.group(1)),str(path)+':heredoc'+str(i),'exec')
PY
"$PROBE_PY" -B -m unittest -v experiments.linear_probe.credit_a2000_tp4.tests
exec "$HF_PY" -B -m experiments.linear_probe.credit_a2000_tp4.audit \
  --output "experiments/linear_probe/runs/credit_a2000_tp4/audit_${SLURM_JOB_ID}.json"
