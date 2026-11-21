#!/bin/bash
#SBATCH --job-name=credit-posthoc-cpu
#SBATCH --account=ugrad_ce
#SBATCH --qos=qos_dada0228_2026_2
#SBATCH --partition=batch_ce_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/data/dada0228/plan-budget/P-and-B/experiments/linear_probe/runs/analysis_credit_tp4_220316/cpu_%j.log
set -euo pipefail
cd /data/dada0228/plan-budget/P-and-B
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PROBE_PY:-/data/dada0228/conda_envs/shortcut-xai/bin/python}
exec "$PY" -B experiments/linear_probe/runs/analysis_credit_tp4_220316/extended.py
