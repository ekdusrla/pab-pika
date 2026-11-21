#!/bin/bash
#SBATCH --job-name=followup-analysis
#SBATCH --partition=batch_ce_ugrad
#SBATCH --account=ugrad_ce
#SBATCH --qos=qos_dada0228_2026_2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/data/dada0228/plan-budget/P-and-B/experiments/linear_probe/runs/followup_v1/analysis_%j.log
set -euo pipefail
cd /data/dada0228/plan-budget/P-and-B
: "${AUDIT_JOB:?followup audit required}"
: "${MAIN_JOB:?completed main GPU job required}"
: "${ROLE:?teacher or solver}"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PROBE_PY:-/data/dada0228/conda_envs/shortcut-xai/bin/python}
exec "$PY" -B -m experiments.linear_probe.followup_v1.analysis --role "$ROLE" \
  --audit "experiments/linear_probe/runs/followup_v1/audit_${AUDIT_JOB}.json" --main-job "$MAIN_JOB"
