#!/bin/bash
#SBATCH --job-name=credit-probe-cpu
#SBATCH --partition=batch_ce_ugrad
#SBATCH --account=ugrad_ce
#SBATCH --qos=qos_dada0228_2026_2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/data/dada0228/plan-budget/P-and-B/experiments/linear_probe/runs/credit_a2000_tp4/probe_%j.log
set -euo pipefail
bash -n "$0"
cd /data/dada0228/plan-budget/P-and-B
: "${AUDIT_JOB:?new credit CPU audit job ID required}"
MODE=${MODE:-train}
ARGS=()
case "$MODE" in
    train) : "${TEACHER_JOB:?successful Teacher job required}";ARGS+=(--teacher-job "$TEACHER_JOB");;
    evaluate) : "${EVALUATION_JOB:?successful held-out Teacher job required}";ARGS+=(--evaluation-job "$EVALUATION_JOB");;
    *) exit 2;;
esac
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PROBE_PY:-/data/dada0228/conda_envs/shortcut-xai/bin/python}
exec "$PY" -B -m experiments.linear_probe.credit_a2000_tp4.train \
  --audit "experiments/linear_probe/runs/credit_a2000_tp4/audit_${AUDIT_JOB}.json" --mode "$MODE" "${ARGS[@]}"
