#!/bin/bash
#SBATCH --job-name=probe-v2-recovery
#SBATCH --partition=batch_ce_ugrad
#SBATCH --account=ugrad_ce
#SBATCH --qos=qos_dada0228_2026_2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/dada0228/plan-budget/P-and-B/experiments/linear_probe/runs/recovery_v2_%j.log
set -euo pipefail
bash -n "$0"
cd /data/dada0228/plan-budget/P-and-B
: "${AUDIT_JOB:?successful V2 CPU audit job ID required}"
HF_PY=${HF_PY:-/data/dada0228/conda_envs/vllm_server/bin/python}
export HF_HOME=/data/dada0228/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
ACK_SAFETY_STOP=${ACK_SAFETY_STOP:-0}
CAPTURE_RESUME=${CAPTURE_RESUME:-1}
case "$CAPTURE_RESUME:$ACK_SAFETY_STOP" in 0:0|1:0|1:1) ;; *) echo 'Invalid resume/acknowledgement flags';exit 2;; esac
ARGS=()
if [ "$CAPTURE_RESUME" = 1 ]; then ARGS+=(--resume); fi
if [ "$ACK_SAFETY_STOP" = 1 ]; then ARGS+=(--ack-safety-stop); fi
# Each stage has its own process, releasing the preceding model before the next load.
for PHASE in preflight validation evaluation final; do
    "$HF_PY" -B -m experiments.linear_probe.recovery_v2.resume \
      --audit "experiments/linear_probe/runs/pilot_v1/audit_v2_${AUDIT_JOB}.json" \
      --phase "$PHASE" "${ARGS[@]}"
done
