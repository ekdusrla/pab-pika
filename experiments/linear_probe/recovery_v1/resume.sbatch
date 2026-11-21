#!/bin/bash
#SBATCH --job-name=probe-overlong-recovery
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
#SBATCH --output=/data/dada0228/plan-budget/P-and-B/experiments/linear_probe/runs/recovery_%j.log

# User submits after CPU audit. No Teacher, Probe training or Solver.
set -euo pipefail
bash -n "$0"
cd /data/dada0228/plan-budget/P-and-B
: "${AUDIT_JOB:?successful CPU audit job ID required}"
RUN=experiments/linear_probe/runs/pilot_v1
HF_PY=${HF_PY:-/data/dada0228/conda_envs/vllm_server/bin/python}
export HF_HOME=/data/dada0228/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
LLAMA="$HF_HUB_CACHE/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
CAPTURE_RESUME=${CAPTURE_RESUME:-1}
ACK_SAFETY_STOP=${ACK_SAFETY_STOP:-0}
case "$CAPTURE_RESUME:$ACK_SAFETY_STOP" in 0:0|1:0|1:1) ;; *) echo 'Invalid resume/acknowledgement flags';exit 2;; esac
ARGS=()
if [ "$CAPTURE_RESUME" = 1 ] && { [ -e "$RUN/train_recovered.json" ] || [ -e "$RUN/train_recovered.json.tmp" ]; }; then
    ARGS+=(--resume)
    if [ "$ACK_SAFETY_STOP" = 1 ]; then ARGS+=(--resume-after-safety-stop); fi
fi
"$HF_PY" -B -m experiments.linear_probe.recovery_v1.recover \
  --source "$RUN/train_capture.json" --data "$RUN/data.json" --audit "$RUN/audit_${AUDIT_JOB}.json" \
  --output "$RUN/train_recovered.json" --recover-overlong "${ARGS[@]}"
# Successful original Train rows are NEVER passed to collect for regeneration.
for SPLIT in validation evaluation; do
    ARGS=()
    if [ "$SPLIT" = evaluation ]; then ARGS+=(--capture-control); fi
    OUTPUT="$RUN/${SPLIT}_capture.json"
    if [ "$CAPTURE_RESUME" = 1 ] && { [ -e "$OUTPUT" ] || [ -e "${OUTPUT}.tmp" ]; }; then
        ARGS+=(--resume)
        if [ "$ACK_SAFETY_STOP" = 1 ]; then ARGS+=(--resume-after-safety-stop); fi
    fi
    "$HF_PY" -B -m experiments.linear_probe.cli collect --data "$RUN/data.json" \
      --split "$SPLIT" --model-path "$LLAMA" --output "$OUTPUT" "${ARGS[@]}"
done
