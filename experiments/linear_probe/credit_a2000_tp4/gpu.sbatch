#!/bin/bash
#SBATCH --job-name=a2000-tp4-credit
#SBATCH --partition=batch_ce_ugrad
#SBATCH --account=ugrad_ce
#SBATCH --qos=qos_dada0228_2026_2
#SBATCH --nodelist=moana-u2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:normal:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/dada0228/plan-budget/P-and-B/experiments/linear_probe/runs/credit_a2000_tp4/gpu_%j.log
set -euo pipefail
bash -n "$0"
cd /data/dada0228/plan-budget/P-and-B
: "${AUDIT_JOB:?new TP4 supplemental audit job ID required, not 220305}"
MODE=${MODE:-smoke}
case "$MODE" in smoke|teacher|evaluation) ;; *) echo 'Invalid MODE';exit 2;; esac
RESUME=${RESUME:-0}
case "$RESUME" in 0|1) ;; *) exit 2;; esac
PY=${VLLM_PY:-/data/dada0228/conda_envs/vllm_server/bin/python}
export VLLM_PY="$PY"
export HF_HOME=/data/dada0228/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
# Never override CUDA_VISIBLE_DEVICES or the scheduler's GPU mapping.
RUN=experiments/linear_probe/runs/credit_a2000_tp4
AUDIT="$RUN/audit_${AUDIT_JOB}.json"
if [ "$MODE" != smoke ]; then
    : "${SMOKE_JOB:?successful Teacher GPU Smoke job required}"
    "$PY" -B -m experiments.linear_probe.credit_a2000_tp4.gpu check-smoke --audit "$AUDIT" --smoke-job "$SMOKE_JOB"
fi
if [ "$MODE" = evaluation ]; then
    "$PY" -B -m experiments.linear_probe.credit_a2000_tp4.gpu check-probe --audit "$AUDIT"
fi
"$PY" -B -m experiments.linear_probe.credit_a2000_tp4.gpu preflight --audit "$AUDIT"
export ROLE=teacher
export SERVER_CONFIG="$RUN/server_${MODE}_${SLURM_JOB_ID}.json"
export PORT=${PORT:-$((20000 + SLURM_JOB_ID % 30000))}
API="http://127.0.0.1:$PORT/v1"
"$PY" -B - "$PORT" <<'PY'
import socket,sys
with socket.socket() as s:s.bind(('127.0.0.1',int(sys.argv[1])))
PY
SERVER_PID='';MONITOR_PID='';CLIENT_PID=''
cleanup() {
    trap '' TERM INT
    if [ -n "$CLIENT_PID" ]; then kill -TERM "$CLIENT_PID" 2>/dev/null || true;wait "$CLIENT_PID" || true; fi
    if [ -n "$SERVER_PID" ]; then kill -TERM "$SERVER_PID" 2>/dev/null || true;wait "$SERVER_PID" || true; fi
    if [ -n "$MONITOR_PID" ]; then kill -TERM "$MONITOR_PID" 2>/dev/null || true;wait "$MONITOR_PID" || true; fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
# Dedicated same-node mp/TP4 launcher. No TP1 launcher/configuration is changed.
bash experiments/linear_probe/credit_a2000_tp4/serve.sbatch > "$RUN/server_${MODE}_${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
"$PY" -B -m experiments.linear_probe.credit_a2000_tp4.gpu monitor --server-pid "$SERVER_PID" &
MONITOR_PID=$!
READY=0
for i in $(seq 1 180); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null || ! kill -0 "$MONITOR_PID" 2>/dev/null; then
        echo 'Server/memory monitor exited';exit 3
    fi
    if "$PY" -B - "$API" <<'PY'
import sys
from urllib.request import urlopen
try:
    with urlopen(sys.argv[1]+'/models',timeout=2) as r:r.read()
except Exception:raise SystemExit(1)
PY
    then READY=1;break;fi
    sleep 5
done
if [ "$READY" != 1 ]; then echo 'Server readiness timeout';exit 3;fi
ARGS=()
if [ "$RESUME" = 1 ]; then ARGS+=(--resume);fi
"$PY" -B -m experiments.linear_probe.credit_a2000_tp4.teacher --audit "$AUDIT" \
  --server-config "$SERVER_CONFIG" --server-pid "$SERVER_PID" --api-base "$API" \
  --mode "$MODE" "${ARGS[@]}" &
CLIENT_PID=$!
while kill -0 "$CLIENT_PID" 2>/dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null || ! kill -0 "$MONITOR_PID" 2>/dev/null; then
        echo 'Server/memory monitor died; checkpoint preserved';exit 3
    fi
    sleep 1
done
CODE=0;wait "$CLIENT_PID" || CODE=$?;CLIENT_PID=''
if [ "$CODE" != 0 ]; then exit "$CODE";fi
kill -TERM "$SERVER_PID" 2>/dev/null || true;wait "$SERVER_PID" || true;SERVER_PID=''
kill -TERM "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID";MONITOR_PID=''
"$PY" -B -m experiments.linear_probe.credit_a2000_tp4.gpu finish --audit "$AUDIT" \
  --mode "$MODE" --server-config "$SERVER_CONFIG"
