#!/bin/bash
# Only inside the corresponding four-GPU sbatch allocation.
set -euo pipefail
cd /data/dada0228/plan-budget/P-and-B
: "${SLURM_JOB_ID:?compute allocation required}"
: "${SLURMD_NODENAME:?compute node required}"
: "${ROLE:?teacher or solver}"
: "${AUDIT_JOB:?followup CPU audit job required}"
MODE=${MODE:-smoke}
case "$ROLE:$MODE" in teacher:smoke|teacher:main|solver:smoke|solver:main) ;; *) exit 2;; esac
PY=${VLLM_PY:-/data/dada0228/conda_envs/vllm_server/bin/python}
RUN=experiments/linear_probe/runs/followup_v1
AUDIT="$RUN/audit_${AUDIT_JOB}.json"
export HF_HOME=/data/dada0228/huggingface HF_HUB_CACHE=/data/dada0228/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
# Fail fast instead of holding two four-GPU jobs under a five-GPU user allowance.
# This only serializes this new experiment, not other users or unrelated jobs.
exec 9>"$RUN/gpu_execution.lock"
flock -n 9 || { echo 'Another followup GPU job is active; submit sequentially';exit 2; }
if [ "$MODE" = main ]; then
    : "${SMOKE_JOB:?matching successful role Smoke required}"
    "$PY" -B -m experiments.linear_probe.followup_v1.hardware check-smoke \
       --audit "$AUDIT" --role "$ROLE" --smoke-job "$SMOKE_JOB"
fi
SESSIONS=(smoke)
if [ "$MODE" = main ]; then
    if [ "$ROLE" = teacher ];then SESSIONS=(warm restart_0 restart_1 restart_2);else SESSIONS=(main);fi
fi
SERVER_PID='';MONITOR_PID='';CLIENT_PID=''
cleanup() {
    trap '' TERM INT
    if [ -n "$CLIENT_PID" ];then kill -TERM "$CLIENT_PID" 2>/dev/null || true;wait "$CLIENT_PID" || true;fi
    if [ -n "$SERVER_PID" ];then kill -TERM "$SERVER_PID" 2>/dev/null || true;wait "$SERVER_PID" || true;fi
    if [ -n "$MONITOR_PID" ];then kill -TERM "$MONITOR_PID" 2>/dev/null || true;wait "$MONITOR_PID" || true;fi
}
on_exit() {
    CODE=$?
    cleanup
    if [ "$CODE" != 0 ];then
        "$PY" -B -m experiments.linear_probe.followup_v1.hardware abort --role "$ROLE" \
           --mode "$MODE" --audit "$AUDIT" --exit-code "$CODE" || true
    fi
    exit "$CODE"
}
trap on_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
INDEX=0
for SESSION in "${SESSIONS[@]}"; do
    PORT=$((20000 + SLURM_JOB_ID % 20000 + INDEX))
    INDEX=$((INDEX+1))
    API="http://127.0.0.1:$PORT/v1"
    "$PY" -B -m experiments.linear_probe.followup_v1.hardware preflight \
       --role "$ROLE" --session "$SESSION" --audit "$AUDIT"
    "$PY" -B - "$PORT" <<'PY'
import socket,sys
with socket.socket() as s:s.bind(('127.0.0.1',int(sys.argv[1])))
PY
    "$PY" -B -m experiments.linear_probe.followup_v1.server --role "$ROLE" --session "$SESSION" \
       --audit "$AUDIT" --port "$PORT" > "$RUN/server_${ROLE}_${SLURM_JOB_ID}_${SESSION}.log" 2>&1 &
    SERVER_PID=$!
    "$PY" -B -m experiments.linear_probe.followup_v1.hardware monitor \
       --role "$ROLE" --session "$SESSION" --audit "$AUDIT" --server-pid "$SERVER_PID" &
    MONITOR_PID=$!
    READY=0
    for i in $(seq 1 180);do
        if ! kill -0 "$SERVER_PID" 2>/dev/null || ! kill -0 "$MONITOR_PID" 2>/dev/null;then exit 3;fi
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
    if [ "$READY" != 1 ];then echo 'Server readiness timeout';exit 3;fi
    "$PY" -B -m experiments.linear_probe.followup_v1.client --role "$ROLE" --session "$SESSION" \
       --audit "$AUDIT" --api-base "$API" --server-pid "$SERVER_PID" &
    CLIENT_PID=$!
    while kill -0 "$CLIENT_PID" 2>/dev/null;do
        if ! kill -0 "$SERVER_PID" 2>/dev/null || ! kill -0 "$MONITOR_PID" 2>/dev/null;then exit 3;fi
        sleep 1
    done
    CODE=0;wait "$CLIENT_PID" || CODE=$?;CLIENT_PID=''
    if [ "$CODE" != 0 ];then exit "$CODE";fi
    kill -TERM "$SERVER_PID" 2>/dev/null || true;wait "$SERVER_PID" || true;SERVER_PID=''
    kill -TERM "$MONITOR_PID" 2>/dev/null || true;wait "$MONITOR_PID";MONITOR_PID=''
done
"$PY" -B -m experiments.linear_probe.followup_v1.hardware finish --role "$ROLE" --mode "$MODE" --audit "$AUDIT"
