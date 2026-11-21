#!/bin/bash
#SBATCH --job-name=qwq-pab-ablation100
#SBATCH --partition=batch_ce_ugrad
#SBATCH --account=ugrad_ce
#SBATCH --qos=qos_dada0228_2026_2
#SBATCH --nodelist=moana-y4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:normal:4
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/dada0228/plan-budget/P-and-B/experiments/llama_qwq/runs/math500_llama31_8b_ablation_pilot_v1/client_%j.log

set -euo pipefail
cd /data/dada0228/plan-budget/P-and-B
PY=/data/dada0228/conda_envs/vllm_server/bin/python
CLIENT=experiments/llama_qwq/ablation_pilot_v1/pilot.py
RUN=experiments/llama_qwq/runs/math500_llama31_8b_ablation_pilot_v1
MODEL=Qwen/QwQ-32B
REVISION=976055f8c83f394f35dbd3ab09a285a984907bd0
export HF_HOME=/data/dada0228/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
RESUME=${RESUME:-0}
RETRY_UNKNOWN=${RETRY_UNKNOWN:-0}
PORT=${PORT:-8000}
if [[ "$RESUME" != 0 && "$RESUME" != 1 ]] || [[ "$RETRY_UNKNOWN" != 0 && "$RETRY_UNKNOWN" != 1 ]]; then
    echo 'RESUME and RETRY_UNKNOWN must be 0 or 1'; exit 2
fi
test -f "$RUN/protocol.json"
exec 9>"$RUN/execution.job.lock"
flock -n 9 || { echo 'Another job holds this Pilot execution lock'; exit 4; }
ARGS=()
if [ "$RESUME" = 1 ]; then ARGS+=(--resume); fi
if [ "$RETRY_UNKNOWN" = 1 ]; then ARGS+=(--retry-unknown); fi
# Check frozen data, prompts, environment, context, and resume state before GPU setup.
CHECK_CODE=0
"$PY" -B "$CLIENT" check "${ARGS[@]}" --require-pending || CHECK_CODE=$?
if [ "$CHECK_CODE" = 10 ]; then exit 0; fi
if [ "$CHECK_CODE" != 0 ]; then exit "$CHECK_CODE"; fi
if [ "${PREFLIGHT_ONLY:-0}" = 1 ]; then exit 0; fi
export TMPDIR
TMPDIR=$(mktemp -d "/tmp/pab_pilot_${SLURM_JOB_ID}.XXXXXX")
SERVER_LOG="$RUN/server_${SLURM_JOB_ID}.log"
if [ -e "$SERVER_LOG" ]; then echo 'Existing server log protected'; exit 2; fi
SERVER_PID=""
CLIENT_PID=""
cleanup() {
    trap '' TERM INT
    if [ -n "$CLIENT_PID" ] && kill -0 "$CLIENT_PID" 2>/dev/null; then
        kill -TERM "$CLIENT_PID" 2>/dev/null || true
        for i in $(seq 1 60); do
            if ! kill -0 "$CLIENT_PID" 2>/dev/null; then break; fi
            sleep 1
        done
    fi
    if [ -n "$SERVER_PID" ]; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
"$PY" -B - "$PORT" <<'PYCODE'
import socket, sys
with socket.socket() as sock:
    sock.bind(('127.0.0.1', int(sys.argv[1])))
PYCODE
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --revision "$REVISION" --tokenizer-revision "$REVISION" \
    --tensor-parallel-size 4 --dtype float16 --max-model-len 12288 \
    --gpu-memory-utilization 0.85 --max-num-seqs 1 --seed 42 \
    --host 127.0.0.1 --port "$PORT" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
READY=0
for i in $(seq 1 180); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then tail -n 60 "$SERVER_LOG"; exit 3; fi
    if "$PY" -B - "$PORT" "$MODEL" <<'PYCODE'
import json, sys
from urllib.request import urlopen
try:
    with urlopen('http://127.0.0.1:' + sys.argv[1] + '/v1/models', timeout=3) as response:
        data = json.load(response)
    assert any(r['id'] == sys.argv[2] for r in data['data'])
except Exception:
    raise SystemExit(1)
PYCODE
    then READY=1; break; fi
    sleep 10
done
if [ "$READY" != 1 ]; then tail -n 60 "$SERVER_LOG"; exit 3; fi
"$PY" -B "$CLIENT" run "${ARGS[@]}" --api-base "http://127.0.0.1:$PORT/v1" --server-pid "$SERVER_PID" &
CLIENT_PID=$!
while kill -0 "$CLIENT_PID" 2>/dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo 'Server exited; checkpoint preserved'; exit 3; fi
    sleep 1
done
CODE=0
wait "$CLIENT_PID" || CODE=$?
CLIENT_PID=""
exit "$CODE"
