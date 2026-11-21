#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?SLURM allocation required}"
MANIFEST=${1:?Approved 500-problem manifest required}
shift
CHUNKS=("$@")
if [ "${#CHUNKS[@]}" -eq 0 ]; then CHUNKS=(1 2 3 4 5 6 7 8 9 10); fi
for chunk in "${CHUNKS[@]}"; do [[ "$chunk" =~ ^([1-9]|10)$ ]] || exit 2; done
PY=/data/dada0228/conda_envs/vllm_server/bin/python
CPU=/data/dada0228/conda_envs/shortcut-xai/bin/python
MODULE=experiments.linear_probe.solver_y3_500_v1
OUT=experiments/linear_probe/runs/solver_y3_500_v1
export HF_HOME=/data/dada0228/huggingface HF_HUB_CACHE=/data/dada0228/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
exec 9>"$OUT/batch.lock"
flock -n 9
client=''; server=''; monitor=''
cleanup() {
    trap '' TERM INT
    for pid in "$client" "$server" "$monitor"; do
        if [ -n "$pid" ]; then kill -TERM "$pid" 2>/dev/null || true; fi
    done
    for pid in "$client" "$server" "$monitor"; do
        if [ -n "$pid" ]; then wait "$pid" 2>/dev/null || true; fi
    done
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

run_client() {
    "$@" &
    client=$!
    local rc=0
    wait "$client" || rc=$?
    client=''
    return "$rc"
}

"$PY" -B -m "$MODULE.hardware" preflight --manifest "$MANIFEST"
# Subset only the capture process to the first scheduler-visible device.
IFS=, read -r first_gpu _ <<< "${CUDA_VISIBLE_DEVICES:?CUDA GPU mapping required}"
for c in "${CHUNKS[@]}"; do
    run_client env CUDA_VISIBLE_DEVICES="$first_gpu" "$PY" -B -m "$MODULE.runner" \
        --manifest "$MANIFEST" --chunk "$c" --phase capture
    run_client env CUDA_VISIBLE_DEVICES='' "$CPU" -B -m "$MODULE.runner" \
        --manifest "$MANIFEST" --chunk "$c" --phase probe
done

port=$((20000 + SLURM_JOB_ID % 10000 * 2))
for role in teacher solver; do
    "$PY" -B -c 'import socket, sys; s=socket.socket(); s.bind(("127.0.0.1", int(sys.argv[1])))' "$port"
    "$PY" -B -m "$MODULE.hardware" serve --manifest "$MANIFEST" --role "$role" --port "$port" \
        > "$OUT/server_${SLURM_JOB_ID}_$role.log" 2>&1 &
    server=$!
    "$PY" -B -m "$MODULE.hardware" monitor --manifest "$MANIFEST" --role "$role" --pid "$server" &
    monitor=$!
    ready=0
    for _ in $(seq 1 180); do
        kill -0 "$server"
        kill -0 "$monitor"
        if "$PY" -B -c 'import sys, urllib.request; urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/v1/models", timeout=2).close()' "$port" 2>/dev/null; then
            ready=1; break
        fi
        sleep 5
    done
    [ "$ready" -eq 1 ]
    for c in "${CHUNKS[@]}"; do
        kill -0 "$monitor"
        run_client "$PY" -B -m "$MODULE.runner" --manifest "$MANIFEST" \
            --chunk "$c" --phase "$role" --port "$port" --pid "$server"
    done
    kill -TERM "$server" 2>/dev/null || true
    wait "$server" || true
    server=''
    kill -TERM "$monitor" 2>/dev/null || true
    wait "$monitor"
    monitor=''
    port=$((port + 1))
done
