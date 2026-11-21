#!/bin/bash
#SBATCH --job-name=qwq-y3-smoke
#SBATCH --partition=batch_ce_ugrad
#SBATCH --account=ugrad_ce
#SBATCH --qos=qos_dada0228_2026_2
#SBATCH --nodelist=moana-y3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:normal:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/dada0228/plan-budget/P-and-B/experiments/linear_probe/runs/solver_y3_500_v1/smoke_%j.log
set -euo pipefail
cd /data/dada0228/plan-budget/P-and-B
: "${SLURM_JOB_ID:?allocated job required}"
: "${CUDA_VISIBLE_DEVICES:?scheduler GPU visibility required}"
manifest=${1:?approved manifest required}
export PAB_SMOKE_NODE=${2:-moana-y3}
case "$PAB_SMOKE_NODE" in moana-y3|moana-y5) ;; *) exit 2 ;; esac
[ "${SLURMD_NODENAME:?missing allocated node}" = "$PAB_SMOKE_NODE" ]
py=/data/dada0228/conda_envs/vllm_server/bin/python
cpu=/data/dada0228/conda_envs/shortcut-xai/bin/python
module=experiments.linear_probe.solver_y3_500_smoke_v1.smoke
out=experiments/linear_probe/runs/solver_y3_500_v1/smoke
mkdir -p "$out"
export HF_HOME=/data/dada0228/huggingface HF_HUB_CACHE=/data/dada0228/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

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

run_client "$py" -B -m "$module" preflight --manifest "$manifest"
IFS=, read -r first_gpu _ <<< "$CUDA_VISIBLE_DEVICES"
run_client env CUDA_VISIBLE_DEVICES="$first_gpu" "$py" -B -m "$module" capture --manifest "$manifest"
run_client env CUDA_VISIBLE_DEVICES='' "$cpu" -B -m "$module" probe --manifest "$manifest"

port=$((40000 + SLURM_JOB_ID % 10000))
"$py" -B -c 'import socket,sys; s=socket.socket(); s.bind(("127.0.0.1",int(sys.argv[1])))' "$port"
"$py" -B -m "$module" monitor --manifest "$manifest" --pid $$ &
monitor=$!
"$py" -B -m "$module" serve --manifest "$manifest" --port "$port" \
    > "$out/$SLURM_JOB_ID/server.log" 2>&1 &
server=$!
ready=0
for _ in $(seq 1 90); do
    kill -0 "$server"
    kill -0 "$monitor"
    if "$py" -B -c 'import sys,urllib.request; urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/v1/models",timeout=2).close()' "$port" 2>/dev/null; then
        ready=1
        break
    fi
    sleep 5
done
[ "$ready" -eq 1 ]
run_client "$py" -B -m "$module" request --manifest "$manifest" --port "$port" --pid "$server"
kill -0 "$server"
kill -TERM "$monitor"
wait "$monitor"
monitor=''
kill -TERM "$server"
wait "$server" || true
server=''
"$py" -B -m "$module" complete --manifest "$manifest"
