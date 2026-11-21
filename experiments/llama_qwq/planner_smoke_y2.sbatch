#!/bin/bash
#SBATCH --job-name=llama-planner-5
#SBATCH --partition=batch_ce_ugrad
#SBATCH --account=ugrad_ce
#SBATCH --qos=qos_dada0228_2026_2
#SBATCH --nodelist=moana-y2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:normal:4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/data/dada0228/plan-budget/P-and-B/logs/llama_planner_%j.log

set -euo pipefail

cd /data/dada0228/plan-budget/P-and-B

PY=/data/dada0228/conda_envs/vllm_server/bin/python

export HF_HOME=/data/dada0228/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TMPDIR=/data/dada0228/tmp

mkdir -p "$TMPDIR" logs results/llama_qwq

MODEL=casperhansen/llama-3.3-70b-instruct-awq
API_BASE=http://127.0.0.1:8000/v1

echo "===== GPU ====="
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv

echo "===== START LLAMA SERVER ====="

"$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size 4 \
    --quantization awq \
    --dtype float16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 2 \
    --host 127.0.0.1 \
    --port 8000 \
    > "logs/llama_server_${SLURM_JOB_ID}.log" 2>&1 &

SERVER_PID=$!

cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "===== WAITING FOR SERVER ====="

READY=0

for i in $(seq 1 180); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: vLLM server terminated"
        tail -n 100 "logs/llama_server_${SLURM_JOB_ID}.log"
        exit 1
    fi

    if curl -fsS "$API_BASE/models" >/dev/null 2>&1; then
        READY=1
        break
    fi

    sleep 10
done

if [ "$READY" -ne 1 ]; then
    echo "ERROR: Server startup timeout"
    tail -n 100 "logs/llama_server_${SLURM_JOB_ID}.log"
    exit 1
fi

echo "===== RUN PLANNER ====="

"$PY" experiments/llama_qwq/run_planner.py \
    --limit 5 \
    --api-base "$API_BASE" \
    --output "results/llama_qwq/planner_5_${SLURM_JOB_ID}.json"

echo "===== FINISHED ====="
