#!/bin/bash
#SBATCH --job-name=cuda-mapping-y5
#SBATCH --partition=batch_ce_ugrad
#SBATCH --account=ugrad_ce
#SBATCH --qos=qos_dada0228_2026_2
#SBATCH --nodelist=moana-y5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:normal:4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/data/dada0228/plan-budget/P-and-B/logs/cuda_mapping_y5_%j.log

set -u

PY=/data/dada0228/conda_envs/vllm_server/bin/python

echo "===== NODE ====="
hostname

echo "===== ORIGINAL ENV ====="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"

test_cuda() {
    echo ""
    echo "===== TEST: $1 ====="

    "$PY" - <<'PY'
import os
import torch

print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
print("CUDA_DEVICE_ORDER:", os.getenv("CUDA_DEVICE_ORDER"))
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())

try:
    x = torch.ones(10, device="cuda:0")
    print("GPU 0 SUCCESS:", x.sum().item())
except Exception as e:
    print("GPU 0 FAILED:", repr(e))
PY
}

test_cuda "SLURM DEFAULT"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
test_cuda "PCI_BUS_ID"

unset CUDA_VISIBLE_DEVICES
test_cuda "CUDA_VISIBLE_DEVICES UNSET"

export CUDA_VISIBLE_DEVICES=0
test_cuda "GPU 0 ONLY"

export CUDA_VISIBLE_DEVICES=0,1,2,3
test_cuda "ALL FOUR GPUS"

echo ""
echo "===== FINISHED ====="
