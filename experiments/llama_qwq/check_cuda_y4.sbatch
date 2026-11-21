#!/bin/bash
#SBATCH --job-name=check-cuda-y5
#SBATCH --partition=batch_ce_ugrad
#SBATCH --account=ugrad_ce
#SBATCH --qos=qos_dada0228_2026_2
#SBATCH --nodelist=moana-y4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:normal:4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/data/dada0228/plan-budget/P-and-B/logs/check_cuda_y4_%j.log

set -euo pipefail

PY=/data/dada0228/conda_envs/vllm_server/bin/python

echo "===== NODE ====="
hostname

echo "===== SLURM GPU ENV ====="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset}"

echo "===== NVIDIA SMI ====="
nvidia-smi -L

echo "===== PYTORCH CUDA TEST ====="

"$PY" - <<'PY'
import os
import torch

print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    try:
        torch.cuda.set_device(i)
        x = torch.ones(10, device=f"cuda:{i}")
        print(
            f"GPU {i}:",
            torch.cuda.get_device_name(i),
            "Test:", x.sum().item(),
        )
    except Exception as e:
        print(f"GPU {i} FAILED:", repr(e))
        raise

print("ALL CUDA TESTS PASSED")
PY
