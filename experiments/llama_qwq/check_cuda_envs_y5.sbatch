#!/bin/bash
#SBATCH --job-name=cuda-envs-y5
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
#SBATCH --output=/data/dada0228/plan-budget/P-and-B/logs/cuda_envs_y5_%j.log

set -u

echo "===== NODE ====="
hostname

echo "===== DRIVER ====="
nvidia-smi --query-gpu=index,name,driver_version --format=csv

for ENV_NAME in vllm_server plan_budget_vllm plan_budget; do
    PY="/data/dada0228/conda_envs/${ENV_NAME}/bin/python"

    echo ""
    echo "===== TEST ENV: ${ENV_NAME} ====="

    if [ ! -x "$PY" ]; then
        echo "Python executable not found"
        continue
    fi

    "$PY" - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

try:
    x = torch.ones(10, device="cuda:0")
    print("CUDA TEST PASSED:", x.sum().item())
except Exception as e:
    print("CUDA TEST FAILED:", repr(e))
PY

done
