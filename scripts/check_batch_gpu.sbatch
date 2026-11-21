#!/usr/bin/bash
#SBATCH -J batch-gpu-check
#SBATCH -A ugrad_ce
#SBATCH -p batch_ce_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=4G
#SBATCH -t 00:05:00
#SBATCH -o logs/batch-gpu-check-%j.out

set -euo pipefail

echo "===== HOST ====="
hostname

echo "===== GPU ====="
nvidia-smi --query-gpu=index,name,memory.total --format=csv

echo "===== DISK ====="
df -h /tmp /data
