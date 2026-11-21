#!/bin/bash
#SBATCH --job-name=pb-gen
#SBATCH --partition=batch_ugrad
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Step 2: probe training data. Per problem: LLaMA call1 (marker decomposition, hidden states at
# markers) + call2 (official PaB level/credit prompt) -> data_v2/gen/problems/{id}.pt
# Resumable. Smoke (20 problems, 4/level):  sbatch --array=0 --export=ALL,SMOKE=1 ...
source experiments/probe_budget/slurm/env.sh
conda activate "$CONDA_PROBE_ENV"
cd experiments/probe_budget
nvidia-smi --query-gpu=name,memory.total --format=csv
if [ -n "${SMOKE:-}" ]; then
    python -u generate_training_data_v2.py --per-level 4 --out data_v2/gen_smoke
    python -u smoke_report.py data_v2/gen_smoke
else
    python -u generate_training_data_v2.py --shard-id $SLURM_ARRAY_TASK_ID --num-shards 8 --out data_v2/gen
fi
