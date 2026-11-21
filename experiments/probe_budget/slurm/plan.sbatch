#!/bin/bash
#SBATCH --job-name=pb-plan
#SBATCH --partition=batch_ugrad
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --exclude=ariel-v8
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Step 4: LLaMA planning, 4 shards, sequential within a shard (= single-stream latency on 1 A5000).
#   STAGE=v2                  MATH-500: call1 (marker) -> probe, + PaB call2   (plan_math500.py)
#   STAGE=v3 DATASET=math500  MATH-500: joint single call                     (plan_v3.py)
#   STAGE=v3 DATASET=travel   TravelPlanner: call1 -> probe, PaB call2, joint (plan_v3.py)
# then: sbatch --export=ALL,STAGE=...,DATASET=... experiments/probe_budget/slurm/plan_merge.sbatch
# SMOKE=N (v3 only): N items per shard into data_v2/<out>/smoke/
source experiments/probe_budget/slurm/env.sh
conda activate "$CONDA_PROBE_ENV"
cd experiments/probe_budget
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
case "${STAGE:?STAGE=v2|v3}" in
  v2) python -u plan_math500.py --shard-id $SLURM_ARRAY_TASK_ID --num-shards 4 ;;
  v3) python -u plan_v3.py --dataset "${DATASET:?DATASET=math500|travel}" \
          --shard-id $SLURM_ARRAY_TASK_ID --num-shards 4 --smoke "${SMOKE:-0}" ;;
  *) echo "unknown STAGE=$STAGE"; exit 1 ;;
esac
