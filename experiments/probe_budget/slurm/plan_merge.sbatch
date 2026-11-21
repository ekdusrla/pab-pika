#!/bin/bash
#SBATCH --job-name=pb-plan-merge
#SBATCH --partition=batch_ugrad
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Merge shard records -> dataset/<MATH-500|TravelPlanner>/plan_v{2,3}_*.json. Fails if any item is missing.
source experiments/probe_budget/slurm/env.sh
conda activate "$CONDA_PROBE_ENV"
export CUDA_VISIBLE_DEVICES=""
cd experiments/probe_budget
case "${STAGE:?STAGE=v2|v3}" in
  v2) python -u plan_math500.py --merge ;;
  v3) python -u plan_v3.py --dataset "${DATASET:?DATASET=math500|travel}" --merge ;;
  *) echo "unknown STAGE=$STAGE"; exit 1 ;;
esac
