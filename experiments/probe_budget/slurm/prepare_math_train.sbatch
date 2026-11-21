#!/bin/bash
#SBATCH --job-name=pb-prepare
#SBATCH --partition=batch_ugrad
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Step 1: MATH train split minus MATH-500 overlap, 400 problems per level (seed 42)
#   -> data_v2/math_train_sample2000.json (already committed; rerun only to regenerate)
source experiments/probe_budget/slurm/env.sh
conda activate "$CONDA_PROBE_ENV"
cd experiments/probe_budget
python -u prepare_math_train.py
