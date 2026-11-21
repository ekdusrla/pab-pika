#!/bin/bash
#SBATCH --job-name=qwq-y1-frozen
#SBATCH --partition=batch_ce_ugrad
#SBATCH --account=ugrad_ce
#SBATCH --qos=qos_dada0228_2026_2
#SBATCH --nodelist=moana-y1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:normal:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/dada0228/plan-budget/P-and-B/experiments/linear_probe/runs/solver_y1_v1/gpu_%j.log
set -euo pipefail
export ROLE=solver
exec bash /data/dada0228/plan-budget/P-and-B/experiments/linear_probe/solver_y1_v1/launch.sh
