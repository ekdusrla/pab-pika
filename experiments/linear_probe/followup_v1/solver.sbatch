#!/bin/bash
# Requires FOUR free 3090s on this one node; current snapshot has none free on y2.
#SBATCH --job-name=five-arm-qwq
#SBATCH --partition=batch_ce_ugrad
#SBATCH --account=ugrad_ce
#SBATCH --qos=qos_dada0228_2026_2
#SBATCH --nodelist=moana-y2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:normal:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/dada0228/plan-budget/P-and-B/experiments/linear_probe/runs/followup_v1/solver_%j.log
set -euo pipefail
export ROLE=solver
exec bash /data/dada0228/plan-budget/P-and-B/experiments/linear_probe/followup_v1/launch.sh
