#!/bin/bash
#SBATCH --job-name=cuda-init-diag
#SBATCH --partition=batch_ce_ugrad
#SBATCH --account=ugrad_ce
#SBATCH --qos=qos_dada0228_2026_2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=cuda_diag_%j.log
set -euo pipefail
cd /data/dada0228/plan-budget/P-and-B
: "${SLURM_JOB_ID:?Require a SLURM allocation}"
work=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/pab-cuda-${SLURM_JOB_ID}.XXXXXXXX")
cc -Wall -Wextra -Werror experiments/linear_probe/solver_y3_500_smoke_v1/cuinit_native.c -ldl -o "$work/cuinit"
trap 'unlink "$work/cuinit"; rmdir "$work"' EXIT
/data/dada0228/conda_envs/vllm_server/bin/python -B \
  -m experiments.linear_probe.solver_y3_500_smoke_v1.cuda_diagnostics one-gpu --native "$work/cuinit"
