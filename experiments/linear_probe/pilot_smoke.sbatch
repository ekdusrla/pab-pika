#!/bin/bash
#SBATCH --job-name=probe-pilot-smoke
#SBATCH --partition=batch_ce_ugrad
#SBATCH --account=ugrad_ce
#SBATCH --qos=qos_dada0228_2026_2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/dada0228/plan-budget/P-and-B/experiments/linear_probe/runs/pilot_smoke_%j.log

# User submits only. This job does NOT run collection, Teacher, training or Solver.
set -euo pipefail
cd /data/dada0228/plan-budget/P-and-B
export HF_HOME=/data/dada0228/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
HF_PY=${HF_PY:-/data/dada0228/conda_envs/vllm_server/bin/python}
exec "$HF_PY" -B -m experiments.linear_probe.pilot_smoke \
  --data experiments/linear_probe/runs/pilot_v1/data.json \
  --context-plan experiments/linear_probe/runs/pilot_v1/context_plan.json \
  --model-path "$HF_HUB_CACHE/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" \
  --output "experiments/linear_probe/runs/pilot_v1/smoke_${SLURM_JOB_ID}.json"
