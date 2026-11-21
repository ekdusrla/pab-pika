# Sourced by every sbatch in this folder. Submit all jobs from the REPO ROOT:
#   mkdir -p logs && sbatch experiments/probe_budget/slurm/<job>.sbatch
# Defaults below are what the reported runs used on ARIEL; override by exporting before sbatch.
set -euo pipefail
[ -f experiments/probe_budget/slurm/env.sh ] || { echo "submit from the repo root"; exit 1; }
: "${CONDA_PROBE_ENV:=pab-probe-cu128}"   # torch + transformers + sklearn: LLaMA planning, probe
: "${CONDA_VLLM_ENV:=pab-vllm-cu121}"     # vllm 0.5.4 / torch 2.4.0: QwQ server (no pip install inside)
: "${LLAMA_PATH:=meta-llama/Llama-3.1-8B-Instruct}"   # HF id or local dir
: "${QWQ_PATH:=Qwen/QwQ-32B}"
export LLAMA_PATH QWQ_PATH PYTHONUNBUFFERED=1
# node-local /tmp filled up on ariel-v8 once: set SCRATCH_TMP to a shared dir to redirect TMPDIR
if [ -n "${SCRATCH_TMP:-}" ]; then export TMPDIR=$SCRATCH_TMP/${SLURM_JOB_ID:-local}; mkdir -p "$TMPDIR"; fi
source "$(conda info --base)/etc/profile.d/conda.sh"
echo "job=${SLURM_JOB_ID:-} node=$(hostname) start=$(date -Is)"
