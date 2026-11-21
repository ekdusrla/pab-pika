#!/bin/bash
#SBATCH --job-name=probe-server
#SBATCH --partition=batch_ce_ugrad
#SBATCH --account=ugrad_ce
#SBATCH --qos=qos_dada0228_2026_2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:normal:1
#SBATCH --mem=48G
#SBATCH --time=6-00:00:00
#SBATCH --output=/data/dada0228/plan-budget/P-and-B/experiments/linear_probe/runs/server_%j.log

# Server ONLY. User submits; never runs an existing Baseline client.
# Teacher: sbatch --export=ALL,ROLE=teacher experiments/linear_probe/serve.sbatch
# Solver:  sbatch --gres=gpu:normal:4 --export=ALL,ROLE=solver experiments/linear_probe/serve.sbatch
# Run CPU/API clients on this same allocated node (server binds loopback).
set -euo pipefail
cd /data/dada0228/plan-budget/P-and-B
PY=${VLLM_PY:-/data/dada0228/conda_envs/vllm_server/bin/python}
ROLE=${ROLE:-teacher}
PORT=${PORT:-8000}
export HF_HOME=/data/dada0228/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
case "$ROLE" in
  teacher)
    MODEL=meta-llama/Llama-3.1-8B-Instruct
    REVISION=0e9e39f249a16976918f6564b8830bc894c89659
    TP=1
    CONTEXT=8192
    ;;
  solver)
    MODEL=Qwen/QwQ-32B
    REVISION=976055f8c83f394f35dbd3ab09a285a984907bd0
    TP=4
    CONTEXT=12288
    ;;
  *) echo 'ROLE must be teacher or solver'; exit 2 ;;
esac
CONFIG=${SERVER_CONFIG:-experiments/linear_probe/runs/server_${SLURM_JOB_ID}.json}
# Read-only cache check, then exclusive NEW server metadata, before weight loading.
"$PY" -B - "$CONFIG" "$MODEL" "$REVISION" "$TP" "$CONTEXT" "$PORT" <<'PY'
import json
import os
import sys
from pathlib import Path
from experiments.linear_probe.common import save_new, file_hash, runtime
output, model, revision, tp, context, port = sys.argv[1:]
snapshot = Path(os.environ['HF_HUB_CACHE']) / ('models--' + model.replace('/', '--')) / 'snapshots' / revision
index = json.loads((snapshot / 'model.safetensors.index.json').read_text())
if not all((snapshot / name).is_file() for name in set(index['weight_map'].values())):
    raise SystemExit('Incomplete offline model snapshot; no download attempted')
config = {'model': model, 'revision': revision, 'temperature': .1, 'top_p': .9,
          'server_settings': {'dtype': 'float16', 'quantization': None,
              'tensor_parallel_size': int(tp), 'max_model_len': int(context),
              'max_num_seqs': 1, 'gpu_memory_utilization': .85, 'host': '127.0.0.1', 'port': int(port)},
          'launch_script_sha256': file_hash('experiments/linear_probe/serve.sbatch'),
          'runtime': runtime(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID')}
save_new(output, config)
print('Server configuration:', output, flush=True)
PY
ARGS=(--model "$MODEL" --revision "$REVISION" --tokenizer-revision "$REVISION"
      --tensor-parallel-size "$TP" --dtype float16 --max-model-len "$CONTEXT"
      --max-num-seqs 1 --gpu-memory-utilization 0.85 --seed 42 --host 127.0.0.1 --port "$PORT")
if [ "$ROLE" = teacher ]; then ARGS+=(--guided-decoding-backend outlines); fi
exec "$PY" -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
