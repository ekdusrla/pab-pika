#!/bin/bash
# Invoked under the TP4 gpu.sbatch allocation, never on master.
set -euo pipefail
cd /data/dada0228/plan-budget/P-and-B
: "${SLURMD_NODENAME:?compute node required}"
: "${SERVER_CONFIG:?new TP4 server metadata path required}"
: "${PORT:?same-allocation API port required}"
PY=${VLLM_PY:-/data/dada0228/conda_envs/vllm_server/bin/python}
export HF_HOME=/data/dada0228/huggingface HF_HUB_CACHE=/data/dada0228/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
"$PY" -B - "$SERVER_CONFIG" "$PORT" <<'PY'
import copy,importlib.metadata,os,sys
from experiments.linear_probe.common import save_new,file_hash,runtime
from experiments.linear_probe.credit_a2000_tp4.audit import SERVER,POLICY
if importlib.metadata.version('vllm')!=POLICY['vllm_version']:raise RuntimeError('Require vLLM 0.6.6.post1')
config=copy.deepcopy(SERVER)
config['server_settings'].update(host='127.0.0.1',port=int(sys.argv[2]))
config.update(slurm_job_id=os.environ['SLURM_JOB_ID'],gpu_policy=POLICY,
    runtime={**runtime(),'engine':'vLLM / mp / TP4 / Outlines'},
    launch_script_sha256=file_hash('experiments/linear_probe/credit_a2000_tp4/serve.sbatch'))
save_new(sys.argv[1],config)
PY
exec "$PY" -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --revision 0e9e39f249a16976918f6564b8830bc894c89659 \
  --tokenizer-revision 0e9e39f249a16976918f6564b8830bc894c89659 \
  --tensor-parallel-size 4 --distributed-executor-backend mp --dtype float16 \
  --max-model-len 8192 --max-num-seqs 1 --gpu-memory-utilization .85 \
  --seed 42 --guided-decoding-backend outlines --host 127.0.0.1 --port "$PORT"
