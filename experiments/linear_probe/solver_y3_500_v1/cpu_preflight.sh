#!/bin/bash
set -euo pipefail
cd /data/dada0228/plan-budget/P-and-B
export HF_HOME=/data/dada0228/huggingface HF_HUB_CACHE=/data/dada0228/huggingface/hub
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
D=experiments/linear_probe/solver_y3_500_v1
R=experiments/linear_probe/runs/solver_y3_500_v1
PY=/data/dada0228/conda_envs/shortcut-xai/bin/python
VLLM=/data/dada0228/conda_envs/vllm_server/bin/python
bash -n "$D/gpu.sbatch"
bash -n "$D/launch.sh"
(cd "$R" && sha256sum -c candidate_manifest.sha256)
"$VLLM" -B - <<'PY'
from importlib.metadata import version
from transformers import AutoTokenizer
from experiments.linear_probe.solver_y3_500_v1.audit import PILOT, OUT
from experiments.linear_probe.common import digest, read, verify
from math_answer_eval import evaluator_info
candidate=verify(read(OUT/'candidate_manifest.json'))
pilot=verify(read(PILOT))
assert evaluator_info('original')==pilot['evaluator']
assert all(version(k)==v for k,v in pilot['legacy_solver_config']['generation_packages'].items())
cap=verify(read(OUT.parent/'pilot_v1/evaluation_recovered.json'))
for name, model in candidate['model_identity'].items():
    tokenizer=AutoTokenizer.from_pretrained(model['snapshot'],local_files_only=True)
    if name=='planner':
        assert digest(tokenizer.chat_template)==cap['sessions'][0]['chat_template_sha256']
    print(name, tokenizer.__class__.__name__, version('transformers'), version('tokenizers'), digest(tokenizer.chat_template))
print('CPU package identities and tokenizer load PASS; runtime CUDA not exercised')
PY
"$PY" -B -m unittest -v experiments.linear_probe.solver_y3_500_v1.tests \
    experiments.linear_probe.solver_y1_v1.tests_identity experiments.linear_probe.solver_y1_v1.tests_mapping
