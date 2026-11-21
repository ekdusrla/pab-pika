#!/bin/bash
#SBATCH --job-name=qwq-llama31-8b-compare
#SBATCH --partition=batch_ce_ugrad
#SBATCH --account=ugrad_ce
#SBATCH --qos=qos_dada0228_2026_2
#SBATCH --nodelist=moana-y4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:normal:4
#SBATCH --mem=48G
#SBATCH --time=6-00:00:00
#SBATCH --output=/data/dada0228/plan-budget/P-and-B/logs/llama31_8b_solver_%j.log

#SBATCH --signal=B:TERM@120

# Dedicated 8B-Planner experiment; never resume the original 70B checkpoint here.
# First run: sbatch --export=ALL,LIMIT=500,RESUME=0 <this-script>
# Continue:  sbatch --export=ALL,LIMIT=500,RESUME=1 <this-script>
# The six-day limit is an upper bound; client completion triggers server cleanup.
# Original Vanilla requests are compatible, but no checkpoint import is performed.
# Both methods run afresh; the shared runner binds checkpoints to Planner config.

ROLE=qwq
MODEL=Qwen/QwQ-32B
REVISION=976055f8c83f394f35dbd3ab09a285a984907bd0
MAX_MODEL_LEN=${MAX_MODEL_LEN:-12288}
OUTPUT=${OUTPUT:-experiments/llama_qwq/runs/math500_llama31_8b_v1/solver.json}
CLIENT_SCRIPT=run_solver_compare.py
set -euo pipefail
cd /data/dada0228/plan-budget/P-and-B
PY=/data/dada0228/conda_envs/vllm_server/bin/python
export HF_HOME=/data/dada0228/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR
TMPDIR=$(mktemp -d "/tmp/pab_${SLURM_JOB_ID}.XXXXXX")
MANIFEST=${MANIFEST:-experiments/llama_qwq/math500_manifest.json}
LIMIT=${LIMIT:-500}
RESUME=${RESUME:-0}
PORT=${PORT:-8000}
API_BASE="http://127.0.0.1:${PORT}/v1"
PLANNER_INPUT=${PLANNER_INPUT:-experiments/llama_qwq/runs/math500_llama31_8b_v1/planner.json}
DECAY=${DECAY:-constant}
MAX_TOKENS=${MAX_TOKENS:-8192}
TEMPERATURE=${TEMPERATURE:-0.1}
TOP_P=${TOP_P:-0.9}
SEED=${SEED:-42}
TIMEOUT=${TIMEOUT:-600}
RETRIES=${RETRIES:-2}
RETRY_WAIT=${RETRY_WAIT:-2}
# Require the complete frozen 500-question plan before cache/tokenizer/server setup.
# This is read-only and cannot initialize or alter any checkpoint.
"$PY" -B experiments/llama_qwq/check_8b_planner.py \
    --planner "$PLANNER_INPUT" --manifest "$MANIFEST"
# Never allow environment overrides to write into the existing 70B run.
"$PY" -B - "$OUTPUT" <<'GUARD_PY'
import sys
from pathlib import Path
old = Path('experiments/llama_qwq/runs/math500_original_v1').resolve()
if Path(sys.argv[1]).resolve().is_relative_to(old):
    raise SystemExit('OUTPUT must not point into the protected 70B experiment')
GUARD_PY
mkdir -p logs "$(dirname "$OUTPUT")"
# Held across server + client: duplicate submissions for the same output fail early.
exec 9>"${OUTPUT}.job.lock"
flock -n 9 || { echo "Another job holds $OUTPUT"; exit 4; }
if [ "$RESUME" != 0 ] && [ "$RESUME" != 1 ]; then
    echo 'RESUME must be 0 or 1'; exit 2
fi
if [ "$RESUME" = 0 ] && { [ -e "$OUTPUT" ] || [ -e "${OUTPUT}.tmp" ]; }; then
    echo 'Existing checkpoint protected; set RESUME=1'; exit 2
fi
if [ "$RESUME" = 1 ] && [ ! -e "$OUTPUT" ] && [ ! -e "${OUTPUT}.tmp" ]; then
    echo 'No checkpoint to resume'; exit 2
fi
SNAPSHOT="${HF_HUB_CACHE}/models--${MODEL//\//--}/snapshots/${REVISION}"
SERVER_LOG="logs/llama31_8b_${ROLE}_server_${SLURM_JOB_ID}.log"
SERVER_PID=""
CLIENT_PID=""
cleanup() {
    trap '' TERM INT
    if [ -n "$CLIENT_PID" ] && kill -0 "$CLIENT_PID" 2>/dev/null; then
        kill -TERM "$CLIENT_PID" 2>/dev/null || true
        # Give the client time to commit checkpoint before stopping the server.
        for i in $(seq 1 60); do
            if ! kill -0 "$CLIENT_PID" 2>/dev/null; then break; fi
            sleep 1
        done
    fi
    if [ -n "$SERVER_PID" ]; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# CPU-only preflight: never import dataset/utils package initializers or load weights.
"$PY" -B - "$ROLE" "$MANIFEST" "$LIMIT" "$SNAPSHOT" "$MAX_MODEL_LEN" "$MAX_TOKENS" "$PLANNER_INPUT" "$DECAY" <<'PYCODE'
import json
import sys
from pathlib import Path
sys.path.insert(0, 'experiments/llama_qwq')
from experiment_state import read_manifest, original_definitions, versions
from transformers import AutoTokenizer
role, path, limit, snapshot, context, cap, planner, decay = sys.argv[1:]
limit, context, cap = int(limit), int(context), int(cap)
if not 1 <= limit <= 500:
    raise SystemExit('LIMIT must be between 1 and 500')
manifest = read_manifest(path)
print('Manifest:', manifest['fingerprint'], manifest['count'], flush=True)
print('Packages:', versions(), flush=True)
if not (Path(snapshot) / 'config.json').is_file():
    raise SystemExit('Missing local model cache: ' + snapshot + '; pinned revision config.json is absent or a broken symlink. Prepare the complete authorized snapshot (config, tokenizer, weight index and shards) before submission. Offline mode is intentional; authentication/access status is not established by this check.')
config = json.loads((Path(snapshot) / 'config.json').read_text())
index = json.loads((Path(snapshot) / 'model.safetensors.index.json').read_text())
if not all((Path(snapshot) / name).is_file() for name in set(index['weight_map'].values())):
    raise SystemExit('Incomplete local cache; no download will be attempted')
print('Model architecture:', config['architectures'], 'revision:', Path(snapshot).name, flush=True)
tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
prompts, instruction, output_format = original_definitions()
plans = None
if role == 'qwq':
    from run_solver_compare import planner_rows, messages_for, allocate_words, LEVEL_WORDS
    from math_answer_eval import evaluator_info
    print('Evaluator:', evaluator_info('original'), flush=True)
    source = json.loads(Path(planner).read_text())
    if source['config']['model'] != 'meta-llama/Llama-3.1-8B-Instruct':
        raise SystemExit('Expected the new Llama-3.1-8B Planner checkpoint')
    plans = planner_rows(source, manifest)
maximum = 0
for row in manifest['records'][:limit]:
    candidates = []
    if role == 'llama':
        template = prompts['break_down']
        values = {'domain': 'math', 'problem': row['query'], 'level': row['level']}
        candidates.append(([{'role': key, 'content': template[key].format(**values)} for key in ('system', 'user')], 1024))
    else:
        plan = plans[row['id']]
        item = {**row, 'steps': plan.get('steps', []), 'credits': plan.get('credits', [])}
        budgets = allocate_words(item['credits'], LEVEL_WORDS[item['level']], decay) if plan['status'] == 'success' else []
        methods = ('vanilla', 'weighted_pab') if plan['status'] == 'success' else ('vanilla',)
        candidates = [(messages_for(item, method, budgets, instruction, output_format), cap) for method in methods]
    for messages, output_cap in candidates:
        count = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        maximum = max(maximum, count + output_cap)
        if count + output_cap > context:
            raise SystemExit('Context exceeded for id ' + row['id'] + '; do not silently change budgets')
print('CPU preflight passed; largest checked input+output:', maximum, flush=True)
print('Planner assessment prompts are checked after decomposition, before each API call.', flush=True)
PYCODE
ARGS=(--manifest "$MANIFEST" --output "$OUTPUT" --limit "$LIMIT" --api-base "$API_BASE"
    --model "$MODEL" --revision "$REVISION" --seed "$SEED" --timeout "$TIMEOUT"
    --retries "$RETRIES" --retry-wait "$RETRY_WAIT"
    --max-model-len "$MAX_MODEL_LEN" --tokenizer-path "$SNAPSHOT")
if [ "$RESUME" = 1 ]; then ARGS+=(--resume); fi
if [ "$ROLE" = qwq ]; then
    ARGS+=(--planner-input "$PLANNER_INPUT" --max-tokens "$MAX_TOKENS"
        --temperature "$TEMPERATURE" --top-p "$TOP_P" --decay "$DECAY" --order alternate --evaluator original)
fi
# Script-local metadata adapter: shared 70B runners remain unchanged.
# Both check-only and execution use exactly the same configuration adapter.
run_client() {
    exec "$PY" -B - "${ARGS[@]}" "$@" <<'CLIENT_PY'
import sys
from pathlib import Path
sys.path.insert(0, 'experiments/llama_qwq')
import run_solver_compare as runner
original_hash = runner.file_hash
def launcher_hash(path):
    if Path(path).resolve() == Path(runner.__file__).with_name('solver_compare_y4.sbatch').resolve():
        path = Path('experiments/llama_qwq/solver_llama31_8b_y4.sbatch')
    return original_hash(path)
runner.file_hash = launcher_hash
sys.argv[0] = runner.__file__
raise SystemExit(runner.main())
CLIENT_PY
}
(run_client --check-only)
if [ "${PREFLIGHT_ONLY:-0}" = 1 ]; then exit 0; fi

"$PY" -B - "$PORT" <<'PYCODE'
import socket
import sys
with socket.socket() as sock:
    sock.bind(('127.0.0.1', int(sys.argv[1])))
PYCODE
nvidia-smi --query-gpu=index,name,memory.total --format=csv
SERVER_ARGS=(--model "$MODEL" --revision "$REVISION" --tokenizer-revision "$REVISION"
    --tensor-parallel-size 4 --dtype float16 --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization 0.85 --max-num-seqs 1 --seed "$SEED"
    --host 127.0.0.1 --port "$PORT")
if [ "$ROLE" = llama ]; then
    SERVER_ARGS+=(--quantization awq --guided-decoding-backend outlines)
fi
"$PY" -m vllm.entrypoints.openai.api_server "${SERVER_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
READY=0
for i in $(seq 1 180); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then tail -n 100 "$SERVER_LOG"; exit 3; fi
    if "$PY" -B - "$API_BASE" "$MODEL" <<'PYCODE'
import json
import sys
from urllib.request import urlopen
try:
    with urlopen(sys.argv[1] + '/models', timeout=3) as stream:
        data = json.load(stream)
    assert any(row['id'] == sys.argv[2] for row in data['data'])
except Exception:
    raise SystemExit(1)
PYCODE
    then READY=1; break; fi
    sleep 10
 done
if [ "$READY" != 1 ]; then tail -n 100 "$SERVER_LOG"; exit 3; fi
run_client --server-pid "$SERVER_PID" &
CLIENT_PID=$!
while kill -0 "$CLIENT_PID" 2>/dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo 'Server died; terminating client to preserve checkpoint for RESUME=1'
        exit 3
    fi
    sleep 1
 done
CODE=0
wait "$CLIENT_PID" || CODE=$?
CLIENT_PID=""
exit "$CODE"
