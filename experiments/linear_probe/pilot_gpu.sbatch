#!/bin/bash
#SBATCH --job-name=linear-probe-pilot
#SBATCH --partition=batch_ce_ugrad
#SBATCH --account=ugrad_ce
#SBATCH --qos=qos_dada0228_2026_2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/data/dada0228/plan-budget/P-and-B/experiments/linear_probe/runs/pilot_%j.log

# User submits with explicit --gres/--cpus-per-task/--mem using submit_pilot.py.
# No node is pinned. Teacher/Solver server+client share the SAME allocation.
set -euo pipefail
# Validate the complete batch script on the allocated compute node, never on master.
bash -n "$0"
cd /data/dada0228/plan-budget/P-and-B
: "${STAGE:?length/capture/teacher/solver required}"
: "${RUN:?new experiment run directory required}"
: "${CPUS_PER_GPU:?approved CPU per GPU policy value required}"
: "${RAM_GB_PER_GPU:?approved RAM GB per GPU policy value required}"
HF_PY=${HF_PY:-/data/dada0228/conda_envs/vllm_server/bin/python}
VLLM_PY=${VLLM_PY:-/data/dada0228/conda_envs/vllm_server/bin/python}
export HF_HOME=/data/dada0228/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
LLAMA="$HF_HUB_CACHE/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
QWQ="$HF_HUB_CACHE/models--Qwen--QwQ-32B/snapshots/976055f8c83f394f35dbd3ab09a285a984907bd0"
EXPECTED_GPUS=1
if [ "$STAGE" = solver ]; then EXPECTED_GPUS=4; fi
"$HF_PY" -B - "$RUN" "$EXPECTED_GPUS" "$CPUS_PER_GPU" "$RAM_GB_PER_GPU" <<'PY'
import os,sys,subprocess
from experiments.linear_probe.common import output_path,save_new,seal
from experiments.linear_probe.resources import parse_nodes,validate_allocation
run,gpus,cpus,ram=sys.argv[1:]; gpus,cpus,ram=map(int,(gpus,cpus,ram))
output_path(run)
assert cpus>0 and ram>0
assert int(os.environ['SLURM_CPUS_PER_TASK'])==gpus*cpus, 'CPU allocation must scale per GPU'
assert int(os.environ['SLURM_MEM_PER_NODE'])==gpus*ram*1024, 'RAM allocation must scale per GPU'
raw=subprocess.check_output(['scontrol','show','node',os.environ['SLURMD_NODENAME'],'-o'],text=True)
nodes=parse_nodes(raw)
assert len(nodes)==1
caps=validate_allocation(nodes[0],gpus,gpus*cpus,gpus*ram*1024)
save_new(run+'/resources_'+os.environ['SLURM_JOB_ID']+'.json',seal({'node':nodes[0],
    'per_gpu_limits':caps,'requested_gpus':gpus,'requested_cpus':gpus*cpus,'requested_memory_gb':gpus*ram}))
import torch
assert torch.cuda.device_count()==gpus
assert all('3090' in torch.cuda.get_device_name(i) for i in range(gpus)), 'Require RTX 3090 allocation'
PY
if [ "$STAGE" = length ]; then
    ARGS=()
    if [ -n "${PREVIOUS:-}" ]; then ARGS+=(--previous "$PREVIOUS"); fi
    exec "$HF_PY" -B -m experiments.linear_probe.length_check run \
        --model-path "$LLAMA" --plan "$RUN/plan.json" --target "${TARGET:?required}" \
        --output "$RUN/length_${TARGET}_${SLURM_JOB_ID}.json" "${ARGS[@]}"
elif [ "$STAGE" = capture ]; then
    case "${CAPTURE_RESUME:-0}:${ACK_SAFETY_STOP:-0}" in
        0:0|1:0|1:1) ;;
        *) echo 'Use CAPTURE_RESUME=0/1; ACK_SAFETY_STOP=1 requires CAPTURE_RESUME=1'; exit 2 ;;
    esac
    for SPLIT in ${SPLITS:-train validation evaluation}; do
        case "$SPLIT" in train|validation|evaluation) ;; *) echo 'Invalid SPLITS'; exit 2 ;; esac
        ARGS=()
        if [ "$SPLIT" = evaluation ] && [ "${CAPTURE_CONTROL:-1}" = 1 ]; then ARGS+=(--capture-control); fi
        CAPTURE_OUTPUT="$RUN/${SPLIT}_capture.json"
        if [ "${CAPTURE_RESUME:-0}" = 1 ] && { [ -e "$CAPTURE_OUTPUT" ] || [ -e "${CAPTURE_OUTPUT}.tmp" ]; }; then
            ARGS+=(--resume)
            if [ "${ACK_SAFETY_STOP:-0}" = 1 ]; then ARGS+=(--resume-after-safety-stop); fi
        fi
        "$HF_PY" -B -m experiments.linear_probe.cli collect --data "$RUN/data.json" \
            --split "$SPLIT" --model-path "$LLAMA" --output "$CAPTURE_OUTPUT" "${ARGS[@]}"
    done
    exit 0
elif [ "$STAGE" != teacher ] && [ "$STAGE" != solver ]; then
    echo 'Unknown STAGE'; exit 2
fi

if [ "$STAGE" = teacher ]; then
    for SPLIT in ${SPLITS:-train validation evaluation}; do
        case "$SPLIT" in train|validation|evaluation) ;; *) echo 'Invalid SPLITS'; exit 2 ;; esac
        "$HF_PY" -B -m experiments.linear_probe.pilot preflight-label --data "$RUN/data.json" \
            --capture "$RUN/${SPLIT}_capture.json" --tokenizer-path "$LLAMA" \
            --output "$RUN/preflight_${SPLIT}_${SLURM_JOB_ID}.json"
    done
fi

export ROLE="$STAGE"
export PORT=${PORT:-$((20000 + SLURM_JOB_ID % 30000))}
export SERVER_CONFIG="$RUN/server_${STAGE}_${SLURM_JOB_ID}.json"
export VLLM_PY
API="http://127.0.0.1:$PORT/v1"
"$VLLM_PY" -B - "$PORT" <<'PY'
import socket,sys
with socket.socket() as s: s.bind(('127.0.0.1',int(sys.argv[1])))
PY
SERVER_PID=""
CLIENT_PID=""
cleanup() {
    trap '' TERM INT
    if [ -n "$CLIENT_PID" ] && kill -0 "$CLIENT_PID" 2>/dev/null; then
        kill -TERM "$CLIENT_PID" 2>/dev/null || true
        wait "$CLIENT_PID" || true
    fi
    if [ -n "$SERVER_PID" ]; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" || true
    fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
# Existing server launcher only; its #SBATCH lines are inert under bash.
bash experiments/linear_probe/serve.sbatch > "$RUN/server_${STAGE}_${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
READY=0
for i in $(seq 1 180); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo 'Server exited'; exit 3; fi
    if "$VLLM_PY" -B - "$API" <<'PY'
import sys
from urllib.request import urlopen
try:
    with urlopen(sys.argv[1]+'/models',timeout=2) as f: f.read()
except Exception: raise SystemExit(1)
PY
    then READY=1; break; fi
    sleep 5
done
if [ "$READY" != 1 ]; then echo 'Server readiness timeout'; exit 3; fi
run_client() {
    "$@" &
    CLIENT_PID=$!
    while kill -0 "$CLIENT_PID" 2>/dev/null; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo 'Server died'; exit 3; fi
        sleep 1
    done
    local code=0
    wait "$CLIENT_PID" || code=$?
    CLIENT_PID=""
    return "$code"
}
if [ "$STAGE" = teacher ]; then
    for SPLIT in ${SPLITS:-train validation evaluation}; do
        run_client "$HF_PY" -B -m experiments.linear_probe.cli label --data "$RUN/data.json" \
            --capture "$RUN/${SPLIT}_capture.json" --teacher-config "$SERVER_CONFIG" \
            --api-base "$API" --output "$RUN/${SPLIT}_labeled.json"
    done
else
    run_client "$VLLM_PY" -B -m experiments.linear_probe.pilot solver-run \
        --requests "$RUN/solver_requests.json" --server-config "$SERVER_CONFIG" \
        --tokenizer-path "$QWQ" --api-base "$API" --output "$RUN/solver.json"
fi
