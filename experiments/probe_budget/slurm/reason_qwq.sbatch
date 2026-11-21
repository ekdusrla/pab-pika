#!/bin/bash
#SBATCH --job-name=pb-qwq
#SBATCH --partition=batch_ugrad
#SBATCH --gres=gpu:normal:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time=24:00:00
#SBATCH --exclude=ariel-v8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Step 5: QwQ-32B reasoning (vLLM, 4x RTX A5000, TP=4), one method per job, resumable (--keep).
#   --export=ALL,DATASET=math|travelplanner,RUN=vanilla|pab|probe|joint
# PLAN_DIR: where the plan json files are (default: what plan_merge.sbatch writes).
#   To reuse the committed MATH-500 plans: PLAN_DIR=experiments/probe_budget/plans/math500
# Result: results/<model>_<dataset>_results_<postfix>.json (+ _posteval.json for TravelPlanner)
source experiments/probe_budget/slurm/env.sh
conda activate "$CONDA_VLLM_ENV"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c 'import dataset.math_dataset; from utils.qwen_math_parser import latex2sympy; assert str(latex2sympy(r"\frac{1}{2}"))=="1/2"; print("eval imports OK")'

# max-model-len: MATH prompt <= ~1.5k + max_tokens 8192 -> 12288.
#   TravelPlanner query+reference up to 12.5k QwQ tokens (mean 6.9k) + 8192 -> 24576.
# Postfixes are the ones used for the reported runs (MATH vanilla/pab/probe = v2, joint = v3).
case "${DATASET:?}" in
  math)          MAXLEN=12288; PLAN_DIR=${PLAN_DIR:-dataset/MATH-500}
                 case "${RUN:?}" in
                   vanilla) PLAN=;                     PF=qwq32b_v2 ;;
                   pab)     PLAN=plan_v2_pab.json;     PF=pab_qwq32b_v2 ;;
                   probe)   PLAN=plan_v2_probe.json;   PF=probe_qwq32b_v2 ;;
                   joint)   PLAN=plan_v3_joint.json;   PF=joint_qwq32b_v3 ;;
                   *) echo "unknown RUN=$RUN"; exit 1 ;;
                 esac ;;
  travelplanner) MAXLEN=24576; PLAN_DIR=${PLAN_DIR:-dataset/TravelPlanner}
                 python -c 'from dataset.TravelPlanner.eval import eval_detail; print("TP eval imports OK")'
                 case "${RUN:?}" in
                   vanilla) PLAN=;                     PF=tp_qwq32b_v3 ;;
                   pab|probe|joint) PLAN=plan_v3_${RUN}.json; PF=${RUN}_tp_qwq32b_v3 ;;
                   *) echo "unknown RUN=$RUN"; exit 1 ;;
                 esac ;;
  *) echo "unknown DATASET=$DATASET"; exit 1 ;;
esac
if [ -z "$PLAN" ]; then
    MODEL=vanilla; ARGS="--model vanilla --raw-dataset"
else
    MODEL=planned_local_weighted
    ARGS="--model planned_local_weighted --decay polynomial --data-path ${PLAN_DIR}/${PLAN}"
fi

PORT=$((20000 + SLURM_JOB_ID % 10000))
mkdir -p logs results
VLOG=logs/vllm_qwq_${DATASET}_${RUN}_${SLURM_JOB_ID}.log
# --max-num-seqs 32 + util 0.88: 164 concurrent seqs at util 0.92 OOM'd once.
vllm serve "$QWQ_PATH" --served-model-name "Qwen/QwQ-32B" --port $PORT \
    --dtype bfloat16 --tensor-parallel-size 4 \
    --max-model-len $MAXLEN --gpu-memory-utilization 0.88 --max-num-seqs 32 \
    --disable-log-requests > "$VLOG" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

READY=0
for i in $(seq 1 120); do
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then READY=1; echo "vLLM ready after ~$((i*10))s"; break; fi
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died:"; tail -40 "$VLOG"; exit 1; }
    sleep 10
done
[ $READY -eq 1 ] || { echo "vLLM not ready in 20 min"; tail -40 "$VLOG"; exit 1; }
grep -m1 "# GPU blocks" "$VLOG" || true

# load_dotenv does not override exported vars, so .env is left untouched
export API_KEY=dummy API_BASE="http://localhost:${PORT}/v1" MODEL_NAME="Qwen/QwQ-32B"
# Watchdog: vLLM 0.5.4 TP=4 can hang ("No available block found" -> "Engine iteration timed out"
# -> AsyncEngineDeadError); the server then answers 500 to everything and the job would still end
# COMPLETED with failed requests. If that happens: kill the client, fail the job (exit 3),
# resubmit the same command (finished ids are skipped).
run_guarded() {
    "$@" & local CPID=$!
    while kill -0 $CPID 2>/dev/null; do
        if grep -q "Engine iteration timed out\|AsyncEngineDeadError" "$VLOG"; then
            echo "vLLM ENGINE DEAD -> abort $(date -Is)"; grep -m3 "timed out\|No available block" "$VLOG"
            kill $CPID; exit 3
        fi
        sleep 20
    done
    wait $CPID
}
echo "[$DATASET/$RUN] reasoning start $(date -Is)"
run_guarded python -m run.run_inf --dataset "$DATASET" --num-workers 32 --queue-size 32 --keep $ARGS --postfix "$PF"
echo "[$DATASET/$RUN] reasoning DONE $(date -Is)"

if [ "$DATASET" = travelplanner ]; then
    # QwQ (same server) converts each plan to JSON (official repo used GPT-4), then the
    # TravelPlanner sandbox checker. Not counted in method latency/tokens.
    echo "[$DATASET/$RUN] post-eval start $(date -Is)"
    run_guarded python -m run.run_eval --dataset travelplanner --model $MODEL --postfix "$PF" --concurrency 32
    echo "[$DATASET/$RUN] post-eval DONE $(date -Is)"
fi
