# Recovered-capture Teacher Credit and Linear Probe

All new source is in experiments/linear_probe/credit_v1; all generated results/logs
are in experiments/linear_probe/runs/credit_v1 (already created for SLURM logs).
No Baseline, root experiment Python, Recovery V1/V2 code, recovered checkpoint or
existing NPZ is modified. No Decomposition generation is performed here.

## Frozen inputs and contracts

Inputs under runs/pilot_v1: data.json, train_recovered.json (24),
validation_recovered.json (6), evaluation_recovered.json (5), final_v2_219684.json.
NPZs remain at their original paths. New labeled JSONs deep-copy records and retain
the same final steps/fingerprint/feature references. SHA256 and a projection of
all original capture fields are checked before and after Teacher processing.
The CPU audit also repeats all feature/identity checks, confirms the final PASS,
checks MATH-500 normalized-hash exclusion and cross-split/benchmark disjointness,
and freezes source/model-metadata/tokenizer hashes and the model-selection grid.
Normalized hashes do not guarantee absence of semantic duplicates.

Teacher requests directly reuse PAB assessment_messages, credit_schema,
parse_credits and run_stage. Settings: pinned Llama-3.1-8B-Instruct,
FP16, TP=1, unquantized, max_model_len8192, max_num_seqs1, GPU utilization .85,
temperature .1, top_p .9, seed42, output cap2048, guided_json/outlines.
The existing serve.sbatch is invoked under bash; its batch resource headers are
inert, and the new outer job supplies 12 CPU/40 GiB/one GPU/two hours.
No enforce-eager or other server inference option is added to the Baseline server.

HF captures previously generated Decomposition and Hidden States without guided
decoding. There is no HF Credit-guided implementation introduced here: Teacher
uses the same vLLM 0.6.6/Outlines path as PAB. The guided schema constrains JSON
structure, not all Credit semantics; positivity and sum=100 remain checked by the
original parser. Requests are compared with the original literal PAB template,
including its original example formatting. The external benchmark questions differ
from legacy evaluation-derived examples deliberately, to prevent data leakage.
Therefore sampling/backend/prompt template match; benchmark contents are not a
claim of byte-identical legacy benchmark prompts or outputs.

HF tokenizer context estimates and actual vLLM prompt tokens are both recorded.
Differences are visible as tokenization_comparison.vllm_minus_hf, not silently
normalized away. A nonzero difference requires review for efficiency comparisons;
actual context must still fit. Seed matching is not a cross-engine bitwise guarantee.

Teacher credits are positive integer percentage points summing to 100, one per
stored final step. Ridge fits these raw target points; StandardScaler applies only
to input features and is fitted on Train only. Predictions use the EXISTING Probe
projection: clamp negatives, reserve one point per step, distribute remaining
integer points by largest remainders, deterministic index ties. Its all-zero-weight
uniform projection is preserved and flagged; Teacher failures never get uniform
labels. This projection is Probe-specific, not a rule claimed to come from PAB.

Budget uses the unchanged PAB allocate_words and LEVEL_WORDS (200/250/350/450/600),
constant schedule. In particular, PAB rounds down and gives leftovers to the largest
weights, not to the largest fractional remainders. This difference from Credit
projection is intentional and tested. Budgets are words in the existing Solver
prompt contract, not a new token cap.

## User execution: audit, Smoke, Teacher, CPU training

Never run these Python modules on the master. All commands below submit jobs by
the user; Codex has not submitted or run them. New source edits require a fresh
credit CPU audit. Recovery audit 219678 cannot substitute for this new audit.

```bash
cd /data/dada0228/plan-budget/P-and-B
CREDIT_AUDIT=$(sbatch --parsable --gres=none --cpus-per-task=1 \
  experiments/linear_probe/credit_v1/audit.sbatch)
```

Require COMPLETED/0:0, all CPU tests passing and CREDIT CPU AUDIT PASS. This job
runs Python AST compilation, shell syntax checks, existing tests plus new tests,
tokenizer-only Credit context checks, and NPZ audits. It never loads Llama weights.
The report is runs/credit_v1/audit_CPUJOB.json. Each prompt+2048 must fit8192.

Recheck node allocation before every GPU submission:

```bash
scontrol show node moana-y2 -d
```

y2 is the default recently validated node; the runtime still checks allocated GPU
name/UUID/capacity/one visible device, CPU/RAM policy and >=21 GiB free memory.
SLURM CUDA_VISIBLE_DEVICES is never changed. UUID inventory and other compute PIDs
are recorded; only this job's own vLLM server/client is stopped on failure.
At the final preparation check on Nov16, all eight y2 GPUs were allocated; GPU jobs
may queue. Node status is not a free-VRAM measurement and must be rechecked before
submission. No fallback to y5 or manual device override is performed.

```bash
TEACHER_SMOKE=$(sbatch --parsable --dependency=afterok:"$CREDIT_AUDIT" \
  --export=ALL,AUDIT_JOB="$CREDIT_AUDIT",MODE=smoke,RESUME=0 \
  experiments/linear_probe/credit_v1/gpu.sbatch)
```

Require TEACHER SMOKE FINAL PASS and complete_smoke_GPUJOB.json passed=true.
The Smoke selects the longest actual Train/Validation Credit prompt, uses its
unchanged final steps and the real cap2048/schema/Teacher request. It makes an
extra calibration Teacher call, not a Decomposition call and not a reused training
label. Its response, actual tokens and latency are saved separately in smoke_JOB.json.
A short completion does NOT establish the 2048-token worst case. CUDA graph warmup,
Outlines compilation, first-use schemas and KV allocation can fail at startup;
no full label job is allowed before this gate.

```bash
TEACHER_JOB=$(sbatch --parsable --dependency=afterok:"$TEACHER_SMOKE" \
  --export=ALL,AUDIT_JOB="$CREDIT_AUDIT",SMOKE_JOB="$TEACHER_SMOKE",MODE=teacher,RESUME=0 \
  experiments/linear_probe/credit_v1/gpu.sbatch)
```

Require Train24 and Validation6 aligned valid labels plus TEACHER LABELS FINAL
PASS, and complete_teacher_JOB.json
passed=true. Actual requests use recovered inputs; the old pilot_gpu.sbatch hardcoded
*_capture.json paths are not used. Original feature data is never rewritten.

```bash
TRAIN_JOB=$(sbatch --parsable --gres=none --cpus-per-task=1 \
  --dependency=afterok:"$TEACHER_JOB" \
  --export=ALL,AUDIT_JOB="$CREDIT_AUDIT",TEACHER_JOB="$TEACHER_JOB",MODE=train \
  experiments/linear_probe/credit_v1/train.sbatch)
```

Require PROBE TRAIN PASS and probe_v1/status.json passed=true. All 33 layers and
alpha [.01,.1,1,10,100,1000] are frozen before Evaluation (198 candidates).
Each candidate fits scaler+Ridge only on Train24 and is scored on Validation6 by
equal-problem mean projected Credit MAE. Ties: lowest layer, then lowest alpha.
There is no refit on Validation. Six validation problems are a small selection
set: do not report their tuned scores as independent generalization performance.
The model, metadata, complete grid scores and validation report are immutable
under probe_v1. Existing/partial artifact directories are refused, not overwritten.

## Independent Evaluation, only after model selection

Current Teacher mode labels Train/Validation only. Evaluation mode checks the
frozen probe metadata and successful training status before starting its server.
Evaluation labels never enter StandardScaler/Ridge fitting or alpha/layer choice.

```bash
EVAL_TEACHER_JOB=$(sbatch --parsable --dependency=afterok:"$TRAIN_JOB" \
  --export=ALL,AUDIT_JOB="$CREDIT_AUDIT",SMOKE_JOB="$TEACHER_SMOKE",MODE=evaluation,RESUME=0 \
  experiments/linear_probe/credit_v1/gpu.sbatch)

EVAL_JOB=$(sbatch --parsable --gres=none --cpus-per-task=1 \
  --dependency=afterok:"$EVAL_TEACHER_JOB" \
  --export=ALL,AUDIT_JOB="$CREDIT_AUDIT",EVALUATION_JOB="$EVAL_TEACHER_JOB",MODE=evaluate \
  experiments/linear_probe/credit_v1/train.sbatch)
```

Evaluation produces step differences, per-problem/micro MAE/RMSE, average-tie Credit
ranks, Spearman (null for constant vectors), and exact Teacher/Probe Budget
differences. No Solver requests run. Keep initial-success/repaired-success groups
and the original failed/repair generation costs from final_v2_219684.json alongside
these reports. Do not attribute repair costs or calibration calls to Probe prediction.

## Safety, costs and failure handling

GPU launch guard checks free memory, CUDA initialization and tiny allocation in a
separate process that exits before weights. A monitor starts during server startup,
polling assigned-device nvidia-smi usage every .2s. Usage>21 GiB or measurement error
stops ONLY this job's server; server/client death fails the job. memory_JOB.json
records peak OBSERVED whole-device usage, sample count, errors and duration.
It is NOT exact torch allocated/reserved peak and can miss sub-sample transients.
Server logs/OOM detection remain necessary. The unchanged .85 vLLM pool targets
roughly 20–21 GiB on a 3090; actual startup/request peaks remain unverified until
the user runs Smoke. Do not silently lower context/utilization or relax the guard.

run_stage records the raw response before semantic validation, every request,
failure, retry, usage and wall latency. Only transient API failures retry (at most
two retries); invalid semantic responses stop, with no automatic changed prompt
or uniform Credit fallback. Unknown interrupted/API costs are flagged by missing
attempt counts. Successful rows resume without a new API call when RESUME=1;
each restart preserves its server launch metadata in a new session. Original
checkpoints/NPZs and completed response histories remain unchanged.

Use RESUME=1 only after reviewing a partial Teacher checkpoint. Completed invalid
responses require investigation and are explicitly blocked from unchanged retries.
Uncommitted .tmp files and partial artifacts are never silently discarded. A new
server job can resume with the same scientific config: job ID, port and launch
metadata are recorded per-session rather than changing the checkpoint contract.
If a CPU fit failed, inspect probe_v1/status.json; never overwrite its directory.

## Resources, gates, files

| Stage | GPU | CPU | RAM | Limit | Expected duration |
|---|---|---|---|---|---|
| CPU audit/tests/tokenizer | none | 1 | 8 GiB | 30 min | 1–10 min |
| Teacher Smoke | RTX3090 x1 | 12 | 40 GiB | 2 h | 2–15 min |
| Teacher Train/Validation | RTX3090 x1 | 12 | 40 GiB | 2 h | 5–40 min |
| Probe fit/Validation | none | 1 | 8 GiB | 30 min | 1–20 min |
| Teacher held-out5 | RTX3090 x1 | 12 | 40 GiB | 2 h | 2–15 min |
| CPU held-out report | none | 1 | 8 GiB | 30 min | <5 min |

These are planning estimates, not GPU measurements. For y2's known 96 CPU/8 GPU
and428743 MiB configuration, CPU12 is the per-GPU ceiling and RAM40 is below the
headroom-adjusted49 GiB ceiling. Runtime allocation() rechecks policy. CPU-only
jobs request one CPU and single-thread BLAS; server and client share one allocation.
No collection job runs concurrently on the same allocation.

All stages need SLURM COMPLETED/0:0 plus their sealed passed gate, not only a log
message. New numeric metrics have no predeclared quality threshold: pipeline PASS
means complete/frozen/valid, not that the Probe outperforms Teacher.

Under runs/credit_v1: audit_JOB.log/json; gpu_JOB.log; launch_JOB.json;
server_MODE_JOB.log/json; memory_JOB.json; smoke_JOB.json; complete_MODE_JOB.json;
train_labeled.json; validation_labeled.json; later evaluation_labeled.json;
probe_v1/{status.json,metadata.json,probe.joblib,validation_report.json};
evaluation_report_CPUJOB.json. Failure implies no dependent scientific stage.

Checks performed by Codex: bash -n on the three new batch scripts passed. Python
compilation/tests, actual CPU research-data audit, model loading, GPU memory Smoke,
label generation and fitting have NOT run. The CPU audit is the first required gate.
