# RTX A2000 12 GiB x4 / TP=4 Teacher

This is a separate execution namespace. Original credit_v1, its PASS audit220305,
Recovery V2, all recovered JSON/NPZ files, prompts, splits and Baseline scripts
remain unchanged. New outputs: experiments/linear_probe/runs/credit_a2000_tp4.
No Python, CUDA test, inference or SLURM submission was executed by Codex.

## Resource observations (read-only SLURM and existing logs)

| Node | Configured GPU | Used/free GRES at observation | CPU | RAM MiB | CPU/GPU | Conservative RAM/GPU |
|---|---|---|---|---|---|---|
| u1 | normal:8 | 0/8 | 48 | 244751 | 6 | 28 GiB |
| u2 | normal:8 | latest 0/8, IDLE | 48 | 122174 | 6 | 14 GiB |
| u3 | normal:8 | 0/8 | 48 | 122174 | 6 | 14 GiB |
| u4 | normal:8 | 1/7 | 48 | 122174 | 6 | 14 GiB |
| u8 | normal:8 | 8/0, ALLOCATED | 32 | 152789 | 4 | 17 GiB |

These are changing scheduler snapshots, not a live VRAM-free test. u2 changed from
five unallocated GPUs to eight during preparation. Current own queued/running jobs
observed were CPU-only; requesting four stays below the stated five-GPU allowance,
but scheduling/account limits and availability must be rechecked before submission.

Existing logs/check_cuda_y5_152789.log (despite its filename, hostname moana-u2)
records four NVIDIA RTX A2000 12GB cards and successful CUDA allocations. This is
historical hardware evidence, not a new test. Other u nodes expose only gpu:normal
in SLURM, so their actual GPU models/VRAM have not been independently confirmed.
The new GPU job pins **u2**, one node, four GPUs, 24 CPU, 48 GiB host RAM, two hours.
This is 6 CPU and12 GiB host RAM per GPU, within u2's6/14 policy ceilings. It does
not reserve a fifth GPU or an entire node. u8 is not a candidate for this CPU policy.
The runtime rejects anything except four unique RTX A2000 12 GiB-class cards.

## Audit reuse and required supplemental audit

credit_v1 audit220305 is COMPLETED/0:0, 58 tests and CREDIT CPU AUDIT PASS. Do not
resubmit credit_v1/audit.sbatch. The new supplemental audit calls its approval
function only to verify seals and unchanged input/code hashes. It does not rerun
the 58 tests, feature decoding, full data audit or tokenizer context computation.

TP1's Teacher signature and3090 memory gate cannot attest TP4 or A2000. Therefore
a NEW CPU-only supplemental audit is required. It inherits the unchanged data,
contexts, steps/fingerprints and selection grid, binds parent audit fingerprint,
and signs a separate TP4 policy/code fingerprint. Five new synthetic CPU tests,
Python AST (including shell Python heredocs), shell checks, installed vLLM version,
CLI/backend availability and Llama partition dimensions are checked. It requests
CPU1, RAM8 GiB, 20 minutes, no GPU. It does not load weights or fit real data.

Installed vLLM0.6.6.post1 exposes --tensor-parallel-size, --distributed-executor-backend
mp and --guided-decoding-backend outlines. Local Llama code shards32 attention and
8 KV heads into8 and2 per rank, respectively;4096 hidden,14336 intermediate and
128256 vocabulary dimensions also divide by4. Static compatibility checks PASS in
the audit does NOT mean NCCL, P2P, kernels or guided generation have been tested.
Only the user's real four-rank Smoke can establish combined runtime compatibility.

## Scientific settings and GPU safety policy

Model/revision remain meta-llama/Llama-3.1-8B-Instruct /
0e9e39f249a16976918f6564b8830bc894c89659. FP16, no quantization,
context8192, max_num_seqs1, gpu_memory_utilization .85, temperature .1,
top_p .9, seed42, cap2048, exact PAB Assessment prompt/benchmarks/guided_json/outlines
are preserved. No Decomposition, Solver or HF Hidden State collection is run.
New serve.sbatch uses same-node multiprocessing with TP4. Train, Validation and
Evaluation all use this same server profile and signatures; TP1 labels cannot be
mixed or resumed here. New Probe training/evaluation paths are included for this
namespace; their algorithm is unchanged from credit_v1.

FP16 weights alone are roughly15 GiB total (~3.75 GiB/rank); KV/cache, activations,
CUDA graphs and NCCL buffers also require memory. Aggregate48 GiB VRAM is not a
proof of per-rank fit. This profile deliberately supports A2000 **12 GiB**, not6 GiB.
Before weights, each assigned GPU must have >=10 GiB free, a unique UUID, successful
CUDA initialization and tiny allocation. Logical IDs0..3 always refer to SLURM's
allocation; CUDA_VISIBLE_DEVICES/physical GPU numbers are never overridden.
P2P access is recorded, not forced; actual NCCL behavior is tested by real vLLM.

A2000 cannot satisfy3090's21 GiB-per-device gate. The new, separately signed policy
uses a maximum **11 GiB observed whole-device usage on EACH GPU**. The monitor
queries all four assigned UUIDs during startup, request execution and shutdown.
It stops only its own server if any card exceeds the limit or measurement fails.
The .85 pool targets roughly10 GiB/card; actual peak remains unmeasured until Smoke.
Polling is not exact torch allocated/reserved peak and can miss transients. Do not
silently change FP16, context, backend, pool fraction or threshold on failure.

## User commands

From project root; run each next stage only after COMPLETED/0:0 and its sealed
passed=true gate. All output directories already exist for SLURM log creation.

```bash
TP4_AUDIT=$(sbatch --parsable --gres=none --cpus-per-task=1 \
  experiments/linear_probe/credit_a2000_tp4/audit.sbatch)
```

Require A2000 TP4 SUPPLEMENTAL AUDIT PASS, audit_NEWJOB.json passed=true.
The AUDIT_JOB below is this NEW ID, not220305. Parent220305 is reused internally.

```bash
scontrol show node moana-u2 -d

TP4_SMOKE=$(sbatch --parsable \
  --nodes=1 --ntasks=1 --nodelist=moana-u2 \
  --gres=gpu:normal:4 --cpus-per-task=24 --mem=48G --time=02:00:00 \
  --dependency=afterok:"$TP4_AUDIT" \
  --export=ALL,AUDIT_JOB="$TP4_AUDIT",MODE=smoke,RESUME=0 \
  experiments/linear_probe/credit_a2000_tp4/gpu.sbatch)
```

Require four CUDA-valid unique12 GiB GPUs, real TP4 server readiness, actual longest
Train/Validation Credit request with cap2048/guided JSON, positive integer Credits
summing to100, memory gate on all four cards and TEACHER SMOKE FINAL PASS.
complete_smoke_JOB.json must be passed=true. A short actual completion does not
verify a2048-token worst case. The Smoke call is additional calibration cost, not
a Decomposition call or a label reused for training. Record generated tokens.

```bash
TP4_TEACHER=$(sbatch --parsable \
  --nodes=1 --ntasks=1 --nodelist=moana-u2 \
  --gres=gpu:normal:4 --cpus-per-task=24 --mem=48G --time=02:00:00 \
  --dependency=afterok:"$TP4_SMOKE" \
  --export=ALL,AUDIT_JOB="$TP4_AUDIT",SMOKE_JOB="$TP4_SMOKE",MODE=teacher,RESUME=0 \
  experiments/linear_probe/credit_a2000_tp4/gpu.sbatch)
```

Require24 Train and6 Validation aligned valid labels, all four memory gates,
TEACHER LABELS FINAL PASS and complete_teacher_JOB.json passed=true. Costs,
failures/retries and actual request/response tokens/latencies are retained.
Only transient failures retry; invalid semantics stop without uniform labels.
Use RESUME=1 only after inspecting partial outputs; successful labels are skipped.
Original captures/NPZs remain immutable and are rechecked by the parent approval.

Optional next CPU Probe stage (same198-candidate selection as credit_v1):

```bash
TP4_TRAIN=$(sbatch --parsable --gres=none --cpus-per-task=1 \
  --dependency=afterok:"$TP4_TEACHER" \
  --export=ALL,AUDIT_JOB="$TP4_AUDIT",TEACHER_JOB="$TP4_TEACHER",MODE=train \
  experiments/linear_probe/credit_a2000_tp4/train.sbatch)
```

After frozen Train-only fit/Validation selection, Evaluation Teacher uses the SAME
gpu.sbatch with MODE=evaluation, sameTP4_AUDIT/TP4_SMOKE and resources, dependency
afterok:TP4_TRAIN. Then CPU train.sbatch MODE=evaluate with EVALUATION_JOB pointing
to that successful Evaluation Teacher job. Evaluation never selects layer/alpha.
All output paths are under runs/credit_a2000_tp4; no TP1 output is read as a label.

## What changes in interpretation

Same model, prompts and sampling do not guarantee identical Credits across TP1
andTP4 because reductions/kernel paths and hardware differ. Use this TP4 Teacher
for all labels; do not merge TP1 andTP4 train/validation/evaluation. Baseline versus
Probe timing must be hardware-matched; A2000x4 latency is not a fair direct
replacement for3090x1 latency when claiming Probe speedup. Keep4-GPU-seconds,
startup, Smoke, API retries and earlier Decomposition repair costs separate.

## Status/results

Codex performed read-only SLURM/config/source/log inspection and bash syntax checks.
The four new shell scripts passed bash -n. Five new CPU tests, supplemental audit,
four-GPU CUDA/NCCL test, model load and guided JSON generation are NOT run.

Inspect under runs/credit_a2000_tp4: audit_JOB.log/json, gpu_JOB.log,
launch_JOB.json (UUIDs/mapping/P2P/free memory per GPU), memory_JOB.json
(peak_per_gpu_used_bytes), server_MODE_JOB.log/json, smoke_JOB.json,
complete_MODE_JOB.json, train_labeled.json, validation_labeled.json, later
evaluation_labeled.json and probe_v1 artifacts. Any missing/failed gate is FAIL or
incomplete, never permission to relax settings or modify old results.
