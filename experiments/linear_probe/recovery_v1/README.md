# Frozen Train checkpoint recovery

This directory is isolated from the root capture code: existing runtime source hashes
and the original parser remain unchanged. No jobs have been submitted and no tests
have been executed on the master node.

## Observed failure

Job 219625 ended FAILED/1:0 after recording 22 successes and two parse failures.
`precalculus/train/511` produced 225 completion tokens and the parser counted seven
steps; `counting_and_probability/train/60` produced 246 tokens and six steps.
Both ended in EOS, with reserved peak 17343447040 bytes (16.15 GiB), below 21 GiB.
The failure occurs before feature alignment/finite validation, not because either
check rejected an array. The checkpoint has safety_stop=null and last successful
problem `prealgebra/train/806`. The final incomplete-Train exception and shell
`set -e` prevented Validation/Evaluation from starting.

These are the recorded parser counts, not a claim that the full drafts have been
decoded on the master node. The compute-node audit decodes all recorded IDs,
requires consecutive 1..7/1..6 headers and reproduces the exact parser exception.
Any discrepancy blocks recovery.

## CPU gate (user submits)

From the project root:

```bash
AUDIT_JOB=$(sbatch --parsable --gres=none experiments/linear_probe/recovery_v1/audit.sbatch)
```

The audit job (2 CPU, 4 GiB, 20 minutes) runs syntax checks, the existing CPU test
modules and new recovery tests. It never loads model weights. It then verifies:

- Original checkpoint seal, data identity, source/model/tokenizer hashes.
- Exactly 22 successful rows and the two expected normal-EOS parser failures.
- Every referenced NPZ checksum, all 33 layers, (steps,4096) float32 arrays,
  finite values, step fingerprints, original prompts and token alignment.
- No orphan/partial files in the original feature directory and no .tmp checkpoint.
- Reserved memory <=21 GiB and no original safety stop.

Read `experiments/linear_probe/runs/audit_JOBID.log`; require COMPLETED/0:0 and
`AUDIT PASS`. The sealed report is `runs/pilot_v1/audit_JOBID.json` and a byte-for-byte
source snapshot is `audit_JOBID.json.source_snapshot.json`. Failed audits save
diagnostics and block GPU recovery. Current hashes cannot prove the absence of
historical overwrites without an earlier independent snapshot.

## Explicit GPU recovery (user submits only after the CPU gate)

Keep AUDIT_JOB from above, or set it explicitly to the successful CPU audit job ID.

```bash
sbatch --parsable \
  --account=ugrad_ce --qos=qos_dada0228_2026_2 --partition=batch_ce_ugrad \
  --nodes=1 --ntasks=1 --gres=gpu:normal:1 --cpus-per-task=12 --mem=40G \
  --time=02:00:00 \
  --exclude='moana-r[1-5],moana-u[1-4,8],moana-y[1-3,5-6]' \
  --dependency=afterok:"$AUDIT_JOB" \
  --export=ALL,AUDIT_JOB="$AUDIT_JOB",CAPTURE_RESUME=1,ACK_SAFETY_STOP=0 \
  experiments/linear_probe/recovery_v1/resume.sbatch
```

The GPU job repeats the integrity audit before loading weights and requires exactly
the audited recovery code and unchanged source. One visible RTX 3090 is checked
before loading the pinned Llama-3.1-8B-Instruct FP16/eager model. All 33 layers,
seed 42, temperature .1, top_p .9, max_new_tokens 1024 and the 21 GiB guard remain.

The two failed rows are repaired into **train_recovered.json**, never the original
train_capture.json. It deep-copies the original histories and successful records;
the 22 successful NPZ files are referenced unchanged, not regenerated or copied.
The original system/user prompt is retained, followed by the recorded failed
assistant draft and the existing PAB RECOVERY_INSTRUCTION user turn. This is an
explicit extra dialogue/generation protocol for these two rows; it is not evidence
of successful unchanged one-call decomposition. Record both failed and repair costs
in later overhead comparisons. No step truncation or 2..5-contract relaxation occurs.
The new output records policy, actual messages, draft/request fingerprints, all
attempts, token/timing costs and memory. Repair features are generated from the
revised text, not relabeled old draft features. A completed invalid repair or storage
failure blocks unchanged retries and requires investigation.

Only after repaired Train validates 24/24 does the script collect Validation 6,
then Evaluation 5 with five no-hidden-state controls. There are at most 18 new
generations if all succeed (2 repairs + 6 + 5 + 5 controls). Expect roughly 5–30
minutes, depending on generation lengths/loading; request a two-hour limit.
The original train_capture.json intentionally remains 22 success / 2 error.
Future Teacher and Probe stages must explicitly use **train_recovered.json**;
the old default Train path must not be used. No Teacher/Probe/Solver stage is run.

## Stops and resumption

CAPTURE_RESUME=1 means existing new checkpoints require --resume; absent new
checkpoints are created exclusively. Successful rows are skipped. ACK_SAFETY_STOP=0
is correct for the current original safety_stop=null. A later OOM/reserved>21 GiB
stops subsequent problems/stages and saves diagnostics. Do not set ACK=1 until the
new stop has been inspected and explicitly approved; then resubmit the same command
with ACK_SAFETY_STOP=1 (adds --resume-after-safety-stop only for existing outputs).
The guard detects overshoot at completed operation boundaries, not midway through
a CUDA allocation. A 1024-token worst case remains unverified.

Interrupted/storage .tmp or orphan NPZ files are never automatically discarded.
Inspect and preserve them before any further action. A later Validation/Evaluation
parse error also stops the pipeline; this adapter repairs only the two audited Train
failures. Editing recovery Python/sbatch files requires a fresh CPU audit. Existing
recovered checkpoints also enforce their original recovery configuration.

Final artifacts: original train_capture.json/.features (unchanged),
train_recovered.json/.features, validation_capture.json/.features,
evaluation_capture.json/.features; logs `runs/recovery_JOBID.log` and fresh audit
`train_recovered.json.preflight_JOBID.json` plus its source snapshot.
Check counts 24/6/5, finite features, fingerprints, evaluation controls, all memory
peaks and safety_stop=null before proceeding beyond capture.

At preparation time y4 is DRAIN; y7 is MIXED with two of three GPUs allocated.
Availability must be rechecked immediately before submission. Both have 96 CPUs
and RealMemory 428743 MiB; CPU12/RAM40 GiB per GPU is below each policy ceiling.
`gpu:normal` is not a model-specific type, so the runtime RTX 3090 check is mandatory.
