# Validation and predeclared Evaluation recovery

No root Python file, recovery_v1 file, Baseline, Teacher, Solver or result has been
modified. This adapter uses the same existing PAB RECOVERY_INSTRUCTION as the
successful Train repairs, retaining the original 2–5-step parser contract.
No SLURM job, Python, test or inference was executed on the master node.

## Recorded Validation failure

Job 219632 FAILED/1:0 on moana-y7. Its log confirms recovered Train 24/24 and
preservation of the original 22 rows/NPZs before Validation. Validation completed
five rows, failed intermediate_algebra/train/1000, then raised the final incomplete
capture exception. Shell set -e prevented Evaluation from starting.

The failed attempt is ValueError / parse_failure / phase step_alignment:
`Expected 2-5 subquestions, got 6`. Prompt 681 tokens; completion 305, normal EOS
(128009), finish_reason stop; generation 8.39584 s; generation+feature wall 8.54280 s;
wall including storage 8.54722 s. Allocated peak 16624961536 bytes (~15.48 GiB),
reserved peak 17112760320 bytes (15.9375 GiB); no OOM or 21 GiB crossing. Extra
forward passes 0. Shape/alignment/finite feature validation was not reached because
parse_steps rejected the step count. Last normal Validation row is prealgebra/train/172;
safety_stop is null. These observations come from narrow checkpoint/log reads;
full output decoding and new checksums belong to the compute-node audit below.

## User-run CPU audit

```bash
cd /data/dada0228/plan-budget/P-and-B
AUDIT_JOB=$(sbatch --parsable --gres=none --cpus-per-task=1 \
  experiments/linear_probe/recovery_v2/audit.sbatch)
```

Requests CPU **1**, RAM 4 GiB, 20 minutes, no GPU. Existing 46 tests plus six new
tests and Python/shell static checks are configured, not yet executed. The audit
loads the tokenizer only; never model weights. Require COMPLETED/0:0, all tests
passing and `AUDIT V2 PASS` in runs/audit_v2_JOBID.log. The sealed report is
runs/pilot_v1/audit_v2_JOBID.json. It verifies source/checkpoint seals and hashes,
original audit_219630, the original 22 reused rows/NPZs, repaired Train 24, Validation
five NPZs, all 33 layers, dtype, shape, finite values, fingerprints, exact token
alignment, original or approved repair messages, source/model/tokenizer provenance,
and orphan/temp files. It decodes the failed Validation draft and requires a
sequential 1..6 list reproducing the exact original exception. Any mismatch blocks
GPU work. Data/split identities are checked against the existing frozen manifest;
no split, benchmark or dataset selection changes are made.

## User-run GPU resume, only after the CPU gate

Recheck candidates immediately before submission:

```bash
scontrol show node moana-y4,moana-y7 -o
```

Preparation-time metadata: y4 is IDLE+DRAIN; y7 MIXED with two of three GPUs
allocated. Each has 96 CPUs and 428743 MiB RAM. CPU12 and 40 GiB are within each
candidate's per-GPU ceiling. The exclusion list retains the prior validated y4/y7
candidates without pinning a single node. gpu:normal is not a GPU model guarantee;
exactly one visible RTX 3090 is checked before model loading.

```bash
sbatch --parsable \
  --account=ugrad_ce --qos=qos_dada0228_2026_2 --partition=batch_ce_ugrad \
  --nodes=1 --ntasks=1 --gres=gpu:normal:1 --cpus-per-task=12 --mem=40G \
  --time=02:00:00 \
  --exclude='moana-r[1-5],moana-u[1-4,8],moana-y[1-3,5-6]' \
  --dependency=afterok:"$AUDIT_JOB" \
  --export=ALL,AUDIT_JOB="$AUDIT_JOB",CAPTURE_RESUME=1,ACK_SAFETY_STOP=0 \
  experiments/linear_probe/recovery_v2/resume.sbatch
```

Stages:

1. Repeat full CPU integrity preflight on the allocated node before weights.
2. Create validation_recovered.json, copying the six existing rows/histories and
   referencing all five successful NPZs unchanged. Only the known failed row gets
   the original system/user prompt + failed assistant draft + unchanged PAB repair
   user instruction. Store actual messages, fingerprints, original draft and all
   extra calls/tokens/times/memory. A completed invalid repair stops without retry.
3. Freeze evaluation_policy_v2.json **before any native Evaluation generation**.
   Run the unchanged CLI collection for Evaluation five problems + no-hidden-state
   controls. Preserve evaluation_capture.json as the native first-attempt artifact.
   Into evaluation_recovered.json, copy all first-attempt successes unchanged and
   apply the predeclared one-repair rule only to eligible failed rows. A repair
   includes a matched no-hidden control with identical messages/settings/seed.
   If there are no failures, no model loads for repair. A failed control can be the
   initial parse failure; actual generation accounting avoids double-counting it.
   If that control failed before a hidden capture was attempted, the first hidden
   capture cost is unobserved: do not impute it or treat the failed control as two
   calls. Final successful repair control/capture costs remain separately measured.
4. Full final audit of all 35 problem features plus classification/cost ledgers.

Models remain pinned Llama-3.1-8B-Instruct, FP16, eager, hidden indices 0..32,
temperature .1/top_p .9/seed42, max_new_tokens1024/max_context8192. The 21 GiB guard
remains at completed-operation boundaries; crossing it stops subsequent processing.
It cannot interrupt the middle of a CUDA allocation. 1024-token worst-case memory
has not been established. Expect roughly 5–30 minutes; the requested limit is 2 h.
Processes exit between stages so models are not resident concurrently.

The frozen Evaluation rule is independent of labels, answers and performance:
normal EOS, sequential numbered list >5, reproduced ValueError at parser alignment,
no memory violation, then **one** unchanged repair. All other failures stop for
inspection. Never tune the repair prompt, seed or attempt limit after looking at
Evaluation outcomes. Policy, code and audit fingerprints are enforced; edits after
the first Evaluation execution block continuation. A fresh CPU audit does not
silently authorize changing an already-declared Evaluation policy.

## Research reporting

For every problem retain initial_success, repaired_success, repair_failed (and
pending/unrepaired failures for incomplete runs). These appear in the new
checkpoint recovery_v2.problem_outcomes and the final ledger. Train histories and
features remain immutable; its categories are derived in the separate final report.

Initial success is the original single-generation decomposition path. Repair adds
the failed draft and explicit user turn; it is a separate robustness policy and
additional Planner calls, not a replacement Credit Assessment call. Report:

- Initial-pass success rate over the full original problem denominator.
- Repair attempt/success rate and final coverage, keeping failed problems visible.
- Credit/Budget metrics separately for initial_success and repaired_success, plus
  an explicitly labelled aggregate on final valid decompositions. Do not invent
  labels/features or silently drop failures from coverage/efficiency statistics.
- Teacher and Probe use exactly the same final steps/fingerprint for each problem.
  Layer selection uses Validation only; Evaluation never changes the policy.
- End-to-end latency/tokens include failed initial generations, repair calls, model
  load and storage separately. Control runs are instrumentation costs and must be
  reported separately from deployed-path costs. Compare second-call savings on
  matched final decompositions and report total Planner overhead as well.

Original Plan-and-Budget's successful first-call prompt is unchanged; the repair
protocol supplies extra dialogue. Existing Train repairs used the same instruction
under recovery_v1. Do not claim repaired problems validate a pure unchanged
one-call decomposition experiment or attribute the added repair cost to the Probe.

## Safety and output files

CAPTURE_RESUME=1 explicitly allows new recovered checkpoints to resume existing
successful rows; missing recovered outputs are created exclusively. ACK_SAFETY_STOP=0
is correct now. For a new recovered-output memory/OOM stop, inspect diagnostics first;
only after explicit acknowledgement may ACK=1 add --ack-safety-stop. It does not
relax 21 GiB. Completed invalid repairs/storage failures and native Evaluation safety
stops are not automatically retried. Native Evaluation with partial rows plus errors
also stops rather than retrying failed prompts blindly. Preserve .tmp/orphan files
and investigate them; no automatic discard or overwrite is implemented.

Original train_capture.json, train_recovered.json and validation_capture.json are
unchanged. Final effective inputs will be train_recovered.json (24),
validation_recovered.json (6), evaluation_recovered.json (5 with controls). Later
Teacher/Probe commands must explicitly use those paths, not the original failed
Validation file. This script runs no Teacher generation, Probe fit or Solver.

Under experiments/linear_probe/runs/ check audit_v2_CPUJOB.log,
recovery_v2_GPUJOB.log and pilot_v1/{audit_v2_CPUJOB.json,preflight_v2_GPUJOB.json,
validation_recovered.json,evaluation_capture.json,evaluation_recovered.json,
evaluation_policy_v2.json,final_v2_GPUJOB.json} plus corresponding .features
directories. Require FINAL V2 PASS, final passed=true, 24/6/5 successful IDs,
safety_stop=null, finite features and correct fingerprints. Never overwrite old
audit reports or completed artifacts to bypass a gate.
