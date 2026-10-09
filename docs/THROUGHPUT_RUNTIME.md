# Execution-only throughput upgrade

The user's latest instruction authorizes all available GPUs to finish the same
repaired-method run faster. This supersedes the four-GPU allocation restriction;
it does not authorize baseline retraining, extra seeds, variants or final tests.

## Preserved scientific protocol

Global batch64, seed3072, four logical training ranks, examples and their order,
original per-rank RNG draws, objective and weighting, consistency subset within
each original microbatch, FP32 planner/world precision, optimizer/momentum, data,
world weights, evaluation cases and 10-second/200-action evaluation limits stay
unchanged. Four microbatches of4 are combined into one physical batch16 per rank.
The consistency calculation includes indices0,1,4,5,8,9,12,13 on each rank, rather
than silently changing the fraction of regularized samples. No gradient/data
approximation, quantization or reduced precision is introduced. Floating-point
matrix/reduction roundoff is possible; bitwise-identical training is not claimed.

On a copied actual checkpoint atstep4000, a one-GPU forward/backward benchmark
measured0.348875 seconds per original update versus0.122456 fused: **2.849x**.
CUDA RNG state was identical. Relative gradient L2 difference2.67e-6, maximum
absolute gradient difference1.27e-7, and copied AdamW parameter difference1.19e-7.
Peak allocation including both model copies and optimizer states was3.475GiB.
No campaign optimizer update was performed by this bounded diagnostic. Its
measured kernel/optimizer check portion took4.631seconds; setup/loading are extra.
The live distributed/data-loading speedup must be measured after deployment.

The original frozen checkout stays untouched. An explicit runtime overlay in
repo-runtime/ adds new modules while retaining model revision75e0815. Hash every
runtime file in campaign/throughput-runtime.json and store that manifest digest
in the run configuration and future checkpoints; Git HEAD alone does not describe
this execution. Original config/manifest and model files remain unchanged.

## Hardware schedule and budgets

- Train on GPUs0–3, physical microbatch16, global batch64, four input workers per
  rank, pinned prefetch, and avoid transferring unused coarse/local tensors.
- Evaluate each immutable intermediate checkpoint on GPUs4–7 while training
  continues. Existing inference code, candidate batch sizes and per-case timers
  remain unchanged. Candidate operations already run in batches; merging live
  episodes would change independent RNG/timing semantics and is not introduced.
- Once optimization ends, the last registered validation may shard its104
  independent cases over all8 GPUs. Training ranks then wait without updating.
  Per-episode RNG/reset identities make sharding independent of case order.
- Preserve the original7,200 optimization seconds on4 training GPUs (8GPU-hours)
  OR20,000 updates, whichever is reached first. Do not reset allowance at resume.
  Validation GPU-hours are charged using actual evaluator GPU count; the legacy
  validation_seconds ledger becomes four-GPU-equivalent seconds, with actual
  wall time retained in each evaluation report. Overlap shortens elapsed time;
  it does not create a second optimization allowance.
- W&B remains the same run rm46k69b. Asynchronous validation uses its checkpoint
  step as an explicit chart axis, so late-arriving results are not mislabeled as
  the newest training weights. Select by validation success with earliest-step
  tie break, and wait for all four reports before writing complete.json.

## Safe handoff and recovery

Coordinator scripts/operations/throughput_continuation.py waits for an atomic
checkpoint rename using inotify, pauses the old training process group, and
requires checkpoint step to equal the charged ledger step. If it missed the
boundary, it resumes and waits for the next checkpoint. It preserves model,
optimizer, four RNG states, ledger and best checkpoint before retiring only the
old verified ours-only parent807392 and child858560. No completed optimization
update is discarded. One conservative second is additionally charged for a
potential interrupted next update; no allowance is restored or extended.

After retirement, the new coordinator acquires all8 GPU locks and resumes the
same run. Check campaign/throughput-migration.json, throughput-migration-status.json,
throughput-runtime.json and ours-only-status.json before any recovery. Never
resume retired processes or original cancelled all-method schedulers. Do not
launch another coordinator while the current one or its training/evaluation
children remain alive. Preserve partial reports, pending validation jobs and
all charged time on failure. Do not overwrite the runtime under active workers.

CPU checks passed27 tests including noise/sample/gradient equivalence, original
sampling order, checkpoint queue durability, evaluator environment isolation,
repair semantics and budget rules. Actual handoff and first live optimized
updates must be verified and recorded in PROGRESS before claiming deployment.

After this run, preserve the historical-reference validation comparison, include
already-recorded same-new-world CEM22/104 at20k, record failure modes and stop.
No additional training or test campaign is automatically dispatched.
