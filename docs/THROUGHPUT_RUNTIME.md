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

## Four-GPU distributed check

A bounded copy of the step6000 checkpoint was also checked on GPUs4–7 while the
original job continued. Original gradient accumulation and the fused execution
both used the actual four-rank data partition and DDP reduction. Every rank
retained identical CUDA RNG state. The global relative gradient L2 difference
was1.76e-6. Per-update forward/backward time was0.407543s original versus0.143119s
fused, **2.848x** faster. No research optimizer updates were performed. The
measured diagnostic portion took3.300s across four GPUs, excluding setup/loading.
The exact one-campaign diagnostic source is preserved in
scripts/operations/benchmark_fused_ddp.py; it writes a separate diagnostic record.

## Handoff correction

The first handoff attempt atstep7000 exposed that torchrun starts its rank workers
in independent process groups. Stopping/killing only the launcher group did not
retire those workers. The GPU-idle gate rejected the new launch, so no duplicate
training was dispatched. The original four ranks kept training and saving their
updates under the original ledger. The7000 checkpoint was preserved but was not
used to roll back that later work.

The corrected handoff identifies every rank by PID, start time, command and
working directory, stops all four worker groups at an atomic checkpoint boundary,
requires each leader to be stopped and the ledger/checkpoint step to match, then
archives and retires those groups individually. The original launcher-only
migration mode now fails closed. The later migration record supersedes the7000
attempt and retains the failed attempt as separate evidence. Completion and
real resumed throughput must be verified before treating the change as live.

## Verified live outcome

At11:40PM Pacific, the corrected handoff preserved step8000 and retired all old
worker groups. Newcoordinator917422 resumed the same run via launcher920245.
No completed update was discarded. At11:43PM, measured optimization throughput
rose from158.18 to441.43 examples/second (**2.791x**). Runtime/source/data/world
hashes and online W&B metadata were verified, with finite/monotone training logs.

At11:46PM all8GPUs were active: training0–3 had advanced to10544 while evaluator
926345 on4–7 processed the immutable10k checkpoint and had completed8 cases.
Intermediate evaluation is therefore verified concurrent with training. The
all8GPU final validation is scheduled after optimization and is not yet observed.
Expected completion including evaluation is12:10–12:25AM Pacific October9 if
current throughput holds. The27 tests and measured FP32/RNG/gradient checks remain
bounded evidence of equivalent computation; they are not proof of identical
floating-point trajectories or improved benchmark accuracy.

## Completion verified — October 9, 07:22 UTC

The same seed3072 head reached20,000 updates with4,751.897 charged four-GPU
optimization seconds. Final registered validation actually ran on eight GPUs
for280.063 seconds, charged0.622362 GPU-hours. Its15/104 score is preserved;
training throughput improvements did not establish better accuracy. All four
rounds are17,17,16,15, with best5k selected by the original tie-break rule.
W&B finished and all owned processes exited; all eight GPUs were empty/idle.
Transient upload retries resolved without intervention. The shutdown log's
missing destroy_process_group warning is recorded for a future cleanup fix;
no execution code was changed during this run or additional evaluation launched.
See [final results](REPAIRED_VALIDATION_RESULTS.md).
