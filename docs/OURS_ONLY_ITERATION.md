# Fixed baselines, repaired-method iteration

Latest user correction, October 8, 2026 (Pacific): baseline training should be
performed once and retained; development should focus on our method. Scheduling
another LeFlow/HWM training round was an overly broad interpretation of the
request to train from scratch after repairs. It has been cancelled before either
baseline started: **zero repeated baseline training updates**.

## Existing reference results

All numbers below are selected validation results on the same 104 resets, not a
completed final-test comparison. LeFlow and HWM already reached 20,000 updates.
CEM has no learned head: it uses the previously trained world.

| Historical reference | Selected successes | Rate |
|---|---:|---:|
| LeFlow adaptation | 27/104 | 25.96% |
| HWM adaptation | 8/104 | 7.69% |
| CEM with selected old world | 7/104 | 6.73% |
| Original proposed method | 23/104 | 22.12% |

Keep all original checkpoints, validation cases, outcomes and compute ledgers
unchanged. No repeated baseline training or extra baseline final-test campaign
is scheduled. The incomplete older PushT pilot is a separate benchmark.

## What continues

The currently running new world model continues with its original budget; its
static state representation is required by our repaired pipeline. We cannot
substitute the old causal-state world without undoing that repair. Once this
world finishes, train exactly `joint_flow_consistent`, seed 3072, from scratch:
four GPUs, global batch64/microbatch4, at most 20,000 updates OR 7,200 optimization
seconds, and four fixed 104-case validation rounds. No baseline head is in this
queue. Existing world-validation CEM diagnostics are part of the registered
world training, not replacement historical baseline scores. All charged compute
is retained, including development/failed attempts; do not hide cumulative R&D
cost behind a per-run limit.

Use identical original training and validation cases. Automatically select our
checkpoint with the existing validation selection rule, write an explicitly
historical-reference score table and paired validation failure inventory, then
stop for review. Do not automatically start more variants or ablations just
because GPUs are available. Iterate on validation; keep the final test reserved
for the selected method rather than repeatedly tuning to it.

## Why the scope matters

The cancelled reruns were intended to put all methods on the repaired shared
representation/world/sampler. That would have been a controlled common-world
comparison, but is not necessary for initial full-pipeline development against
fixed historical references. Our repairs now change more than the planner alone;
report those differences and all compute explicitly. The old LeFlow sampler also
has a confirmed implementation defect. Beating that reference is useful internal
evidence, not proof of superiority over a corrected LeFlow or published SOTA.
If a corrected baseline is needed for a definitive claim, evaluate that need
explicitly once; do not silently retrain baselines each iteration.

## Operational handoff

At 21:49:38 Pacific on October 8 (04:49:38 UTC October 9), waiting-only parallel
coordinator 795290 was cancelled after verifying that it had dispatched no jobs.
Its stopped all-method parent 778158 remains stopped; NEVER resume it. The world
launcher 780827 and workers were untouched. The replacement
`scripts/operations/ours_only_continuation.py` waits for world completion, then
retires only that cancelled parent, acquires GPUs0–3 and launches our one head
using frozen execution source 75e0815. GPUs4–7 are released and idle.

Read `campaign/baseline-queue-cancelled.json` and `campaign/ours-only-status.json`
for live scope and process identities. The continuation does not launch final
tests or hand back to either cancelled scheduler. On a failure it preserves
checkpoints/charged ledgers, stops only its own child group if necessary and
reports the error; recovery must verify no duplicate jobs and retain charged
time. The old campaigns remain held. Do not overwrite model code or protocol
under a running job; the new scope is an external orchestration amendment.

Validation analysis reproduces the saved 23/27/8/7 successes and verifies exact
case IDs, reset seeds, episode hashes and model seed before pairing outcomes.
It rejects mismatched identities; it deliberately does not masquerade as the
strict same-world/same-code final comparison. New output filenames are
`historical-validation-comparison.json` and `validation-failure-analysis.json`.

## Verified continuation

At **9:54 PM Pacific, October 8**, continuation PID **807392** was alive in
`waiting_for_world`, with only `joint_flow_consistent` scheduled. Cancelled
coordinator795290 was absent, old parent778158 remained stopped, and world
launcher780827 continued unchanged. Only the world run directory exists: no
baseline or proposed head has started yet. World had reached update7896 and
completed its first validation. External continuation SHA-256:
`712a0f2b141ba8bf2daf35855bb74fec4a2664818ae9bb5160d842e7dd7a5857`.

The already-registered first CEM world diagnostic scored **23/104** with the
new world at5k updates, versus the historical selected CEM reference7/104. This
is an interim world diagnostic, not our planner's result or a replacement of
saved baseline scores. It demonstrates why the changed shared representation/
world must be disclosed: improvements cannot be attributed solely to the
planner. Preserve and include the already-computed CEM diagnostic for the
finally selected new world alongside the historical-reference analysis; this
requires no extra training or evaluation. Final tests remain unrun for this
iteration. Latest monitor instructions enforce this scope and prohibit automatic
subsequent training runs.

## World completion and head handoff — October 9, 05:42 UTC

The registered world finished at 20,000 updates with all four validations;
the ours-only coordinator started only `joint_flow_consistent` at 05:42:09 UTC.
Launcher 858560 and ranks 858566–858569 use GPUs 0–3, seed 3072, under the unchanged
budget. Online head run is
[rm46k69b](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/rm46k69b).
No baseline was retrained and no final test is queued.

The selected world is the 20k checkpoint, chosen by minimum prediction loss
0.00974052, SHA-256
`ae3f910da43f0a43dd65945394444a0b11719ed042cdacffacc2faaf824ec5a3`.
Its recorded CEM diagnostic is **22/104**, below the intermediate 24/104 peak;
report 22 alongside historical CEM 7 and the other fixed references. Do not
reselect the world using CEM outcomes or omit the decline. Lower prediction
loss did not imply higher controller success across these final checkpoints.
The diagnostic and paired case changes are preserved in
[selected-world-diagnostic.json](reports/20261009-repaired-comparison/selected-world-diagnostic.json).
All 82 failures hit the controller allowance; no physical cause is established.
The repaired head has no completed validation yet at the 05:48 UTC snapshot.

## Completed iteration — October 9, 07:22 UTC

The authorized world and single repaired head are complete. Head rounds scored
17,17,16,15 /104; earliest-best5k remains selected at17/104. W&B finished and all
owned processes exited. Final testing remains reserved; no next variant is
authorized. See [the final validation review](REPAIRED_VALIDATION_RESULTS.md) for
paired failure cases, same-world CEM22, historical references, compute and limits.
The monitor is to be paused after verified publication.
