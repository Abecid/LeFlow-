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
