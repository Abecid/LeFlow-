# Additional idle GPUs — October 9, 2026 UTC

The user explicitly authorized using the other four GPUs after inspecting the
other agent's experiments and judging whether they justify retaining resources.
At 04:31 and 04:37 UTC, GPUs 4–7 had zero memory, zero utilization and no GPU
processes. The 112-core host load was about 10. No active trainer was stopped.

## Other pilot assessment

The other chat, **Develop JEPA subgoal planner**, ran a separate PushT BTM/flow
pilot over frozen LeWM features. Its flow method had occupied GPUs 4–7. This is
not the current MetaWorld/V-JEPA experiment, and its scores cannot be compared
numerically with our new task suite.

| Fixed validation comparison | BTM | Flow |
|---|---:|---:|
| Selected checkpoint, hard offset 100 | 25% (5/20) | 20% (4/20) |
| Selected checkpoint, mean over offsets | 35.00% | 33.33% |
| Latest matched 23k updates, offset 100 | 10% | 5% |
| Latest matched 23k updates, offset 25 | 5% | 45% |
| Descriptive primary mean over 32 matched checkpoints | 8.125% | 10.469% |

Selected primary difference: +5 percentage points, paired 95% interval [-15,+25].
One seed and repeated use of the same cases do not establish superiority or
retained success with a speedup. BTM used one generator call versus flow's 16,
but controlled end-to-end efficiency was not demonstrated. Falling latent loss
did not give dependable task success. BTM reached 23,830 saved updates; flow
stopped at 23,000 saved / 23,040 logged, below the 23,830 target. Both formal
completion markers are absent. The stopped old supervisor and zombie BTM launcher
are not active training. The latest full other-agent read was 04:26 UTC; our
04:37:51 UTC read independently verified all four checkpoint hashes unchanged,
99/96 evaluation files, unchanged logged steps, and idle GPUs 4–7.

Judgment: enough evidence to deprioritize this inactive pilot, not evidence that
BTM can never work. Preserve all of its checkpoints, data and negative results;
do not resume, terminate or delete its old processes/artifacts. Compact evidence
is in reports/20261009-repaired-comparison/other-pilot-review.json. Original
[assessment](https://github.com/Abecid/LeFlow-/blob/codex/server-launch-20261004/docs/research/pusht_pair_20261004/reports/20261006T050931Z.md).

## Scheduling amendment, unchanged scientific protocol

Use the same four-GPU execution jobs on pools **0–3** and **4–7**. The running
world job continues on 0–3, unchanged. All heads must wait for its final selected
world checkpoint. Then ours and LeFlow start on separate pools; HWM uses the
first pool that finishes. Only after all training finishes, the same four final
test jobs use these two pools. No new models, seeds, episodes, optimizer updates,
ablations or larger time ceilings are introduced. Each training job still uses
four GPUs, global batch 64, microbatch 4, seed 3072, the same 20k-update/7,200-second
optimization ceiling, and the same periodic evaluation schedule. Actual budgets
remain recorded separately by each job. Concurrent CPU/I/O contention may affect
wall-clock throughput; report observed timings without claiming a measured
speedup from the schedule alone.

The frozen model/training/evaluation checkout remains at **75e0815**, and its
configuration and manifest are unchanged. `campaign.json` retains the original
four-GPU registration; `parallel-status.json` records the subsequent explicit
**eight-GPU aggregate / four-GPU per-job** amendment, physical pools, process
identities, scheduling-script hash, timestamps, commands, failures and handoff.
This changes orchestration only, not the code associated with checkpoints.

`scripts/operations/parallel_flow_campaign.py` runs as a separate, published
scheduler outside the execution checkout. It obtains advisory locks on idle
GPUs 4–7 after three idle checks, then suspends ONLY original supervisor 778158;
its world-training launcher 780827 and workers continue uninterrupted. This
prevents the old sequential scheduler dispatching duplicate jobs. The new
scheduler waits for the world launcher to exit and verifies its completion and
selected checkpoint before launching heads. Every child executes the frozen
modules in the frozen checkout with four ranks. After completed tests, the
original parent resumes, validates/skips existing reports, computes comparison
and failure-analysis artifacts, and publishes W&B results as before.

If a handled scheduling/child failure occurs, the coordinator terminates/reaps
only its own child process groups before resuming the original parent, which can
recover unfinished jobs from their exact checkpoints with charged time retained.
A killed coordinator must not be blindly restarted or followed by a parent
resume while its children are alive. Inspect parallel-status.json and actual
process identities first. Never restart the intentionally held old campaigns.

Four focused scheduler tests cover two-pool dispatch without duplicate/overlap,
no next-job dispatch on failure, rejecting signals after PID identity change, and matching the original explicit
CUDA-device initialization and four-GPU visibility.
At **04:43:43 UTC**, coordinator **795290** was verified in
`waiting_for_world` with all four extra locks held. Original supervisor **778158**
was intentionally stopped (`T`); world launcher **780827** remained active (`S`)
and GPUs 0–3 showed active evaluation. It had reached 5,000 updates and its first
periodic evaluation. GPUs 4–7 remained idle while awaiting the shared world.
No head or final-test task has been dispatched early.

The scheduler's model-execution source remains 75e0815. Its external operational
script is published at e0125c3, SHA-256
`746f61d05cb9ca1b9fea7438c8647dadc16fa11fe33432781be89c6bb870cdd2`.
The initial coordinator was deliberately replaced before head training to match
the original explicit CUDA initialization; the old signal-15 log is accounted
for, and its state was archived. The active monitor knows the two-pool schedule,
intentional parent suspension and recovery rules. See the saved
parallel-verification.json, parallel-status.json and parallel-allocation.json.
