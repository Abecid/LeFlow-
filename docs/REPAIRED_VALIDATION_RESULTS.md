# Repaired-method validation review — October 9, 2026

The repaired method selected **17/104 successes (16.35%)**, below the already
recorded same-world CEM diagnostic at 22/104 and historical LeFlow at 27/104.
Its four registered rounds declined **17 → 17 → 16 → 15**. This iteration does
not provide evidence that the repairs improved task success. The registered
selection rule retains the earliest tied best, checkpoint 5,000.

These are completed **validation** results, not final-test results. All reports
contain the same 104 reset IDs, reset seeds, cached episode hashes and model
seed 3072. The 13 training tasks each contribute eight cases. No held-out-task
result is available; final tests remain reserved. No baseline was retrained or
re-evaluated for this review.

| Selected validation reference | Successes | Rate | Checkpoint |
|---|---:|---:|---:|
| Repaired proposed method | 17/104 | 16.35% | 5,000 |
| CEM with selected new world, already recorded | 22/104 | 21.15% | World 20,000 |
| Historical LeFlow adaptation | 27/104 | 25.96% | 20,000 |
| Historical original proposed method | 23/104 | 22.12% | 5,000 |
| Historical HWM adaptation | 8/104 | 7.69% | 20,000 |
| Historical CEM with selected old world | 7/104 | 6.73% | World 20,000 |

The historical implementations use a different representation/world. Both old
flow heads have the confirmed sampler output restriction; LeFlow is also an
adaptation rather than an exact paper reproduction. The new representation,
compatible world, endpoint parameterization and scoring changes are combined.
These results cannot isolate a planner or sampler effect, establish superiority
over corrected LeFlow, or substantiate a SOTA claim. Generated-plan consistency
is already described in LeFlow's paper; our novelty claim remains narrowed.

## Observed failures

The selected checkpoint succeeds on drawer-close 8/8, handle-press 8/8 and reach 1/8,
and on none of the other ten tasks. **Every one of its 87 failures exhausted the
10-second controller allowance**, after 92–96 primitive actions, below the 200
primitive-action ceiling. The later rounds have 87, 88 and 89 timeout failures.
Timeout is an observed stopping condition; it does not establish a physical
cause such as failed grasping, bad contact or unreachable goals.

With the same selected new world, CEM and ours share 14 successful cases.
CEM alone succeeds on coffee-button resets 00000, 00001, 00002, 00004, 00005, 00006,
00007 and reach 00003. Ours alone succeeds on drawer-close 00004, 00007 and
handle-press 00001. These eight CEM-only cases are useful priorities for reviewing
why the learned planner misses outcomes already achieved with this world.

Historical LeFlow has 15 successes where ours fails: four coffee-button cases,
three dial-turn cases, seven door-close cases and reach 00005. Ours has five
successes where historical LeFlow fails. The complete paired inventories,
including reset/episode identities and stopping conditions, are preserved in
[final-validation-review.json](reports/20261009-repaired-comparison/final-validation-review.json).

From 10k to 15k, drawer-close 00005 changes from success to timeout. From 15k to 20k,
dial-turn 00006 and drawer-close 00005 improve, while drawer-close 00002,
handle-press 00002 and reach 00007 regress. Aggregate deterioration therefore does
not mean every case deteriorated. Latent subgoal distances and predicted costs
remain proxies; these records do not demonstrate that either is a reliable
measure of executable physical progress.

## Uncertainty and selection

The JSON review includes 10,000 paired bootstrap resamples within the 13 fixed
training-task strata, sharing each sampled reset across all methods. These
conditional intervals cover paired-reset uncertainty only. They do not measure
training-seed variation, task-generalization uncertainty, or remove optimism
from selecting checkpoints on these same cases. With only eight cases per task,
all-zero/all-one strata have degenerate empirical resampling intervals. Historical
implementation confounds are unaffected by resampling.

For ours minus same-world CEM, the observed difference is −4.81 percentage points:
paired 95% interval [−8.65, −0.96], or [−9.62, +0.96] after a Bonferroni adjustment across
five reference comparisons. Ours minus historical LeFlow is −9.62 points,
paired 95% [−15.38, −3.85]. These are descriptive selected-validation differences;
they do not establish a general method ranking. Training seed 3072 is the only
training seed. The separate bootstrap RNG only makes offline analysis repeatable.

## Compute, execution and preservation

The head reached 20,000 updates at 4,751.897 optimization seconds on four GPUs,
**5.279885 GPU-hours**, below the 7,200-second allowance. All four validations used
**2.287832 GPU-hours**. The final registered validation used eight GPUs for
280.063 seconds (0.622362 GPU-hours), after optimization stopped. This was
validation, not a final test. The migration preserved every completed update
and charged one additional conservative optimization second within the ledger.

The compatible world used 2.978770 optimization and 1.862768 validation GPU-hours.
Across both MetaWorld campaigns, cumulative optimization is **29.370732 GPU-hours**
and completed registered validation **13.200919 GPU-hours**. Separate bounded
repair diagnostics contribute 0.136526 and measured throughput checks 0.004953
GPU-hours. Their known subtotal is 42.713129 GPU-hours, not a complete infrastructure
bill: preparation, preflights, interrupted historical tests, diagnostic setup/
loading, CPU checks and post-compute upload/shutdown occupancy are not fully totaled.
See [compute-accounting.json](reports/20261009-repaired-comparison/compute-accounting.json).

Original model/config source remains clean at 75e0815. Runtime manifest
431ae6e5e04be96360e62962395b0184e263320fac45e8e6deb2f4ed1cbc7f66 and all listed
files match; data manifest and world hashes are unchanged. Best checkpoint SHA
2afbb9693eab1b91a589c287e5e8d6422d9b8c108775c2815c42ffd51e74151f identifies the 5k
checkpoint, which predates the throughput migration. It is not relabeled as
trained with the fused runtime. The final 20k checkpoint separately records the
runtime overlay and is retained alongside immutable evaluation checkpoints.

The model and reports are complete. At 07:22 UTC, W&B was **finished**, the
coordinator recorded `completed_validation_review`, every owned process had
exited and all eight GPUs were empty/idle. Final uploads encountered transient
EOF/header-timeout retries, then synced successfully without intervention. The
shutdown log warns that `destroy_process_group()` was not called; actual GPU
release was independently verified. This is a future cleanup improvement, not
an unresolved worker or lost result. Both automatic review files and the richer
six-reference paired inventory are preserved. The existing monitor will be
paused after publication; no further run or final test is authorized.
[W&B run](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/rm46k69b)
retains the same run ID and explicit validation checkpoint axis.

## Proposed follow-up for user review only

No further experiment is authorized or queued. Before considering another
training run, review the saved CEM-only coffee-button/reach cases and the
historical door-close/dial-turn failures. A separately approved bounded diagnostic
could compare candidate ranking with short executable progress under the same
frozen world/checkpoint, or isolate the consistency term and history/static-goal
choice under equal original training ceilings. Those would be new diagnostics
or ablations requiring explicit authorization; none was executed here. Do not
use final-test outcomes to choose them or grant our method extra optimization.
