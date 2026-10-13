# Vanilla conditional action flow: completed development result

One fresh seed3072 run completed20,000 updates and four fixed104 evaluations. The curve is **77 / 84 / 89 / 98**, selecting **98/104 (94.23%) at20000**.

[Online run](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/j1m6jzog) and [complete evidence](reports/20261012-additional-baselines/vanilla_action_flow/). Source `0a19c7c67d791fd99376216399b03e8cb71d78c0` and214 runtime hashes, original data/world/bank tensors, all milestone/selected checkpoints and fixed-case identities pass the CPU audit. Coordinator exited successfully; W&B completion is independently recorded below when verified.

| Update | Success /104 | Timeouts | Mean decision ms |
| --- | ---: | ---: | ---: |
| 5000 | 77 | 0 | 55.89 |
| 10000 | 84 | 0 | 57.07 |
| 15000 | 89 | 0 | 56.29 |
| 20000 | 98 | 0 | 55.83 |

| Task | Selected success /8 |
| --- | ---: |
| assembly | 8 |
| button-press-topdown | 8 |
| coffee-button | 8 |
| dial-turn | 8 |
| door-close | 8 |
| door-open | 8 |
| drawer-close | 8 |
| drawer-open | 8 |
| faucet-open | 8 |
| handle-press | 8 |
| pick-place | 4 |
| plate-slide | 8 |
| reach | 6 |

| Preserved selected reference | Reference /104 | Wins | Regressions |
| --- | ---: | ---: | ---: |
| retrieval_repair_ttt | 80 | 19 | 1 |
| execution_revision | 80 | 20 | 2 |
| flow_reasoning | 79 | 22 | 3 |
| leflow_release | 21 | 77 | 0 |

This baseline has **no runtime retrieval, world search, workspace revision, TTT or residual repair**. All records have zero predicted world transitions. It uses only current/final-goal image features and an eight-step conditional action flow, executing two actions before replanning. The exact consumed training state/action fields match the established sampler; dropping unused history fields is a lossless data-loading change.

Selected success exceeds GMM80 and retrieval80 by18 cases. The six failures are four pick-place and two reach cases; all reach200 actions, with no controller timeouts. Eleven tasks score8/8. Increasing training improves success across these checkpoints, in contrast with several prior complex controllers. This is a system-level result, not an isolated causal finding about any removed component.

The retained proposed repair controller80 does not outperform this98 baseline. Do not relabel the standard BC control as a new reasoning/TTT method or preserve the previous claim that80 is the strongest measured reference.

The protocol is unchanged:6222 successful expert TRAIN episodes, frozen shared encoder/world, seed3072/global64,20k updates OR8 optimizer-GPUh, four104 development evaluations,200 primitive actions and10 cumulative controller seconds. Four physical A800s were used; actual consumption was **0.613096 optimization GPUh** and **1.475266 evaluation GPUh**. Preparation is separately preserved. Equal ceilings are not equal consumed compute. Model binaries remain on the server; the audit records every checkpoint path/hash.

One training seed and repeatedly reused development checkpoint selection do not establish independent generalization or native-benchmark SOTA. Final tests remain sealed. Retain all failures and per-case/paired records; calibration or surrogate losses are not physical-success labels.
