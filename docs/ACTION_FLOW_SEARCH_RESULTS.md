# Action flow with fixed-world search: completed development result

One fresh seed 3072 run completed 20,000 updates and four fixed 104 evaluations. The curve is **74 / 79 / 84 / 78**, selecting **84 /104 (80.77%) at 15,000**.

[Online run](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/ms76k64y) and [complete evidence](reports/20261012-additional-baselines/action_flow_search/). Source `0a19c7c67d791fd99376216399b03e8cb71d78c0` and 214 runtime hashes, original data/world/bank tensors, all milestone/selected checkpoints and fixed-case identities pass the CPU audit. Coordinator exited successfully. W&B independently reports the run finished; its completion record is preserved in the batch archive.

| Update | Success /104 | Timeouts | Mean decision ms |
| --- | ---: | ---: | ---: |
| 5000 | 74 | 0 | 90.00 |
| 10000 | 79 | 0 | 89.95 |
| 15000 | 84 | 0 | 91.62 |
| 20000 | 78 | 0 | 91.64 |

| Task | Selected success /8 |
| --- | ---: |
| assembly | 5 |
| button-press-topdown | 8 |
| coffee-button | 8 |
| dial-turn | 8 |
| door-close | 8 |
| door-open | 6 |
| drawer-close | 8 |
| drawer-open | 8 |
| faucet-open | 6 |
| handle-press | 8 |
| pick-place | 1 |
| plate-slide | 8 |
| reach | 2 |

| Preserved selected reference | Reference /104 | Wins | Regressions |
| --- | ---: | ---: | ---: |
| retrieval_repair_ttt | 80 | 10 | 6 |
| execution_revision | 80 | 12 | 8 |
| flow_reasoning | 79 | 12 | 7 |
| leflow_release | 21 | 68 | 5 |
| vanilla_action_flow | 98 | 2 | 16 |
| diffusion_policy_port | 74 | 15 | 5 |

The controller samples three independent pools of four flow candidates for each of eight retrieved/direct targets and ranks five-block predicted consequences with the frozen world: 480 world transitions per decision. It has no adaptive correction, learned workspace or residual repair. Local-goal training and retrieval differ from vanilla flow, so this is a system comparison rather than an isolated world-search ablation.

All 20 selected failures reach 200 actions; none time out. Pick-place scores 1/8 and reach 2/8; assembly scores 5/8. Success falls from 84 to 78 between 15k and 20k despite continued demonstration-loss improvement. The logs do not identify the causal reason for this regression. The selected search controller remains below vanilla flow 98; its paired cases are preserved.

The protocol is unchanged: 6,222 successful expert TRAIN episodes, frozen shared encoder/world, seed 3072/global batch 64, 20k updates OR 8 optimizer-GPUh, four 104 development evaluations, 200 primitive actions and 10 cumulative controller seconds. Four physical A800s were used; actual consumption was **0.601418 optimization GPUh** and **1.836139 evaluation GPUh**. Preparation is separately preserved. Equal ceilings are not equal consumed compute. Model binaries remain on the server; the audit records every checkpoint path/hash.

One training seed and repeatedly reused development checkpoint selection do not establish independent generalization or native-benchmark SOTA. Final tests remain sealed. Retain all failures and per-case/paired records; calibration or surrogate losses are not physical-success labels.
