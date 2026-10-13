# SAGE-inspired MetaWorld port: completed development result

One fresh seed 3072 run completed 20,000 updates and four fixed 104 evaluations. The curve is **62 / 77 / 77 / 75**, selecting **77/104 (74.04%) at 10,000**.

[Online run](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/kfv02fj4) and [complete evidence](reports/20261012-additional-baselines/sage_port/). Source `0a19c7c67d791fd99376216399b03e8cb71d78c0` and 214 runtime hashes, original data/world/bank tensors, all milestone/selected checkpoints and fixed-case identities pass the CPU audit. Coordinator exited successfully and W&B independently reports finished; the batch final-health receipt preserves that check.

| Update | Success /104 | Timeouts | Mean decision ms |
| --- | ---: | ---: | ---: |
| 5000 | 62 | 0 | 76.77 |
| 10000 | 77 | 0 | 75.02 |
| 15000 | 77 | 0 | 75.37 |
| 20000 | 75 | 0 | 75.01 |

| Task | Selected success /8 |
| --- | ---: |
| assembly | 0 |
| button-press-topdown | 8 |
| coffee-button | 8 |
| dial-turn | 1 |
| door-close | 8 |
| door-open | 8 |
| drawer-close | 8 |
| drawer-open | 7 |
| faucet-open | 8 |
| handle-press | 8 |
| pick-place | 0 |
| plate-slide | 8 |
| reach | 5 |

| Preserved selected reference | Reference /104 | Wins | Regressions |
| --- | ---: | ---: | ---: |
| retrieval_repair_ttt | 80 | 9 | 12 |
| execution_revision | 80 | 10 | 13 |
| flow_reasoning | 79 | 7 | 9 |
| leflow_release | 21 | 56 | 0 |
| vanilla_action_flow | 98 | 1 | 22 |
| diffusion_policy_port | 74 | 7 | 4 |

The duration-conditioned deterministic spatial subgoal Transformer is supervised on factual future TRAIN features, with an eight-mode GMM trained on its predicted subgoal. Deployment compares eight predefined durations using four proposals and three CEM rounds (480 frozen-world transitions per decision). No execution-memory adaptation, our workspace, residual-flow repair or privileged simulator state is added. This is a SAGE-inspired shared-backbone port, not the authors' native generator/checkpoint.

Selected success is 77/104 with 0 timeouts. Pick-place is 0/8, assembly 0/8 and reach 5/8. All selected failure action counts and paired identities remain in the full evaluation; the observed pattern does not identify a unique cause. The strongest measured vanilla-flow control remains 98/104.

The protocol is unchanged: 6,222 successful expert TRAIN episodes, frozen shared encoder/world, seed 3072/global batch 64, 20k updates OR 8 optimizer-GPUh, four 104 development evaluations, 200 primitive actions and 10 cumulative controller seconds. Four physical A800s were used; actual consumption was **0.714042 optimization GPUh** and **1.807850 evaluation GPUh**. Preparation is separately preserved. Equal ceilings are not equal consumed compute. Model binaries remain on the server; the audit records every checkpoint path/hash.

One training seed and repeatedly reused development checkpoint selection do not establish independent generalization or native-benchmark SOTA. Final tests remain sealed. Retain all failures and per-case/paired records; calibration or surrogate losses are not physical-success labels.

Ten original saved selected-checkpoint [contact sheets](reports/20261012-additional-baselines/sage_port/contact-sheets/) were individually visually inspected. Each preserves measured RGB/action indices and the actual expert goal; source trajectory hashes and per-image visual checks are stored beside it. They are a saved subset, not a representative sample of all 104 cases.
