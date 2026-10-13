# Feedback World Model-inspired port: completed development result

One fresh seed 3072 run completed 20,000 updates and four fixed 104 evaluations. The curve is **42 / 65 / 75 / 76**, selecting **76 /104 (73.08%) at 20,000**.

[Online run](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/rc8s6gp6) and [complete evidence](reports/20261012-additional-baselines/feedback_wm_port/). Source `0a19c7c67d791fd99376216399b03e8cb71d78c0` and 214 runtime hashes, original data/world/bank tensors, all milestone/selected checkpoints and fixed-case identities pass the CPU audit. Coordinator exited successfully. W&B independently reports the run finished; its completion record is preserved in the batch archive.

| Update | Success /104 | Timeouts | Mean decision ms |
| --- | ---: | ---: | ---: |
| 5000 | 42 | 62 | 106.95 |
| 10000 | 65 | 39 | 107.37 |
| 15000 | 75 | 29 | 107.58 |
| 20000 | 76 | 28 | 107.17 |

| Task | Selected success /8 |
| --- | ---: |
| assembly | 2 |
| button-press-topdown | 8 |
| coffee-button | 8 |
| dial-turn | 3 |
| door-close | 8 |
| door-open | 7 |
| drawer-close | 6 |
| drawer-open | 8 |
| faucet-open | 7 |
| handle-press | 8 |
| pick-place | 0 |
| plate-slide | 8 |
| reach | 3 |

| Preserved selected reference | Reference /104 | Wins | Regressions |
| --- | ---: | ---: | ---: |
| retrieval_repair_ttt | 80 | 7 | 11 |
| execution_revision | 80 | 9 | 13 |
| flow_reasoning | 79 | 9 | 12 |
| leflow_release | 21 | 57 | 2 |
| vanilla_action_flow | 98 | 0 | 22 |
| diffusion_policy_port | 74 | 5 | 3 |

The port adds execution-residual EMA correction and differentiable predicted-prefix guidance to the shared conditional DDPM/DDIM policy. It preserves the factual feedback direction without adding our nonlinear memory, repair workspace or meta-training objective. This is an explicitly documented Feedback-WM-inspired adaptation, not unchanged author robotics code.

All 28 selected failures exhaust the 10-second controller allowance after 166–198 actions. Pick-place is 0/8; assembly 2/8, dial-turn 3/8 and reach 3/8 are also weak. This does not show that extra time would make those cases succeed. The selected score 76 versus Diffusion Policy 74 is a two-case net difference; all four matched-step checkpoints and best/last weights were compared on CPU. They are not bit-identical (60 of 61 tensors differ; maximum absolute difference at 20k is 0.00053736). Thus this is an independently trained method-package comparison, not a feedback-only ablation. The port wins five cases and loses three against Diffusion Policy.

The protocol is unchanged: 6,222 successful expert TRAIN episodes, frozen shared encoder/world, seed 3072/global batch 64, 20k updates OR 8 optimizer-GPUh, four 104 development evaluations, 200 primitive actions and 10 cumulative controller seconds. Four physical A800s were used; actual consumption was **0.581692 optimization GPUh** and **2.275778 evaluation GPUh**. Preparation is separately preserved. Equal ceilings are not equal consumed compute. Model binaries remain on the server; the audit records every checkpoint path/hash.

One training seed and repeatedly reused development checkpoint selection do not establish independent generalization or native-benchmark SOTA. Final tests remain sealed. Retain all failures and per-case/paired records; calibration or surrogate losses are not physical-success labels.
