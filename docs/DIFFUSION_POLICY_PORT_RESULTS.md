# Diffusion Policy port: completed development result

One fresh seed 3072 run completed 20,000 updates and four fixed 104 evaluations. The curve is **43 / 69 / 70 / 74**, selecting **74/104 (71.15%) at 20,000**.

[Online run](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/cnlxvbpf) and [complete evidence](reports/20261012-additional-baselines/diffusion_policy_port/). Source `0a19c7c67d791fd99376216399b03e8cb71d78c0` and 214 runtime hashes, original data/world/bank tensors, all milestone/selected checkpoints and fixed-case identities pass the CPU audit. Coordinator exited successfully; W&B completion is independently recorded below when verified.

| Update | Success /104 | Timeouts | Mean decision ms |
| --- | ---: | ---: | ---: |
| 5000 | 43 | 0 | 80.88 |
| 10000 | 69 | 0 | 80.85 |
| 15000 | 70 | 0 | 81.01 |
| 20000 | 74 | 0 | 81.37 |

| Task | Selected success /8 |
| --- | ---: |
| assembly | 1 |
| button-press-topdown | 7 |
| coffee-button | 8 |
| dial-turn | 2 |
| door-close | 8 |
| door-open | 8 |
| drawer-close | 6 |
| drawer-open | 8 |
| faucet-open | 7 |
| handle-press | 8 |
| pick-place | 0 |
| plate-slide | 8 |
| reach | 3 |

| Preserved selected reference | Reference /104 | Wins | Regressions |
| --- | ---: | ---: | ---: |
| retrieval_repair_ttt | 80 | 6 | 12 |
| execution_revision | 80 | 8 | 14 |
| flow_reasoning | 79 | 7 | 12 |
| leflow_release | 21 | 55 | 2 |

The port uses a100-step DDPM epsilon objective,16-step deterministic DDIM, a temporal conditional1D U-Net and globally pooled frozen V-JEPA image/goal features. It receives no retrieval, predicted-world selection or error correction. All records have zero predicted world transitions.

Its30 selected failures all reach200 actions, with zero timeouts. Pick-place remains0/8 and assembly1/8; dial-turn2/8 and reach3/8 are also weak. Vanilla flow uses a spatial Transformer conditioner and3.51M parameters; this diffusion port uses pooled conditioning and2.50M. The difference cannot establish that flow matching is intrinsically better than diffusion.

This is a documented image-goal/V-JEPA adaptation, not the authors’ native visual encoder, proprioception interface or unchanged model architecture.

The protocol is unchanged: 6,222 successful expert TRAIN episodes, frozen shared encoder/world, seed 3072/global batch 64, 20k updates OR 8 optimizer-GPUh, four 104 development evaluations, 200 primitive actions and 10 cumulative controller seconds. Four physical A800s were used; actual consumption was **0.592600 optimization GPUh** and **2.238335 evaluation GPUh**. Preparation is separately preserved. Equal ceilings are not equal consumed compute. Model binaries remain on the server; the audit records every checkpoint path/hash.

One training seed and repeatedly reused development checkpoint selection do not establish independent generalization or native-benchmark SOTA. Final tests remain sealed. Retain all failures and per-case/paired records; calibration or surrogate losses are not physical-success labels.

Ten original saved selected-checkpoint [contact sheets](reports/20261012-additional-baselines/diffusion_policy_port/contact-sheets/) were individually visually inspected. Each preserves measured RGB/action indices and the actual expert goal; source trajectory hashes and per-image visual checks are stored beside it. They are a saved subset, not a representative sample of all 104 cases.
