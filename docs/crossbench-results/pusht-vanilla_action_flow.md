# pusht — vanilla_action_flow completed run

The fixed four-checkpoint success curve is **7 / 13 / 15 / 17 /50**, selecting **17/50 at 20,000 updates**. The fresh policy reached 20,000 updates, consuming 0.518629 optimizer GPUh and 0.017374 evaluation GPUh. Train/eval wall time: 9.19 minutes.

[Online run](https://wandb.ai/attentionx2023/leflow-crossbench/runs/jljz9gud) is finished. The CPU source/data/case/checkpoint audit passes 245 runtime files, same case-plan hash at every checkpoint, finite model tensors and exact selected/milestone weights. Full [evidence](../reports/20261013-crossbench-baselines/runs/pusht/vanilla_action_flow/) preserves periodic metrics, all physical outcomes and checkpoint hashes.

| Update | Success /50 | Timeouts | Mean decision ms |
| --- | ---: | ---: | ---: |
| 5,000 | 7 | 0 | 10.72 |
| 10,000 | 13 | 0 | 9.87 |
| 15,000 | 15 | 0 | 9.91 |
| 20,000 | 17 | 0 | 9.85 |

Selected failure reasons: {'action_limit': 33}. Selected mean action count is 42.76; world transitions total 0. These are factual selected trajectories, not outcomes of rejected actions.

This benchmark uses released frozen LeWM features rather than MetaWorld V-JEPA. The latent cache retains 14,948 TRAIN episodes; 14,948 episodes provide 1,439,235 native-safe starts among 1,496,570 nominal starts. Raw out-of-box commands are preserved, not clipped or given fabricated labels. Factual history and complete goal prefixes respect native limits; all controllers share this rule. Validation cases remain unchanged. Full-source normalization/backbone exposure are documented in the campaign runbook.

One seed and50 repeatedly selected development cases are screening evidence, not independent final evaluation or native-method SOTA. Other controllers continue under the same ceilings; no extra seed, MetaWorld retraining or sealed test was run.
