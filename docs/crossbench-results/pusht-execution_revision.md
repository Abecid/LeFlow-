# pusht — execution_revision completed run

The fixed four-checkpoint success curve is **33 / 30 / 38 / 33 /50**, selecting **38/50 at 15,000 updates**. The fresh policy reached 20,000 updates, consuming 1.681470 optimizer GPUh and 0.039420 evaluation GPUh. Train/eval wall time: 27.13 minutes.

[Online run](https://wandb.ai/attentionx2023/leflow-crossbench/runs/0a7tgv4j) is finished. The CPU source/data/case/checkpoint audit passes 245 runtime files, same case-plan hash at every checkpoint, finite model tensors and exact selected/milestone weights. Full [evidence](../reports/20261013-crossbench-baselines/runs/pusht/execution_revision/) preserves periodic metrics, all physical outcomes and checkpoint hashes.

| Update | Success /50 | Timeouts | Mean decision ms |
| --- | ---: | ---: | ---: |
| 5,000 | 33 | 0 | 68.54 |
| 10,000 | 30 | 0 | 67.46 |
| 15,000 | 38 | 0 | 65.66 |
| 20,000 | 33 | 0 | 64.69 |

Selected failure reasons: {'action_limit': 12}. Selected mean action count is 34.56; world transitions total 173,280. These are factual selected trajectories, not outcomes of rejected actions.

This benchmark uses released frozen LeWM features rather than MetaWorld V-JEPA. The latent cache retains 14,948 TRAIN episodes; 14,948 episodes provide 1,439,235 native-safe starts among 1,496,570 nominal starts. Raw out-of-box commands are preserved, not clipped or given fabricated labels. Factual history and complete goal prefixes respect native limits; all controllers share this rule. Validation cases remain unchanged. Full-source normalization/backbone exposure are documented in the campaign runbook.

One seed and50 repeatedly selected development cases are screening evidence, not independent final evaluation or native-method SOTA. Other controllers continue under the same ceilings; no extra seed, MetaWorld retraining or sealed test was run.
