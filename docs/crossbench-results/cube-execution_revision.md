# cube — execution_revision completed run

The fixed four-checkpoint success curve is **28 / 30 / 32 / 31 /50**, selecting **32/50 at 15,000 updates**. The fresh policy reached 20,000 updates, consuming 1.609121 optimizer GPUh and 0.423356 evaluation GPUh. Train/eval wall time: 31.80 minutes.

[Online run](https://wandb.ai/attentionx2023/leflow-crossbench/runs/r2gk5vnj) is finished. The CPU source/data/case/checkpoint audit passes 245 runtime files, same case-plan hash at every checkpoint, finite model tensors and exact selected/milestone weights. Full [evidence](../reports/20261013-crossbench-baselines/runs/cube/execution_revision/) preserves periodic metrics, all physical outcomes and checkpoint hashes.

| Update | Success /50 | Timeouts | Mean decision ms |
| --- | ---: | ---: | ---: |
| 5,000 | 28 | 0 | 68.63 |
| 10,000 | 30 | 0 | 68.14 |
| 15,000 | 32 | 0 | 68.62 |
| 20,000 | 31 | 0 | 67.09 |

Selected failure reasons: {'action_limit': 18}. Selected mean action count is 21.28; world transitions total 111,840. These are factual selected trajectories, not outcomes of rejected actions.

This benchmark uses released frozen LeWM features rather than MetaWorld V-JEPA. The latent cache retains 8,000 TRAIN episodes; 8,000 episodes provide 1,408,000 native-safe starts among 1,408,000 nominal starts. Raw out-of-box commands are preserved, not clipped or given fabricated labels. Factual history and complete goal prefixes respect native limits; all controllers share this rule. Validation cases remain unchanged. Full-source normalization/backbone exposure are documented in the campaign runbook.

One seed and50 repeatedly selected development cases are screening evidence, not independent final evaluation or native-method SOTA. Other controllers continue under the same ceilings; no extra seed, MetaWorld retraining or sealed test was run.
