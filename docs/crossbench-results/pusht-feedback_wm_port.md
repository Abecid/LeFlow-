# pusht — feedback_wm_port completed run

The fixed four-checkpoint success curve is **15 / 13 / 24 / 27 /50**, selecting **27/50 at 20,000 updates**. The fresh policy reached 20,000 updates, consuming 0.507886 optimizer GPUh and 0.050503 evaluation GPUh. Train/eval wall time: 9.54 minutes.

[Online run](https://wandb.ai/attentionx2023/leflow-crossbench/runs/5ol8rxc6) is finished. The CPU source/data/case/checkpoint audit passes 245 runtime files, same case-plan hash at every checkpoint, finite model tensors and exact selected/milestone weights. Full [evidence](../reports/20261013-crossbench-baselines/runs/pusht/feedback_wm_port/) preserves periodic metrics, all physical outcomes and checkpoint hashes.

| Update | Success /50 | Timeouts | Mean decision ms |
| --- | ---: | ---: | ---: |
| 5,000 | 15 | 0 | 79.64 |
| 10,000 | 13 | 0 | 78.42 |
| 15,000 | 24 | 0 | 78.20 |
| 20,000 | 27 | 0 | 78.45 |

Selected failure reasons: {'action_limit': 23}. Selected mean action count is 36.10; world transitions total 3,339. These are factual selected trajectories, not outcomes of rejected actions.

This benchmark uses released frozen LeWM features rather than MetaWorld V-JEPA. The latent cache retains 14,948 TRAIN episodes; 14,948 episodes provide 1,439,235 native-safe starts among 1,496,570 nominal starts. Raw out-of-box commands are preserved, not clipped or given fabricated labels. Factual history and complete goal prefixes respect native limits; all controllers share this rule. Validation cases remain unchanged. Full-source normalization/backbone exposure are documented in the campaign runbook.

One seed and50 repeatedly selected development cases are screening evidence, not independent final evaluation or native-method SOTA. Other controllers continue under the same ceilings; no extra seed, MetaWorld retraining or sealed test was run.
