# pusht — action_flow_search completed run

The fixed four-checkpoint success curve is **37 / 29 / 24 / 31 /50**, selecting **37/50 at 5,000 updates**. The fresh policy reached 20,000 updates, consuming 0.532956 optimizer GPUh and 0.041907 evaluation GPUh. Train/eval wall time: 9.81 minutes.

[Online run](https://wandb.ai/attentionx2023/leflow-crossbench/runs/xdhwrl18) is finished. The CPU source/data/case/checkpoint audit passes 245 runtime files, same case-plan hash at every checkpoint, finite model tensors and exact selected/milestone weights. Full [evidence](../reports/20261013-crossbench-baselines/runs/pusht/action_flow_search/) preserves periodic metrics, all physical outcomes and checkpoint hashes.

| Update | Success /50 | Timeouts | Mean decision ms |
| --- | ---: | ---: | ---: |
| 5,000 | 37 | 0 | 73.31 |
| 10,000 | 29 | 0 | 68.25 |
| 15,000 | 24 | 0 | 68.45 |
| 20,000 | 31 | 0 | 68.41 |

Selected failure reasons: {'action_limit': 13}. Selected mean action count is 31.24; world transitions total 156,480. These are factual selected trajectories, not outcomes of rejected actions.

This benchmark uses released frozen LeWM features rather than MetaWorld V-JEPA. The latent cache retains 14,948 TRAIN episodes; 14,948 episodes provide 1,439,235 native-safe starts among 1,496,570 nominal starts. Raw out-of-box commands are preserved, not clipped or given fabricated labels. Factual history and complete goal prefixes respect native limits; all controllers share this rule. Validation cases remain unchanged. Full-source normalization/backbone exposure are documented in the campaign runbook.

One seed and50 repeatedly selected development cases are screening evidence, not independent final evaluation or native-method SOTA. Other controllers continue under the same ceilings; no extra seed, MetaWorld retraining or sealed test was run.

Visual evidence: [inspected contact sheet](../reports/20261013-crossbench-baselines/visual-review/pusht-action_flow_search.png), containing all archived frames from the first ten registered cases. [Review scope and findings](../reports/20261013-crossbench-baselines/visual-review/README.md). This subset does not replace the complete 50-case reports.
