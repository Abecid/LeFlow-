# pusht — retrieval_repair_ttt completed run

The fixed four-checkpoint success curve is **34 / 39 / 37 / 38 /50**, selecting **39/50 at 10,000 updates**. The fresh policy reached 20,000 updates, consuming 1.870220 optimizer GPUh and 0.035844 evaluation GPUh. Train/eval wall time: 30.04 minutes.

[Online run](https://wandb.ai/attentionx2023/leflow-crossbench/runs/jvbbolss) is finished. The CPU source/data/case/checkpoint audit passes 245 runtime files, same case-plan hash at every checkpoint, finite model tensors and exact selected/milestone weights. Full [evidence](../reports/20261013-crossbench-baselines/runs/pusht/retrieval_repair_ttt/) preserves periodic metrics, all physical outcomes and checkpoint hashes.

| Update | Success /50 | Timeouts | Mean decision ms |
| --- | ---: | ---: | ---: |
| 5,000 | 34 | 0 | 55.21 |
| 10,000 | 39 | 0 | 54.62 |
| 15,000 | 37 | 0 | 54.59 |
| 20,000 | 38 | 0 | 54.30 |

Selected failure reasons: {'action_limit': 11}. Selected mean action count is 31.48; world transitions total 104,960. These are factual selected trajectories, not outcomes of rejected actions.

This benchmark uses released frozen LeWM features rather than MetaWorld V-JEPA. The latent cache retains 14,948 TRAIN episodes; 14,948 episodes provide 1,439,235 native-safe starts among 1,496,570 nominal starts. Raw out-of-box commands are preserved, not clipped or given fabricated labels. Factual history and complete goal prefixes respect native limits; all controllers share this rule. Validation cases remain unchanged. Full-source normalization/backbone exposure are documented in the campaign runbook.

One seed and50 repeatedly selected development cases are screening evidence, not independent final evaluation or native-method SOTA. Other controllers continue under the same ceilings; no extra seed, MetaWorld retraining or sealed test was run.

Visual evidence: [inspected contact sheet](../reports/20261013-crossbench-baselines/visual-review/pusht-retrieval_repair_ttt.png), containing all archived frames from the first ten registered cases. [Review scope and findings](../reports/20261013-crossbench-baselines/visual-review/README.md). This subset does not replace the complete 50-case reports.
