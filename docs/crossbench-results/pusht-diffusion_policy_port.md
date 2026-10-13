# pusht — diffusion_policy_port completed run

The fixed four-checkpoint success curve is **4 / 11 / 10 / 11 /50**, selecting **11/50 at 10,000 updates**. The fresh policy reached 20,000 updates, consuming 0.511090 optimizer GPUh and 0.033507 evaluation GPUh. Train/eval wall time: 9.33 minutes.

[Online run](https://wandb.ai/attentionx2023/leflow-crossbench/runs/v4gff79r) is finished. The CPU source/data/case/checkpoint audit passes 245 runtime files, same case-plan hash at every checkpoint, finite model tensors and exact selected/milestone weights. Full [evidence](../reports/20261013-crossbench-baselines/runs/pusht/diffusion_policy_port/) preserves periodic metrics, all physical outcomes and checkpoint hashes.

| Update | Success /50 | Timeouts | Mean decision ms |
| --- | ---: | ---: | ---: |
| 5,000 | 4 | 0 | 35.98 |
| 10,000 | 11 | 0 | 37.11 |
| 15,000 | 10 | 0 | 36.61 |
| 20,000 | 11 | 0 | 37.29 |

Selected failure reasons: {'action_limit': 39}. Selected mean action count is 44.36; world transitions total 0. These are factual selected trajectories, not outcomes of rejected actions.

This benchmark uses released frozen LeWM features rather than MetaWorld V-JEPA. The latent cache retains 14,948 TRAIN episodes; 14,948 episodes provide 1,439,235 native-safe starts among 1,496,570 nominal starts. Raw out-of-box commands are preserved, not clipped or given fabricated labels. Factual history and complete goal prefixes respect native limits; all controllers share this rule. Validation cases remain unchanged. Full-source normalization/backbone exposure are documented in the campaign runbook.

One seed and50 repeatedly selected development cases are screening evidence, not independent final evaluation or native-method SOTA. Other controllers continue under the same ceilings; no extra seed, MetaWorld retraining or sealed test was run.

Visual evidence: [inspected contact sheet](../reports/20261013-crossbench-baselines/visual-review/pusht-diffusion_policy_port.png), containing all archived frames from the first ten registered cases. [Review scope and findings](../reports/20261013-crossbench-baselines/visual-review/README.md). This subset does not replace the complete 50-case reports.
