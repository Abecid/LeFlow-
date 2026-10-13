# cube — diffusion_policy_port completed run

The fixed four-checkpoint success curve is **28 / 32 / 36 / 38 /50**, selecting **38/50 at 20,000 updates**. The fresh policy reached 20,000 updates, consuming 0.513880 optimizer GPUh and 0.384184 evaluation GPUh. Train/eval wall time: 14.62 minutes.

[Online run](https://wandb.ai/attentionx2023/leflow-crossbench/runs/ey27d26g) is finished. The CPU source/data/case/checkpoint audit passes 245 runtime files, same case-plan hash at every checkpoint, finite model tensors and exact selected/milestone weights. Full [evidence](../reports/20261013-crossbench-baselines/runs/cube/diffusion_policy_port/) preserves periodic metrics, all physical outcomes and checkpoint hashes.

| Update | Success /50 | Timeouts | Mean decision ms |
| --- | ---: | ---: | ---: |
| 5,000 | 28 | 0 | 36.15 |
| 10,000 | 32 | 0 | 35.97 |
| 15,000 | 36 | 0 | 35.88 |
| 20,000 | 38 | 0 | 35.87 |

Selected failure reasons: {'action_limit': 12}. Selected mean action count is 20.98; world transitions total 0. These are factual selected trajectories, not outcomes of rejected actions.

This benchmark uses released frozen LeWM features rather than MetaWorld V-JEPA. The latent cache retains 8,000 TRAIN episodes; 8,000 episodes provide 1,408,000 native-safe starts among 1,408,000 nominal starts. Raw out-of-box commands are preserved, not clipped or given fabricated labels. Factual history and complete goal prefixes respect native limits; all controllers share this rule. Validation cases remain unchanged. Full-source normalization/backbone exposure are documented in the campaign runbook.

One seed and50 repeatedly selected development cases are screening evidence, not independent final evaluation or native-method SOTA. Other controllers continue under the same ceilings; no extra seed, MetaWorld retraining or sealed test was run.

Visual evidence: [inspected contact sheet](../reports/20261013-crossbench-baselines/visual-review/cube-diffusion_policy_port.png), containing all archived frames from the first ten registered cases. [Review scope and findings](../reports/20261013-crossbench-baselines/visual-review/README.md). This subset does not replace the complete 50-case reports.
