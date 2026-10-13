# Native PushT/Cube campaign — measured development results

Sixteen authorized fresh runs; completed MetaWorld results are reused. Training source is frozen at `747f635ad36af2cbe3e2bbb0b7997efe9caccaa1`. All runs use seed3072/global64, 20k updates OR8 optimizer-GPUh, four fixed50-case evaluations,50 primitive actions AND10 cumulative controller seconds. Native success and backbone differ from MetaWorld; do not pool them as SOTA.

| Dataset | Method | State | Success curve /50 | Best measured |
| --- | --- | --- | --- | ---: |
| pusht | execution_revision | training / periodic evaluation | pending | pending |
| pusht | leflow | training / periodic evaluation | 12 / 8 / 9 / 12 | 12 |
| pusht | retrieval_repair_ttt | training / periodic evaluation | 34 / 39 / 37 | 39 |
| pusht | vanilla_action_flow | completed, audited | 7 / 13 / 15 / 17 | 17 |

The remaining registered methods are queued until their dataset is prepared. No missing score is estimated. All archived case outcomes are actual native simulator success flags. CPU/synthetic checks alone are not benchmark results.

Source/data/protocol details: [launch record](CROSSBENCH_RUNS_20261013.md). Complete/interim training logs and all physical milestone reports: [archive](reports/20261013-crossbench-baselines/).
