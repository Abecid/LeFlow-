# Frozen LeFlow P&P transfer — measured development results

6/12 inference runs completed; zero training updates. These use retained PushT/Cube trained heads, not author checkpoints. Native Reacher/TwoRoom rows remain pending checkpoint access. All six variants share strict FP32, each task’s frozen checkpoint, original50 cases and50-action/10-controller-second allowance.

| Dataset | Method | Success /50 | Timeouts | Mean decision ms |
| --- | --- | ---: | ---: | ---: |
|cube|leflow|40|0|57.39|
|cube|leflow_more_samples|37|0|61.24|
|cube|leflow_more_steps|37|0|72.43|
|pusht|leflow|12|0|50.43|
|pusht|leflow_more_samples|16|0|53.75|
|pusht|leflow_more_steps|11|0|64.51|

| Dataset | Method versus current LeFlow reference | Additional successes | Lost successes |
| --- | --- | ---: | ---: |
|cube|leflow|0|0|
|cube|leflow_more_samples|1|4|
|cube|leflow_more_steps|0|3|
|pusht|leflow|0|0|
|pusht|leflow_more_samples|4|0|
|pusht|leflow_more_steps|2|3|

Queue failure hold: False. Complete case reports, prediction-error traces, deadlines, candidates, full measured calls and checkpoint/frozen-state hashes are preserved in [the archive](reports/20261014-self-refine/).

All scalar outcomes are actual native simulator success flags. Timings are measured, not flow-call counts presented as elapsed compute. Executed-prefix prediction errors are factual; rejected candidates have no known physical outcomes. Different candidate counts do not preserve nested pools across all later decisions. No claim of TTT, training improvement, original-checkpoint reproduction, SOTA or held-out final generalization is supported by this one-seed reused-development pilot. Strict-FP32 PushT is an explicit numerical setting versus prior default-TF32 records; old baseline files remain untouched. No stored trajectory is claimed visually reviewed until it is rendered and inspected.

[Registration and formulation](SELF_REFINE_SHARED_RUN_20261014.md).
