# Frozen LeFlow P&P transfer — measured development results

12/12 inference runs completed; zero training updates. These use retained PushT/Cube trained heads, not author checkpoints. Native Reacher/TwoRoom rows remain pending checkpoint access. All six variants share strict FP32, each task’s frozen checkpoint, original50 cases and50-action/10-controller-second allowance.

| Dataset | Method | Success /50 | Timeouts | Mean decision ms |
| --- | --- | ---: | ---: | ---: |
|cube|guidance_only|36|0|396.13|
|cube|guided_pnp|37|0|386.14|
|cube|leflow|40|0|57.39|
|cube|leflow_more_samples|37|0|61.24|
|cube|leflow_more_steps|37|0|72.43|
|cube|pnp|38|0|74.80|
|pusht|guidance_only|12|0|392.71|
|pusht|guided_pnp|4|0|392.90|
|pusht|leflow|12|0|50.43|
|pusht|leflow_more_samples|16|0|53.75|
|pusht|leflow_more_steps|11|0|64.51|
|pusht|pnp|7|0|64.46|

| Dataset | Method versus current LeFlow reference | Additional successes | Lost successes |
| --- | --- | ---: | ---: |
|cube|guidance_only|0|4|
|cube|guided_pnp|1|4|
|cube|leflow|0|0|
|cube|leflow_more_samples|1|4|
|cube|leflow_more_steps|0|3|
|cube|pnp|2|4|
|pusht|guidance_only|4|4|
|pusht|guided_pnp|1|9|
|pusht|leflow|0|0|
|pusht|leflow_more_samples|4|0|
|pusht|leflow_more_steps|2|3|
|pusht|pnp|2|7|

Queue failure hold: False. Complete case reports, prediction-error traces, deadlines, candidates, full measured calls and checkpoint/frozen-state hashes are preserved in [the archive](reports/20261014-self-refine/).

All scalar outcomes are actual native simulator success flags. Timings are measured, not flow-call counts presented as elapsed compute. Executed-prefix prediction errors are factual; rejected candidates have no known physical outcomes. Different candidate counts do not preserve nested pools across all later decisions. No claim of TTT, training improvement, original-checkpoint reproduction, SOTA or held-out final generalization is supported by this one-seed reused-development pilot. Strict-FP32 PushT is an explicit numerical setting versus prior default-TF32 records; old baseline files remain untouched. No stored trajectory is claimed visually reviewed until it is rendered and inspected.

[Registration and formulation](SELF_REFINE_SHARED_RUN_20261014.md).

## Completion and scope

All12 transfers completed600 physical episodes, passed the292-file frozen-source
audit and unchanged-policy/world checks, and finished their12 online runs. No
owned GPU evaluation remains. All failures reached50 actions; there were no
controller timeouts. Full first-ten measured trajectory archives are preserved
for each run, but no new contact sheet is delivered or claimed visually reviewed.

The strict-FP32 references reproduce every retained LeFlow case’s prior success
flag:12/50 PushT and40/50 Cube. Those historical records remain unchanged.
More candidates improves PushT to16/50; world-guided P&P regresses to4/50 and
37/50. No world-guidance or P&P variant improves Cube’s reference in this pilot.
[Failure and behavior analysis](SELF_REFINE_SHARED_FAILURE_ANALYSIS_20261014.md)
records paired regressions and the mismatch between lower surrogate errors and
actual task success.

The original four-task author-checkpoint benchmark remains unrun; these are
retained-checkpoint adaptations. Author-checkpoint access or a separately
authorized shared native-baseline training plan is still needed for that study.
