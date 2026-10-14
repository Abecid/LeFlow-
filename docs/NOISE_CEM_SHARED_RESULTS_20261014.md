# Frozen LeFlow noise-space search — measured development results

1/12 inference runs completed; zero training updates. These use retained PushT/Cube trained heads, not author checkpoints. Native Reacher/TwoRoom rows remain pending checkpoint access. All six new variants and the reused LeFlow reference share strict FP32, each task’s frozen checkpoint, original50 cases and50-action/10-controller-second allowance.

| Dataset | Method | Success /50 | Timeouts | Mean decision ms |
| --- | --- | ---: | ---: | ---: |
|pusht|noise_random_shell|15|0|144.93|

| Dataset | Method versus current LeFlow reference | Additional successes | Lost successes |
| --- | --- | ---: | ---: |
|pusht|noise_random_shell|6|3|

| Dataset | CEM vs matched random control | Additional successes | Lost successes |
| --- | --- | ---: | ---: |

Queue failure hold: False. Complete case reports, prediction-error traces, deadlines, candidates, full measured calls and checkpoint/frozen-state hashes are preserved in [the archive](reports/20261014-noise-cem/).

All scalar outcomes are actual native simulator success flags. Timings are measured, not flow-call counts presented as elapsed compute. Executed-prefix prediction errors are factual; rejected candidates have no known physical outcomes. Different sequential versus one-batch sampling does not promise identical pools across all later decisions. No claim of TTT, training improvement, original-checkpoint reproduction, SOTA or held-out final generalization is supported by this one-seed reused-development pilot. Strict-FP32 PushT is an explicit numerical setting versus prior default-TF32 records; old baseline files remain untouched. No stored trajectory is claimed visually reviewed until it is rendered and inspected.

[Registration and formulation](NOISE_CEM_SHARED_RUN_20261014.md). The unchanged reference is12/50 PushT and40/50 Cube. All6 variants cost3 times the baseline candidate-level flow/world work per decision; compare CEM with matched192-candidate random controls to isolate adaptive search.
