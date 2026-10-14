# Frozen LeFlow noise-space search — measured development results

11/12 inference runs completed; zero training updates. These use retained PushT/Cube trained heads, not author checkpoints. Native Reacher/TwoRoom rows remain pending checkpoint access. All six new variants and the reused LeFlow reference share strict FP32, each task’s frozen checkpoint, original50 cases and50-action/10-controller-second allowance.

| Dataset | Method | Success /50 | Timeouts | Mean decision ms |
| --- | --- | ---: | ---: | ---: |
|cube|noise_bon192|36|0|81.57|
|cube|noise_cem|41|0|156.87|
|cube|noise_cem_shell|39|0|156.62|
|cube|noise_random|40|0|158.13|
|cube|noise_random_shell|39|0|157.75|
|pusht|noise_bon192|13|0|66.70|
|pusht|noise_bon192_shell|11|0|65.78|
|pusht|noise_cem|9|0|143.05|
|pusht|noise_cem_shell|10|0|143.12|
|pusht|noise_random|12|0|142.34|
|pusht|noise_random_shell|15|0|144.93|

| Dataset | Method versus current LeFlow reference | Additional successes | Lost successes |
| --- | --- | ---: | ---: |
|cube|noise_bon192|0|4|
|cube|noise_cem|3|2|
|cube|noise_cem_shell|2|3|
|cube|noise_random|2|2|
|cube|noise_random_shell|3|4|
|pusht|noise_bon192|6|5|
|pusht|noise_bon192_shell|4|5|
|pusht|noise_cem|4|7|
|pusht|noise_cem_shell|5|7|
|pusht|noise_random|3|3|
|pusht|noise_random_shell|6|3|

| Dataset | CEM vs matched random control | Additional successes | Lost successes |
| --- | --- | ---: | ---: |
|pusht|noise_cem vs noise_random|3|6|
|pusht|noise_cem_shell vs noise_random_shell|3|8|
|pusht|noise_cem vs noise_bon192|3|7|
|pusht|noise_cem_shell vs noise_bon192_shell|3|4|
|cube|noise_cem vs noise_random|3|2|
|cube|noise_cem_shell vs noise_random_shell|2|2|
|cube|noise_cem vs noise_bon192|6|1|

Queue failure hold: False; 9/12 processes have exited after online log synchronization. Complete case reports, prediction-error traces, deadlines, candidates, full measured calls and checkpoint/frozen-state hashes are preserved in [the archive](reports/20261014-noise-cem/).

All scalar outcomes are actual native simulator success flags. Timings are measured, not flow-call counts presented as elapsed compute. Executed-prefix prediction errors are factual; rejected candidates have no known physical outcomes. Different sequential versus one-batch sampling does not promise identical pools across all later decisions. No claim of TTT, training improvement, original-checkpoint reproduction, SOTA or held-out final generalization is supported by this one-seed reused-development pilot. Strict-FP32 PushT is an explicit numerical setting versus prior default-TF32 records; old baseline files remain untouched. No stored trajectory is claimed visually reviewed until it is rendered and inspected.

[Registration and formulation](NOISE_CEM_SHARED_RUN_20261014.md). The unchanged reference is12/50 PushT and40/50 Cube. All6 variants cost3 times the baseline candidate-level flow/world work per decision; compare CEM with matched192-candidate random controls to isolate adaptive search.
