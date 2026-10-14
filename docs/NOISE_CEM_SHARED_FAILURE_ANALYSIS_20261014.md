# Noise-space search: failure and behavior analysis

12/12 full physical evaluations are available; zero optimizer updates. All methods reuse the frozen LeFlow head and original development cases/limits. Full method and paired success tables are in [the result report](NOISE_CEM_SHARED_RESULTS_20261014.md).

|Task|Variant|Success /50|ms/decision|Median selected terminal MSE|Median factual prefix prediction MSE|
|---|---|---:|---:|---:|---:|
|cube|noise_bon192|36|81.57|0.00972|0.00702|
|cube|noise_bon192_shell|39|81.99|0.00785|0.00769|
|cube|noise_cem|41|156.87|0.00593|0.00658|
|cube|noise_cem_shell|39|156.62|0.00777|0.00621|
|cube|noise_random|40|158.13|0.00875|0.00656|
|cube|noise_random_shell|39|157.75|0.00931|0.00702|
|pusht|noise_bon192|13|66.70|0.02330|0.00809|
|pusht|noise_bon192_shell|11|65.78|0.02506|0.00785|
|pusht|noise_cem|9|143.05|0.01781|0.00806|
|pusht|noise_cem_shell|10|143.12|0.01704|0.00783|
|pusht|noise_random|12|142.34|0.02588|0.00809|
|pusht|noise_random_shell|15|144.93|0.02493|0.00798|

The measured outcome is mixed and does not support a reliable CEM improvement over LeFlow or matched random search. Unprojected CEM gives9/50 PushT and41/50 Cube versus12/40 for LeFlow and12/40 for sequential random search. Shell CEM gives10/39 versus15/39 for matched shell-random search. CEM versus its sequential control swaps3 wins/6 losses on PushT and3/2 on Cube; shell CEM swaps3/8 on PushT and2/2 on Cube. The one-case Cube gain is insufficient for a general superiority claim.

All600 cases have zero controller timeouts; every failure exhausts the50-action allowance. This regression therefore is not explained by deadline truncation. The CEM rounds improve their own predicted cost: the median within-decision reduction is28.6% on PushT and30.3% on Cube, compared with8.8% and13.8% for independent sequential random search. Adaptive proposals are usually selected, but optimizing this surrogate more strongly does not consistently improve completion. No rejected action was physically executed, so this does not isolate world-model error, proposal quality or goal representation as the cause.

CEM decisions take143ms PushT and157ms Cube, roughly2.8 times the reused reference. A192-candidate batch takes66–67ms PushT and82ms Cube, about1.3–1.4 times reference latency, with the same nominal3072/960 candidate work. That is a measured batching advantage, not proof of identical random pools or better outcomes.


The retained LeFlow reference is12/50 PushT at50.43ms/decision and40/50 Cube at57.39ms/decision. All six search variants spend3 times its candidate-level flow/world work per decision; the192-candidate random controls are the correct adaptive-search comparisons. Equal candidate work does not imply equal observed latency. No trained weight, task representation or action rule is changed.

For unprojected CEM and sequential random sampling, every first-round first-decision best cost is bitwise identical to the retained reference on each completed task. Later rounds retain an evaluated incumbent, and recorded incumbent costs never increase. Subsequent decisions visit different physical states and may use different random pools; they are not counterfactual comparisons. Shell projection and one-batch192 sampling do not promise an identical first pool.

A non-increasing predicted cost does not guarantee task success. Candidate gradients, TTT or new critics are absent. Lower selected cost across separate methods may reflect different visited states or exploitation of the surrogate; these runs cannot distinguish those causes or reveal the physical outcome of rejected candidates. Prefix-error medians likewise use each method’s own trajectories and numbers of observed prefixes, and are diagnostic rather than independent accuracy experiments. Final prefixes ending the episode are not observed by a next planner decision.

CPU185 numerical/adapter/regression checks and both real-TRAIN checkpoint gates passed before launch. Real sampler/RNG parity, action-unit bounds, exact3072/960 work counts, shell norms, deadline propagation and frozen-state checks passed. The completed-run audit separately verifies all cases, weights, source and process/log completion. No new sampler/indexing bug has been demonstrated. These checks do not establish that latent terminal MSE ranks physically successful plans.

All raw reports, candidate-cost distributions, selected round and noise-scale telemetry, actual executed actions, reset audits, paired successes/regressions and factual prefix errors are preserved. All50 trajectory archives per run remain on the server; their full inventory is published after completion, with first10 archives per run in Git. No trajectory figure is delivered or claimed visually inspected. This one-seed reused-development pilot supports a screening comparison, not a SOTA, novelty or held-out-generalization claim.
