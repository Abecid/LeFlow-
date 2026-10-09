# Sampler repair and executable-progress diagnostics

> Latest authorization: the user has now explicitly requested training from
> scratch and a matched final comparison. See [REPAIRED_COMPARISON](REPAIRED_COMPARISON.md).
> The no-training boundary described below was valid during the repair stage;
> the original campaign remains held, while the new campaign is authorized.


All original training is complete. The final-test process tree was stopped at
02:45:55 UTC on October 9 under the user's revised instruction. Completed
training, all four validations per model, checkpoints and partially collected
test records remain preserved. No partial test outcomes were read for repairs.
No additional training, optimizer updates, seeds or benchmark runs were launched.

## Implemented repairs

The opt-in, **untrained** recipe is `config/flow_metaworld_repair.json`.

1. **Sampler:** both flow planners can predict a clean latent endpoint as a
   residual around full-dimensional observed start/goal bridge anchors. Sampling
   uses `z <- z + (clean-z)/(steps-i)` and assigns the clean prediction directly
   at the last step. This is Euler integration of
   `v=(clean-z)/(1-t)` with a numerically stable final update. Initial noise outside
   the width-256 output subspace is no longer forced to survive. Full-dimensional
   anchors preserve observation information without widening the transformer.
   The corresponding state loss is clean-endpoint MSE, equivalent to
   `(1-t)^2`-weighted velocity MSE; it is an explicit objective change, not a
   lossless conversion of the previously trained velocity head.
2. **Representation:** world, planner, inverse and coarse-model targets use the
   existing cached `image_goals` feature view for every timestep. Live states
   repeat the current frame exactly as goals do. This removes the causal-history
   versus static-image input convention mismatch. Cached pixels/features and
   training-only normalization values are preserved. Using this convention with
   dynamics trained on causal histories would be invalid, so protocol hashes
   continue to reject old checkpoints under the repaired configuration.
   This choice also removes temporal history from the state input; whether that
   tradeoff improves control has not been established without training.
3. **Ranking:** both flow methods can use one continuous model rollout from the
   actual observed state. Candidate cost is final-goal cosine distance plus 0.1
   times average waypoint discrepancy. Generated waypoints never reset the
   predicted state used for scoring. Existing controller-time checks remain
   active. Training consistency and deployment both use eight sampler steps in
   the new recipe.
4. **Execution control:** the repaired recipe disables training and final tests.
   New entry points also reject training/testing in a root carrying the user's
   hold record. Legacy defaults remain available to interpret historical models;
   the frozen execution checkout and old model semantics are unchanged.

The revised sampler removes the confirmed irreducible-noise mechanism; it does
not establish that the learned residual representation has sufficient capacity,
that straight-line bridge anchors are physically executable, or that newly
trained models would achieve better control. Those remain empirical questions.

## Verification without training

**29 tests passed in 2.68 seconds on CPU**, covering the actual 1,024-dimensional
latent/256-dimensional head, terminal noise removal, known clean endpoints at
1/4/8 steps, finite gradients through both flow objectives, consistent static
cache views for all four model types, identical live state/goal input construction,
continuous rollout scoring, launch holds, and existing method wiring. Gradient
checks perform backward passes but no optimizer updates.

The repair recipe preserves the original task lists, datasets, reset/split rules,
seed 3072, four comparison methods, global batch, 20,000-update/7,200-second caps,
four validation rounds, 104-case validation suite, 200-action limit, and 10-second
controller limit. Static feature selection uses already cached data. No repaired
checkpoint was trained or substituted into the existing comparison.

## Bounded simulator diagnostic

Preregistered in [REPAIR_PLAN](REPAIR_PLAN.md) before execution: one existing
validation reset per each of the 13 training tasks, eight candidates and both
flow methods. All **26 conditions / 208 candidate rollouts** completed. Each
candidate received the same short CEM refinement, then its five-control-step
(ten primitive actions) chunk was executed from the paired simulator reset.
The diagnostic compared original and continuous candidate scores against observed
subgoal distance and progress toward the static goal. Frozen original checkpoints
were loaded with their original configuration and code identity.

Elapsed wall time including launch was **122.873 seconds on four GPUs**, or
**0.136526 GPU-hours** (8.19 GPU-minutes), below the ten-minute cap. Optimization
updates: **zero**. This diagnostic cost is separate from the unchanged training
and registered validation ledgers.

| Mean within-condition Spearman correlation | Ours | LeFlow |
| --- | ---: | ---: |
| Original candidate score, oriented toward better actual goal progress | -0.035 | -0.00003 |
| Continuous score, oriented toward better actual goal progress | 0.205 | 0.004 |
| Predicted versus actual short-chunk goal progress | 0.031 | 0.099 |
| Predicted versus actual distance to the proposed subgoal | 0.899 | 0.875 |

The high last row should not be mistaken for useful planning: raw distances can
share target-dependent variation, and closeness to a proposed subgoal does not
imply progress toward the task goal. Goal-progress calibration is weak.

For ours, continuous ranking changed the selected candidate in 11/13 conditions.
Mean actual static-goal progress increased from 0.01455 to 0.01518 cosine-distance
units. Mean regret relative to the best tested candidate decreased from 0.00775
to 0.00712. The mean reward of selected chunks increased from 7.073 to 7.629.
LeFlow's selected candidate did not change in any condition. No selected chunk
for either method completed a task within these short diagnostic executions.

**Interpretation:** the original scores do not reliably rank executable goal
progress in this small diagnostic. Continuous scoring shows a modest directional
improvement for ours but is not validated as a reliable progress predictor. This
does not establish improved full-episode success or the performance of the
untrained repaired architecture. There is only one reset per task, selected from
validation; no confidence or SOTA claim is warranted.
Long-horizon plans can also require short-term detours, so weak first-chunk goal
progress correlation alone does not establish that every such plan is poor.

## Current boundary

Code repairs, regression checks and the authorized bounded diagnostic are done.
Full testing remains held. No further training, diagnostic sweep, ablation or
comparison is queued. Testing trained repaired models would require separately
authorized runs under the same original per-method limits and exactly the same
train/evaluation sets; ours must never receive extra optimization relative to
the comparison methods.

Evidence: [summary](reports/20261007-joint-flow/repair-validation-20261009/summary.json),
[allocation and measured runtime](reports/20261007-joint-flow/repair-validation-20261009/completion.json),
four `rank-*.json` records in that directory, and
[the hold record](reports/20261007-joint-flow/post-training-hold.json).
