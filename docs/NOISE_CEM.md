# JEPA-scored noise-space CEM for LeFlow

## Status

Implemented in `leflow_refinement/noise_search.py` and integrated into both
`NativeRuntime.plan` and `RefinementController.plan`. The existing native and
shared-execution evaluation CLIs accept the new variants. This is an inference-only
extension of LeFlow, not a new trained model or a reproduction of LeWAM's joint
architecture.

**Validation:** 66 new CPU numerical/adapter tests pass locally and in GitHub
Actions. Existing refinement regression tests and the existing released-model
interface job also pass at commit `c88923df2433cc8f70a27fe1a09c99c2c5629c41`.
CI run: https://github.com/Abecid/LeFlow-/actions/runs/38525474899 .
The validation receipt is [NOISE_CEM_VALIDATION.json](NOISE_CEM_VALIDATION.json).

These are software/interface checks, not new robot task-success results. No
training, noise-search benchmark campaign, hyperparameter sweep or sealed-test
access was launched in this implementation session.

## Algorithm and exact conventions

The input noise is `[B,N,H-1,D]`: B environments, N candidates per round, H latent
transitions and D coordinates per encoded state. D comes from the checkpoint's
encoded states. Existing adapters infer action widths and enforce their existing
normalization/clipping. The complete frozen Euler flow integrates each noise
proposal, and the frozen inverse decoder maps the resulting path into actions.
The frozen world model predicts their outcomes. The score is the existing
**predicted terminal latent MSE**, not distance from the clamped proposed endpoint.
There is no new critic, latent-gradient correction or action averaging.

Each decision starts with independent per-environment Gaussian parameters
`mean=0`, `std=1`. Round r proposes `raw = mean + std * epsilon`. The E candidates
with lowest predicted terminal MSE define elite mean and population variance.
The updates are:

```text
mean_next = alpha * mean + (1-alpha) * elite_mean
std_next  = sqrt(alpha * std**2 + (1-alpha) * elite_variance)
std_next  = max(std_next, min_std)  # coordinatewise
```

The configurable initial defaults are N=64, R=3, E=8, 16 flow steps, H=5,
`alpha=0.25` (weight on OLD moments) and `min_std=0.1` (variance floor 0.01).
E, alpha and the floor are untuned implementation defaults, not claimed
paper-optimal settings. The last round is evaluated without an unused refit.

For shell variants, reshape the complete interior into m=(H-1)*D coordinates and
send `sqrt(m) * raw / norm(raw)` through the flow. This is ONE norm per candidate,
not a norm per temporal node. Zero vectors are redrawn, with a finite retry limit.
Elite fitting uses the RAW proposals and their corresponding scores under this
projection. The selected raw proposal and the actual projected input are both
returned. Shell projection changes the input distribution and does not guarantee
physical feasibility.

Keep the best evaluated candidate across all completed rounds. Exact ties retain
the earlier candidate. Return its cached decoded actions, without another rollout
or averaging different actions. Incumbent predicted cost cannot increase across
rounds under the same score; physical success is not guaranteed to improve.
Distributions reset at every decision. Model parameters and buffers are not
updated. No environment outcomes are supplied to the search.

## Named comparison variants

| Name | Populations | Shell projection | Fit elite moments |
| --- | --- | --- | --- |
| `noise_cem` | 3 x 64 | No | Yes |
| `noise_cem_shell` | 3 x 64 | Yes | Yes |
| `noise_random` | 3 x 64 | No | No |
| `noise_random_shell` | 3 x 64 | Yes | No |
| `noise_bon192` | 1 x 192 | No | No |
| `noise_bon192_shell` | 1 x 192 | Yes | No |

All use 16 flow steps and H=5 by default: **3,072 candidate-level flow evaluations
and 960 predicted world transitions per environment**, with zero backward passes.
For B environments, multiply these candidate-level totals by B. A 3 x 64 search
uses 48 flow batch calls versus 16 for a 192-candidate batch. Equal operation
counts do not establish equal latency or memory usage.

`noise_random` is the matched sequential-batching/random-draw control.
`noise_bon192` separately measures the large-batch operating point. Gaussian RNG
kernels may consume random numbers differently for different batch shapes, so a
shared seed is not a claim of bit-identical pools across these batching regimes.
Projected independent controls isolate the effect of changing noise magnitude.

The old six `variants()` and native `--variant all` remain unchanged. New variants
require an explicit name or `variants(include_noise=True)` in Python. Existing
P&P, world-guided P&P and gradient-only controls are preserved. Existing registered
campaigns are not silently expanded.

## Run using existing checkpoints

Use an isolated evaluation directory and the repository's pinned GPU environment.
Do not edit an immutable active training/evaluation checkout. Keep checkpoint,
world, cache, case seeds, action limits, execution prefix, hardware and precision
identical within each comparison. The evaluators retain source/checkpoint/split
identity checks; do not bypass them to reuse incompatible results.

### Shared execution: compare against existing PushT/Cube baselines

Set `CACHE` to the existing dataset cache containing `manifest.json`, `cache.h5`,
and the original source metadata. Set `CHECKPOINT` to that dataset's frozen
LeFlow `best.pt`, and `OUTPUT` to a NEW evaluation directory. Then run:

```bash
: "${CACHE:?Set the existing dataset cache directory}"
: "${CHECKPOINT:?Set the matching frozen LeFlow checkpoint}"
: "${OUTPUT:?Set a new result directory}"
python -m leflow_refinement.crossbench \
  --cache "$CACHE" --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" --variant noise_cem_shell --device cuda:0
```

Use separate output directories for `noise_cem`, `noise_random`,
`noise_random_shell`, `noise_bon192` and `noise_bon192_shell` as required.
The existing controller deadline and five-primitive execution prefix remain in
force. A deadline exception is propagated: the method never returns an old plan
as permission to execute after the deadline.

### Native author-format checkpoint protocol

Use the existing native evaluator arguments. `SOURCE` is the source HDF5,
`SPLIT` is the explicit prepared validation split, `CHECKPOINT` is an author-format
LeFlow planner checkpoint, `WORLD` is its matching LeWM checkpoint, and `OUTPUT`
is a new result root. Set `TASK` to pusht, cube, tworoom or reacher.

```bash
python -m leflow_refinement.native_eval \
  --task "$TASK" --source "$SOURCE" --split-file "$SPLIT" \
  --checkpoint "$CHECKPOINT" --world-checkpoint "$WORLD" \
  --output "$OUTPUT" --variant noise_cem_shell --device cuda:0
```

This retains the existing native execution protocol, which differs from the
shared-execution protocol. Do not pool the two families or compare a new inference
method to a differently trained checkpoint as a search-only causal experiment.
The implementation does not download or create missing author checkpoints.

## Telemetry and tests

Existing flow, inverse, world and backward counters remain authoritative. Extra
`noise_search` telemetry records completed rounds, per-round best and incumbent
costs, mean input-noise norm, minimum fitted standard deviation and zero-vector
resampling counts. Native telemetry includes planner time; the existing evaluators
record full controller timing and actual simulator success when explicitly run.
Returned `all_costs` is `[B,R*N]`; `selected_indices` indexes that concatenated
round-major pool. `selected_noise` is `[B,H-1,D]`, the exact flow input for the
returned plan. Selected actions are `[B,H,block_action_dim]`.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/test_noise_cem*.py
python -m leflow_refinement.crossbench --help
python -m leflow_refinement.native_eval --help
```

The new tests cover one-round Euler parity, both action-block dimensions (10 and
25), batched environments, elite moments and variance floors, shell normalization
and zero handling, deterministic RNG, cached incumbents, unchanged weights and
BatchNorm buffers, nested inference mode, exact work counts, deadline propagation,
invalid inputs, CLI selection and preservation of the old six-method campaign.
