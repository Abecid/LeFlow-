# First controller-grounded iteration

**Completed:**20,000 updates, all four registered validations, selected5k79/104.
See [final results](CONTROLLER_GROUNDED_RESULTS.md), including complete compute,
checkpoint provenance, failure analysis and comparison limits. Sections below
preserve the launch formulation and historical recovery record.

Authorized October 9, 2026: implement, train one candidate from scratch, compare
with saved baselines and inspect failure modes. This supersedes the research-only
hold for this isolated iteration. No baseline reruns, seeds, automatic variants,
ablations or final tests are included.

## Central mechanism

**Choose a subgoal after evaluating the bounded local controller's response to
it, and execute the prefix of that same response.** This closes the proposed
design's action handoff: no coarse route scored with one action sequence is
followed by an independently substituted sequence after target selection.

This first implementation uses direct, batched controller evaluation. It does
not train a second model to imitate imagined controller outcomes and call them
real reachability labels. It also omits the separate recorded-action coarse
predictor: it would model an action sequence the controller need not execute.
The hierarchy consists of observed-route target selection and short local
control. Efficient learned approximation of this response can be considered
later only if the first results justify it.

The new learned component is a four-component Gaussian-mixture policy over
five control steps (40 action coordinates). Its current/target/final-goal input
uses the existing 32×1024 frozen spatial features; a width256/depth3 transformer
proposes multiple action modes. The encoder, normalization and selected fine
world remain frozen. This is a proposed direction, not a proven novelty or
performance claim. The [contribution audit](CONTRIBUTION_AUDIT_20261009.md)
continues to define the relevant prior-art boundaries.

## Fixed algorithm

1. Build a route memory only from the same 6,222 successful training experts.
   Route starts use a five-control-step grid before or at first success;
   spans are 5, 10, 20, 40 or 60 steps with valid recorded actions. No evaluation
   states, task labels, proprioception or oracle goal duration enter retrieval.
2. Shortlist256 routes with a fixed random projection: individually normalize
   each spatial token, project1024→8 with a seed3072 Gaussian matrix, concatenate
   the32 token projections and divide by sqrt(32). This256D index is an explicit
   approximate retrieval choice. Rerank using exact original-token cosine cost
   `R_j = d(z, recorded_start_j) + d(goal, recorded_end_j)`.
3. Retain seven distinct first-waypoint targets. Add the final goal as an eighth
   target, with route cost `d(z,goal)`. That option requires no retrieved route.
4. For each target use four action candidates: three policy proposals (including
   its highest-probability mean) and a recorded first chunk. The direct-goal
   option uses its policy mean in the recorded slot. Three total candidate
   batches perform initial scoring and two CEM refinements. Each target retains
   its best candidate, uses two elites, and has a0.05 standard-deviation floor.
5. Score a five-step continuous fine rollout using
   `J_j(U)=0.5*d(predicted_state_4,q_j)+0.5*d(predicted_state_5,q_j)`.
   Choose the target and its actual refined actions jointly by minimizing
   `R_j + min_{U returned by local search} J_j(U)`. The units are the same token
   cosine distances; equal weights are fixed defaults, not swept optima.
6. Execute exactly the first control block (two primitive actions) from that
   selected sequence. Observe and repeat. All encoding, retrieval, search and
   logging within the decision are charged against the same cumulative10sec.

There are480 fine candidate transitions per decision:8 targets×4 candidates×
5 steps×3 batches. There is no additional60-step fine rollout, generated latent
path, coarse predictor, learned temporal critic, or optimizer substitution.
The fixed total candidate count is distributed across targets; per-target
search is consequently narrower. That tradeoff must be assessed from results.

## Training and label meaning

Use the exact existing task/episode/window sampling draws for60-step windows,
seed3072 and global batch64. A separate deterministic stream selects a final-goal
offset from5/10/20/40/60 inside the sampled window. Local target is the recorded
state five control steps ahead; action supervision is the recorded first chunk.

The loss is per-coordinate mixture negative log likelihood plus0.1 times the
mean of observed-prefix and observed-endpoint prediction losses. The likelihood
posterior of the demonstrated action selects the supervised mixture component.
Roll that component's predicted mean actions through the frozen world; compare
its first and fifth predicted states with the actual recorded states at those
times, using the existing MSE-plus-token-cosine loss. Execution supervision
ramps from zero at update1000 to full weight at2000. Gradients pass through the
frozen world into the action policy, never updating world/encoder parameters.

These observed states are targets for generated actions, not outcome labels
claiming those generated actions were physically executed. The likelihood term
anchors actions to recorded behavior. The model-based loss can still exploit
world errors; policy likelihood, predicted/observed progress and actual task
success must be reported separately. The first version estimates local control
response in the frozen model, not full closed-loop real-world reachability.

## Resource and comparison contract

- One new policy, trained from scratch. Seed3072; at most20,000 updates/global64
  and28,800 aggregate optimization GPU-seconds, whichever is reached first.
  All8 A800s train with physical batch8; the aggregate ceiling is3600 seconds
  at8 GPUs. Optimizer-boundary overrun, if any, is explicitly recorded.
- Validate the same104 cases at5k/10k/15k/20k when reached, using all8 GPUs with
  training paused. Save earliest tied best by task success. Preserve and restore
  training RNG across evaluation. Initialization/indexing/preflights and
  evaluation have separate charged ledger entries; no new training is hidden
  as preparation. Per-run ceilings are equal, not actual FLOPs or realized time.
- Reuse selected world SHA256
  `ae3f910da43f0a43dd65945394444a0b11719ed042cdacffacc2faaf824ec5a3`.
  Verify unchanged episode entries, normalization, encoder and world provenance
  while registering a separate execution protocol/manifest. Existing checkpoints
  and execution source remain immutable.
- Compare saved same-world CEM22/104, repaired flow17/104 and historical
  LeFlow27/104, HWM8/104 and old CEM7/104. Historical representations/worlds and
  sampler defects remain confounds. Baselines are not rerun. Test remains sealed.
- A compact memory index reuses existing latents; no encoder recache or new
  simulator trajectories. Stage the raw half-precision grid on each GPU and
  normalize gathered entries in float32. Retrieval approximation and the new
  controller are algorithm changes, not claimed lossless optimizations.

## Failure evidence registered before launch

Log every chosen training route/target, each target's route and controller cost,
whether the controller response changes the retrieval-only choice, refinement
round, action diversity, and predicted progress after the executed prefix.
At the next observation log that prefix's actual prediction error and progress
toward the previously selected target. Do not treat a later observation after
replanning as the outcome of an unexecuted five-step chunk.

Preserve trajectories for six fixed existing validation cases: coffee-button0,
reach3, door-close0, dial-turn0, drawer-close4 and handle-press0. Saving occurs
outside the controller clock and remains part of evaluation occupancy. Inspect
these for physical failure descriptions; do not infer contact failures solely
from timeout. Identify source coverage, unstable target switching, unhelpful
policy modes, model optimism, insufficient search per target and latency limits.
All are hypotheses until supported by observations.

## Literature and implementation provenance

The [broader review](NEXT_METHOD_FORMULATION.md) records pinned upstream code.
This implementation is new project code; it is not a claimed reproduction.
Anchored Planning motivates observed target support; Hi-LeWM motivates checking
the high/low interface; SAGE motivates a one-pass mixture proposal; Planning
Limits motivates a local fine horizon. TD-JEPA's manipulation findings motivate
retaining spatial cost. LeFlow supplies the proposal/refinement precedent.
HAC and later controller-aware HRL limit the novelty claim; no generic invention
of subgoal reachability or planning consistency is claimed.

Historical prelaunch status: source implemented, local compilation passed. Local behavioral
tests could not import h5py; they will run in the established server environment.
Real-data bounded forward/backward and full encoder/controller latency checks
are required before launching optimization. Results and exact source identity
will be appended after verification.

## Verified launch

All four behavioral tests passed in the established server environment. Eight
sampled window/action checks and four raw-bank bitwise checks passed. The
training-data-only preflight measured70.97ms average full controller latency,
9.46GiB peak GPU memory and zero optimizer updates/simulator episodes. See
[preflight evidence](reports/20261009-controller-grounded/preflight.json).
CPU index construction was changed from serialized HDF5 threads to16 processes
and exact dense-read/subsample operations; source features are unchanged.

Source `74a0a715cc393ba26285deb4571006ee7bc4d286` is frozen in the isolated
server checkout. Eight-GPU training began October9 at19:25UTC, with
[W&B run s12cr0yl](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/s12cr0yl).
The data/window/index evidence is preserved beside the preflight report.
First evaluation was pending at the initial launch report; all four rounds are
now complete, as documented in the final results linked above.

### Renderer recovery at update5,000

The first launch omitted the existing Mesa environment, so evaluator startup
failed before any scored episode completed. This was a launcher error. The
corrected [launcher](reports/20261009-controller-grounded/launch.sh) restores the
exact renderer used for dataset collection and checks one training reset image
bitwise before launching. That CPU-only check passed. The same frozen source,
5,000-update checkpoint, optimizer/RNG state and W&B run resumed at19:36UTC;
no update was discarded or repeated. Optimization usage remained0.685226GPUh.

[Recovery accounting](reports/20261009-controller-grounded/renderer-recovery.json)
preserves the failure and timestamps. The first attempt reserved2,949.440GPU
seconds, including2,466.815 measured optimization GPU-seconds. The remaining
482.625GPU-seconds is an upper bound for startup/checkpoint IO/failed evaluator
initialization/shutdown occupancy, recorded separately from optimization and
completed validation. The first registered evaluation resumes before update5,001.
