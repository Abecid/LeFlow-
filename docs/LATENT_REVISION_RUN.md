# Execution-error-conditioned latent revision: first candidate

**Completed October 9, 2026.** All four registered validations are finished;
the selected 20k checkpoint scores 77/104. See the
[completed comparison and failure analysis](LATENT_REVISION_RESULTS.md).
The registration and launch record below are preserved as historical evidence.

October 9, 2026. The user explicitly authorized implementing and training one
fresh candidate under the existing method/data/evaluation allowance. This
document registers the actual formulation before its first validation result.
No baseline rerun, extra seed, automatic follow-up variant or final test is
part of this iteration.

## Central mechanism

Use recent observed model errors to condition a compact latent workspace that
changes action proposals and corrects optimistic controller costs. The frozen
world still predicts physical JEPA states; the workspace contains internal
planning vectors. A four-slot, width-256 shared attention block performs four
iterations at inference. It repeatedly attends to current/target/goal features
and the last four **already observed** execution residuals. This is latent
deliberation over a causal history, not generation of imagined reachable states.

The strongest measured previous pipeline (79/104 validation) supplies the
unchanged route bank recipe, spatial encoder, fine world, four-component action
mixture and bounded CEM search. The new policy is initialized from scratch;
the previous trained action head is not used as a free pretrained component.
No VLM, language trace dataset, image decoder, RL rollout, or external evaluator
is added. The [literature/code review](LARC_APPLICATION_20261009.md) establishes
RD-VLA/MPCoT overlap and the limits of a novelty claim. This is a candidate
mechanism, not a claimed first invention of latent reasoning or calibrated
physical reachability.

## Causal inputs and training data

Main task/episode/start draws, action targets, future goal offsets and global
batch64 match the prior controller-grounded head. The same 6,222 successful
expert training episodes are used; validation/test examples never train the
new module. For each main start, read up to four preceding control transitions
from that same training episode. Their end times are no later than the current
observation. This adds causal context from the unchanged training set; it is
an explicit method input change. Missing history at episode start is masked.

Historical recorded actions are passed through the frozen world independently
for one step, under no-grad. Per-spatial-token observed-minus-predicted residuals
and observed movement, plus three target-relative progress/error scalars,
condition the workspace. Residual/movement inputs are scaled by10 and scalar
progress by100; normalization of the underlying JEPA states stays unchanged.
Invalid history is zeroed before projections and masked in attention.

At evaluation, history contains the actual observed start/end and predicted
prefix from the previously executed action. It is updated only on the next
observation. The workspace is rebuilt from this four-transition history each
decision; there is no untrained indefinitely persistent hidden state. Episode
reset clears history and pending predictions. No current action outcome enters
its own proposal or score. Training and inference both use observed history,
although expert-history versus policy-history distribution shift remains.

## Objective and factual calibration

The action proposal retains per-coordinate mixture NLL plus the prior0.1
generated-action observed-prefix/endpoint auxiliary loss, ramped from update
1,000 to2,000. The latter still uses frozen-world predictions of generated
actions; observed demonstration states are targets, not claimed outcomes of
those different generated actions.

The new calibration supervision instead uses **recorded actions with their
actual recorded outcomes**. For a supported target `q`, train signed cost
corrections for the one-control-step prefix and the terminal window at control
steps4/5:

```
e1 = d(z1_observed, q) - d(F1(z0, recorded_actions), q)
e5 = mean[d(z4_observed,q), d(z5_observed,q)]
     - mean[d(F4(z0, recorded_actions),q), d(F5(z0, recorded_actions),q)]
```

The prefix readout sees only the first action and predicted prefix token costs;
the terminal readout also sees the full recorded five-step action chunk and
terminal-window token costs. Neither sees the outcome label. Each is conditioned
on the workspace, with output bounded to±0.1 token-cosine units. The separate
prefix interface cannot use unexecuted action suffixes to predict a prefix.

Calibration targets are uniformly relabeled among the recorded local endpoint,
the separately sampled future goal, and another example's observed endpoint
within the same physical training batch. Actions and outcome labels stay paired
with their source trajectory. This avoids teaching that every terminal cost
must be zero just because the query is always its own expert endpoint. These
targets add no simulator data or extra training episodes.

Signed residual labels are divided by0.02 for smooth-L1 supervision. Let `E_K`
be the per-example mean prefix/terminal calibration loss after latent refinement,
and `E_0` the same readout's loss from the initial workspace on the identical
recorded action/target/outcome. The added objective is:

```
0.1 * mean(E_K + 0.25 E_0 + 0.25 max(0, E_K - stop_gradient(E_0)))
```

This is a supervised paired usefulness term, inspired by LARC, not GRPO and not
its frozen-reference objective. The shallow reference is jointly supervised,
so simply worsening it is not rewarded. Training samples one through four
workspace iterations; inference uses four, fixed before evaluation. Gradients
flow through the new workspace/policy/readouts, with all encoder/world weights
frozen. All new trainable components share one method budget.

## Bounded controller and conservative scoring

Retrieve seven supported waypoints plus the direct goal. Use four actions per
target, including a recorded chunk for retrieved targets, with the same three
total CEM batches, five-step world horizon and480 fine-world transitions per
decision. The workspace conditions the mixture proposal once per target.
Batch all candidate correction readouts; do not add world rollouts per thought.

If `J` is the frozen world's terminal-window target cost and `e_hat5` the
learned signed correction, local search and outer selection use:

```
J_revised = J + max(0, e_hat5)
joint_score = unchanged_route_cost + best_local_response(J_revised)
```

This conservative choice penalizes suspected optimism and never rewards an
unsupported generated action with a learned cost reduction below the frozen
world's score. It is not a certified upper confidence bound; positive errors
can still be mispredicted. The signed prefix correction is logged separately
against the next actual observation, with corrected predicted costs clipped
to the cosine-cost range[0,2]. Execute exactly the selected response's first
two primitive actions. No action substitution follows target selection.

The reasoner does not consume imagined outcomes as if they were observed, and
does not require positive goal progress every step. Supported longer routes
and locally necessary setbacks remain possible. The main remaining risk is
that factual error calibration on expert actions fails on generated actions
or unfamiliar recovery states.

## Fixed resource and evaluation contract

- Seed3072 only; global batch64; at most20,000 updates and28,800 aggregate
  optimization GPU-seconds, whichever occurs first. All8 A800 GPUs are used
  with physical batch8. Additional history/factual rollouts, recurrent passes
  and readout training are charged inside the optimizer-step ledger.
- Reuse the same cached V-JEPA2.1 features, state normalization and selected
  fine world hash `ae3f910da43f0a43dd65945394444a0b11719ed042cdacffacc2faaf824ec5a3`.
- Pause training for the same104 validation cases at5k/10k/15k/20k when reached;
  select earliest tied best success. Restore training RNG after evaluation.
- Same cumulative10-second controller clock and200 primitive actions. Encoding,
  history processing, all latent iterations, retrieval, corrections and search
  are charged. Same success/return/action/latency metrics and W&B logging.
- Compare saved controller-grounded79/104, LeFlow27, same-world CEM22,
  repaired flow17, HWM8 and preserved older references. No reference is rerun.
  Historical world/representation/sampler confounds remain. Equal ceilings are
  not equal realized FLOPs or equal total accumulated research compute.
- Initial CPU tests, training-only preflight, cache construction and evaluation
  are reported separately. No parameter update is hidden in preparation.

## Registered failure diagnostics and verification

Retain all prior chosen-target/prefix traces. Also log signed/applied corrections,
whether corrections change the best target within the actual candidate pool,
history length, reasoning depth, raw/corrected prefix progress and their errors
against the subsequent observation. These establish calibration only for the
executed prefix, not counterfactual rejected branches or actual full-chunk
execution. Compare late stalling, false positive progress, latency failures,
per-task success and paired regression cases against the saved79/104 method.

Preserve the same six trajectory cases plus preselected pick-place0, assembly0,
faucet-open3 and reach5 for visual failure inspection. Saving frames is outside
the controller clock but remains in evaluation occupancy; cases/outcomes and
success criteria are unchanged.

Six new server CPU behavioral tests passed before launch: dataset draw equality
and causal history; future labels excluded from predictor inputs; masked-history
invariance and valid-history sensitivity; prefix independence from action suffix;
policy/calibrator gradients with a frozen world; and exact execution handoff,
480-transition limit and memory reset. Full-size training-data forward/backward,
real-sample equality and full encoder/controller latency remain prelaunch gates.
Source will be frozen in an isolated server checkout before training starts.

## Verified launch

Execution source `899f8f2564e011ae87065a70acf28c50dade9997` is frozen in
`/home/mtxu/adam/LeFlow-experiments/20261009-latent-revision/repo`.
Coordinator1886479 was dispatched October9 at21:57:19UTC. The exact registered
training reset rendered bitwise identically. All four old behavioral tests and
six new tests passed on this source. The training-only full-size preflight
verified eight exact old/new main-sample/action pairs and causal history ends;
backpropagation left the world unchanged. It made zero optimizer updates or
simulator policy calls.

The new model has4,531,782 parameters. Full encoder/controller latency averaged
78.58ms across five timed passes (76.57–81.31ms), with9.48GiB peak GPU allocation.
The one-GPU preflight cost0.005534GPUh. Route index construction took25.16CPU
seconds and produced the same130,662 states/209,109 routes from6,222 training
episodes, occupying8.003GiB. Exact evidence is in the
[preflight](reports/20261009-latent-revision/preflight.json),
[reuse record](reports/20261009-latent-revision/reuse.json) and
[index metadata](reports/20261009-latent-revision/route-bank.json).

Eight-GPU training is streaming to
[W&B run5thxkk6y](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/5thxkk6y).
It passed1,600 updates at approximately870–930 examples/second after the full
objective became active. This is a launch snapshot, not a task-success result;
the first registered validation is at5,000 updates. Future results will be
reported against both the saved79/104 controller and historical references.

## Verified completion

The run stopped normally at 20,000 updates after validations of 73, 75, 74 and
77 successes out of 104. The final checkpoint is selected. Optimization used
3.243509 aggregate GPU-hours, registered validation 2.047476 GPU-hours and
preflight 0.005534 GPU-hours. Source, data identity, world and route-bank hashes
were unchanged. W&B synced; owned workers exited and all eight GPUs were idle
at the 22:40 UTC health check. No additional run or final test was dispatched.

Best checkpoint: `/home/mtxu/adam/LeFlow-experiments/20261009-latent-revision/campaign/runs/latent_revision_3072/best.pt`.
Its SHA256 is `cbe214dc379c0f7078817bdef52aee7fa9da30f7f669a9acca6d19a2529dacfe`.
See the [postrun audit](reports/20261009-latent-revision/postrun-audit.json) and
[final process/logging snapshot](reports/20261009-latent-revision/final-health.json).
