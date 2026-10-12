# Execution-calibrated goal progress: registered first candidate

The user explicitly authorized implementation and training on October 11. The
single new candidate is `progress_objective_ttt`, initialized fresh on the same
6,222 successful expert episodes, with seed3072, global64 and the existing
20,000-update OR28,800 aggregate optimization GPU-second ceiling. Four fixed104
development evaluations remain scheduled at5k/10k/15k/20k, with the existing cap
boundary rule. All eight available GPUs are requested; this changes per-rank
random-stream assignment relative to the recent four-GPU runs. It preserves the
allowance and data, not bitwise optimization or equal consumed compute.

## Evidence and implementation

The preserved strongest results remain80/104. Failed retrieval-repair pick-place
episodes end with reach/direct-goal routes, an association rather than causal
proof. The new CPU-only feature diagnostic used40 reference and20 disjoint TRAIN
episodes per task, three separated temporal stages, a fixed spatial random
projection and global cross-task5NN. Accuracy was95.0% overall and91.7% pick-place
versus33.3% chance. No optimizer/model update, GPU, development or sealed-test
episode was used. Raw cosine ordered92.39% of separated expert pairs correctly:
the old score is not uniformly bad, and expert discrimination does not establish
reliability on generated outcomes. Full identities/results are in
`reports/20261011-progress-objective-ttt/feature-probe.json`.

Start from retrieval-repair, with its unchanged global action flow, exact retained
incumbent, bounded0.25 residual flow and two world-verified rounds. Add one small
spatial goal-conditioned readout `R(z,g)` in[0,1]. It predicts clipped expert
remaining block count/60. Supervise temporally separated pairs from the same
training episode and the existing hindsight goal queries; include endpoint-zero
and a bounded ordering margin. These are weak expert-time labels, not physical
success probabilities, shortest-path distances or guarantees of monotonic
progress. No task identity, trajectory clock, proprioception or simulator state
is a policy input. The coarse endpoint is a hindsight cached state near first
success; zero-based success indexing and two-action caching can place it one
primitive action before first success. No successful-terminal classification
label is asserted. The time label refers to that assigned hindsight endpoint.
All new learned parameters train inside the same20k allowance.

For every candidate, add0.05 times half the predicted prefix remaining cost plus
one quarter each of the last two predicted remaining costs. Every target uses
the SAME final goal, avoiding comparisons between unrelated local potentials.
Add a soft target-stage penalty for backward progress or a remaining-cost drop
larger than5/60, corresponding to the five-block local chunk. This is a bounded
preference, not an oracle filter; the original visual feasibility cost remains.

A second scalar ridge fit uses the past four actually executed transitions to
predict observed-minus-predicted remaining-cost error. Its keys share the existing
pre-outcome action/state features. Labels are detached to prevent the readout
collapsing to erase its own calibration error. The offline outer calibration
loss evaluates the next RECORDED action. Corrections are bounded0.25 and only
positive predicted optimism penalizes generated candidates. The episodic delta
also conditions the reasoner; its revised evidence contains the progress-adjusted
candidate scores. Weights remain fixed across both search rounds and reset per
episode. No invented physical labels for generated actions or suffix adaptation.

Controller limits remain10 cumulative seconds/200 primitive actions,320 world
transitions and12 flow velocity evaluations per decision. New scoring overhead
must pass the unchanged95ms training-data preflight gate. No frozen baseline
implementation or running checkout is modified. New hooks in shared controller
code are identity/zero by default for historical methods.

## Foundations and differences

[ProWorld](https://arxiv.org/html/2608.01926) motivates goal-conditioned coarse
temporal ordering; it trains a hyperbolic world model. We keep our world frozen
and learn a small readout. Its promised code was not available at review time.
[ReGuide](https://reguide-project.github.io/) motivates phase-aware guidance and
bounded trust; official clustering code at89babcf was inspected in the preceding
review. Its visual/proprioceptive phases and rollout-data expansion are not copied
into this fixed visual-only experiment. We retain our world-verified residual
generation rather than introducing another differentiable guidance solver.
This is an adaptation inspired by these findings, not a reproduction of either
method. Generic feedback/retrieval/flow alone is not a novelty claim.

## Decision and failure analysis

Primary measure: physical task success on the same104 cases under the controller
allowance, compared with preserved80 and the released-method ports. Publish all
four scores, paired gains/regressions, latency/timeouts and actual compute.
Secondary diagnostics: readout weak-label MAE/order, prior versus adapted factual
error, stage penalties, route drift, incumbent selection and bounded repairs.
Observe whether progress scores change decisions and whether those decisions
actually succeed. Proxy error reduction alone is insufficient.

Inspect pick-place failures and the assembly/faucet/reach cases, including saved
videos/contact sheets. Likely unresolved risks are expert-only supervision,
off-distribution generated states, object information lost by the frozen encoder,
and inaccurate five-block forecasts. Make the next change only from concrete
failure evidence. No blind queue, additional seed, baseline retraining or sealed
final test. A score above80 on repeatedly used development cases still does not
establish independent generalization, component causality or SOTA.

## Runtime record

Server record: `/home/mtxu/adam/LeFlow-experiments/20261011-progress-objective-ttt`.
Separate bank/cache: `/tmp/mtxu-progress-objective-ttt-20261011`.
Current status and immutable execution source are maintained in the active ledger.
Full metrics, evaluations, failure analysis and measured cumulative compute must
be published to origin/main at completion.

Verified running October11 21:56 Pacific: source e274fed5850b967c5e8e176e5cab3611d992e8bf,
166 runtime hashes unchanged, all46 tests and8-rank synchronized finite gradients
passed. Full-controller preflight89.52ms. Coordinator1324049; W&B
[u4dd8tyw](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/u4dd8tyw) running.
At least1,300 updates,0.301860 optimization GPUh, no validation yet; all8 A800s
allocated. Median recent update0.105s gives about10minutes to the first completed
evaluation and roughly45–55minutes for the full run from launch, including the
four evaluations. These are throughput estimates, not deadlines. All new trained
parameters are fresh; the no-update feature probe spent0.458CPU seconds.

The latest training-batch temporal MAE is0.02294 and pair ordering100%; these are
training proxies, not development success. The decisive80/104 reference is unchanged.
Use scripts/analysis/progress_objective_details.py on saved eval reports for
applied-versus-raw progress error and per-task route tails; rejected candidates
still have no physical outcome labels. The existing15-minute monitor continues
evaluation/failure review and completion publication, with no speculative queue.
