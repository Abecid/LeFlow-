# H-JEPA / EB-JEPA: implications for the completed MetaWorld comparison

Reviewed October 9, 2026, against project revision
`618e7b396997d30f20d37d62bdb1aa1aae1306c6`. This is a literature and source-code
review. No training, simulator evaluation, new diagnostic, ablation or test was
launched. The completed campaign and paused monitor remain unchanged.

## Evidence and source identity

- [H-JEPA project](https://h-jepa.com/) and
  [H-JEPA, arXiv:2610.06805v1](https://arxiv.org/html/2610.06805v1), October 5,
  2026: methods, planning results and appendices E, G, H, I and K.
- [EB-JEPA, arXiv:2602.03604v3](https://arxiv.org/html/2602.03604v3), April 8,
  2026: training/planning objectives and Table 4. This is the user's second
  link, a separate library/tutorial paper, not H-JEPA itself.
- Official H-JEPA code, commit `840e76b5894d9acd335785518318d17814918265`:
  [hierarchical costs](https://github.com/kevinghst/H-JEPA/blob/840e76b5894d9acd335785518318d17814918265/h_jepa/hierarchical_solver.py),
  [solver](https://github.com/kevinghst/H-JEPA/blob/840e76b5894d9acd335785518318d17814918265/stable_worldmodel/solver/gd.py),
  [regularization and inverse dynamics](https://github.com/kevinghst/H-JEPA/blob/840e76b5894d9acd335785518318d17814918265/h_jepa/loss.py).
- Official EB-JEPA code, commit `966e61e9285b3a876f49b9774e9720d9a99a7925`:
  [planning objectives and optimizers](https://github.com/facebookresearch/eb_jepa/blob/966e61e9285b3a876f49b9774e9720d9a99a7925/eb_jepa/planning.py).

H-JEPA learns different state spaces at different temporal scales. Its
projected-cost gains are environment-dependent: navigation benefits, while
Push-T and Cube show no clear benefit from changing only the cost space.
Its manipulation evidence therefore supports temporal decomposition more
consistently than an abstract-distance replacement alone. Deeper models can
lose eligible training clips. DROID uses offline path fidelity, not closed-loop
task success. Its cost-monotonicity analysis follows expert trajectories and
re-anchors local segments; it does not establish candidate executability.
These qualifications matter more here than the headline AntMaze result.

EB-JEPA Table 4 reports Two Rooms success of 97% for MPPI with cumulative cost,
96% for CEM with that cost, and 89% for MPPI with terminal-only cost. This is
evidence to examine objective design before replacing CEM, not evidence of
MetaWorld performance. It also investigates recursive rollout training and
action supervision for learned representations.

## What our implementation and results actually establish

The [completed validation review](REPAIRED_VALIDATION_RESULTS.md) selects ours
at **17/104**, versus **22/104** for already-recorded CEM with the same selected
world and **27/104** for historical LeFlow. All 87 selected-checkpoint failures
hit the 10-second controller ceiling after 92–96 primitive actions. Timeout
identifies why evaluation stopped, not why physical progress was insufficient.
Historical references differ in representation/world and have known defects;
none establishes a corrected-SOTA comparison. Final tests remain reserved.

Source audit:

| Component | Confirmed implementation | Implication for the next design |
|---|---|---|
| State and cost | `models.py:distance` averages token cosine distances in frozen V-JEPA features. | We have not established that this distance measures controllable task progress. |
| Existing hierarchy | Joint flow generates 12 segments in the same feature space. HWM learns macro-actions but its coarse dynamics outputs that same state space. | Neither supplies a separately learned abstract state representation. |
| World training | `Segments` supplies five-control-step windows; `System.forward` trains a continuous autoregressive rollout against all five observed successors. | Multistep training already exists. Calling it single-step training would be wrong. |
| Candidate scoring | `continuous_plan_score` rolls 32 candidates for 60 control steps from the actual start, then adds endpoint goal cost and 0.1 waypoint mismatch. | The scoring horizon is twelve times the world-training horizon. Long-rollout reliability remains unverified. |
| Planner consistency | Each generated bridge is rolled for five steps from its generated start state. | Local consistency need not imply that a continuously executed 12-segment chain works. |
| Local refinement | Three CEM rounds, 32 candidates and five steps; its cost takes the minimum goal distance over time. | A transient close encounter can receive a good score without sustained local achievement. |
| Execution | Only the first control block, containing two primitive actions, executes before replanning. | We repeatedly pay for a much longer proposed future. |

Each complete planning call structurally includes 60 batched fine-world calls
for proposal ranking and another 15 for refinement, plus eight flow steps and
visual encoding. That is a code-derived count, not a latency profile; we have
not measured which component dominates the time ceiling in this review.

The [earlier bounded diagnostic](REPAIR_RESULTS.md) found only 0.205 mean
within-condition correlation between continuous score and actual short-chunk
latent goal progress for the old proposed checkpoint. That proxy is not task
success, and the diagnostic did not test this repaired checkpoint. It cannot
be used as evidence that the repaired ranking works.

## Recommended direction, conditional on another authorized iteration

Prioritize **a shallow planner with action-conditioned coarse prediction and
short local verification**, targeting both the excessive scoring horizon and
the uncertain connection between generated waypoints and executable actions.
This is a proposed adaptation, not an established best method or an exact
H-JEPA reproduction.

Keep the existing image cache and frozen visual backbone for a first
compute-constrained design. A small trainable state adapter and coarse
action-conditioned predictor could operate above them. Use the learned
proposal to initialize bounded action search; obtain candidate subgoals from
the coarse action rollout and check the immediate chunk with the fine world.
Avoid letting free state predictions alone certify reachability. Retaining a
frozen backbone cannot recover task information that its features discarded.

The official hierarchical cost implementation projects lower-level predicted
states into the target space and optionally compares encoded lower actions
with the macro-action target. Its solver constrains macro-actions using their
training distribution. Those are concrete references for connecting our
coarse proposals to local execution. Such constraints reduce unsupported
search but do not prove that a macro-action is physically feasible.

If an adapter is learned, inverse-action prediction must send gradients into
that adapter, with separate protection against representation collapse.
Our existing action-output loss and LeFlow's inverse head on frozen features
are not equivalent to shaping a trainable state representation. Current frozen
features have not been diagnosed with slow-feature collapse. Do not add a
regularizer to frozen tensors and claim it repairs them.

For multi-GPU implementation, the H-JEPA regularizer aggregates characteristic
function statistics across ranks. A port must also verify shared projection
directions, gradient scaling and accumulation across microbatches. Averaging
independent small-batch regularizer losses generally does not reproduce one
global-batch loss. No upstream multi-GPU equivalence test was executed here.

Treat cumulative or terminal-window costs as candidate objective choices.
Our own inference is that local subgoal achievement is a safer starting point
than demanding monotonic distance to the final image: manipulation may require
approaching, grasping or moving away before completing the task. Do not copy
the Two Rooms cost or its hyperparameters without that distinction. Keep CEM
unless evidence justifies changing the optimizer.

## Evidence needed and preserved constraints

Before a new expensive comparison, the unresolved questions are whether scores
rank real executable progress, whether proposed first chunks reach useful
subgoals, and where controller time is spent. The eight existing CEM-only
successes are especially informative paired failure cases. Expert-path cost
monotonicity is a useful supplementary diagnostic, but candidate ranking,
observed execution and eventual success are the relevant checks. Any new
simulator diagnostic needs a stated compute allowance; none is queued here.

A future selected variant should retain seed 3072, the registered train and
validation cases, the training ceilings and four periodic validation rounds,
and the 10-second controller / 200-primitive-action limits. Newly trained
adapters or coarse predictors count toward the method's allowance, not free
extra training. Keep the existing baselines frozen and report any new world
or representation difference as a comparison limitation. Do not add depths,
seeds, sweeps, automatic ablations or test-set feedback.

Primary evidence remains paired task success within the fixed controller
budget. Additional latency and progress diagnostics explain outcomes; lower
latent loss, high feature rank or offline path similarity cannot replace it.
Changing horizons, costs, commit frequency or planner early stopping is an
algorithmic change, not a lossless throughput optimization.

Verification: checked the cited local code paths, configuration, completed
validation review, earlier diagnostic report, paper methods/tables and pinned
official source files. Reviewed the final saved report and its repository
diff before publication. No new performance or upstream reproduction claim
is made.
