# Next MetaWorld formulation: demonstrated routes, coarse checks, local control

Design review, October 9, 2026. Project evidence is pinned to
`4a990492848e11500cb6962fb53c7384ffefb77e`. This extends the
[H-JEPA / EB-JEPA review](HJEPA_EBJEPA_REVIEW_20261009.md) with a broader primary
literature and implementation review. **Status: design only.** No model code,
running process, experiment queue, monitor, baseline or test result was changed.

## Decision and empirical motivation

Recommend one candidate: **retrieve a demonstrated route, check its near-term
actions with an action-conditioned coarse predictor, and optimize a short action
chunk toward its first recorded waypoint using the existing fine world model.**
A small, one-pass mixture policy supplies local action proposals. Keep the
cached visual backbone and selected fine world frozen. Train only the coarse
predictor and mixture policy, together within the original method allowance.

This is the best-supported next hypothesis from this review, not evidence that
it is the best-performing method. It deliberately makes fewer unsupported future
predictions than our repaired joint state/action flow. It does not replace that
flow with a deeper hierarchy, a new learned distance, or an online data campaign.

Our [completed validation](REPAIRED_VALIDATION_RESULTS.md) provides the following
constraints on a useful redesign:

| Measured or audited fact | Design implication |
|---|---|
| Ours selects 17/104; the same selected world's saved CEM result is 22/104. Historical LeFlow is 27/104. | A learned planner must demonstrate an advantage over the frozen references; lower training loss is insufficient. |
| All 87 selected-checkpoint failures hit the ten-second controller limit after 92–96 primitive actions. Mean decision latency is 208.95 ms. | Reduce repeated planning work. With two primitive actions per decision, reaching the 200-action ceiling requires roughly 100 ms per decision on average. |
| The fine world is trained on continuous five-control-step rollouts, but proposal ranking uses 60 steps. | Keep fine verification at five steps; train the new coarse predictor over its actual four-step inference horizon. |
| Generated-bridge consistency resets at proposed states; inference ranking rolls continuously from the real current state. | Coarse compatibility checks must start from the live state and remain continuous. |
| Current action identification is 0.877 at 16 choices; prediction loss beats persistence. | Preserve the useful existing world initially. These offline scores do not certify its control ranking. |
| The older bounded diagnostic found weak correlation between predicted and actual short-chunk progress; it did not test the repaired checkpoint. | Record observed execution and success; do not claim repaired candidate scores are calibrated. |

Timeout is an observed stopping condition, not a diagnosis of contact, grasping
or perception failure. No new failure video was inspected for this design.
The eight CEM-only successes, especially seven coffee-button cases, remain
valuable paired cases for the next registered failure analysis.

## What the broader literature changes

These are lessons for our setting, not transferred success claims. Numerical
results from other papers use their own tasks, observations and budgets.

| Primary source and review scope | Applicable evidence | Decision / transfer limit |
|---|---|---|
| [Anchored Planning / Aim Short to Reach Far](https://arxiv.org/html/2609.30036v1), September 24; methods, results and retrieval/planning code | Nearby observations retrieved from training trajectories can serve as local targets for a frozen model. Recorded targets remain useful even when learned predictions improve. Its action-ranking variant compares candidates against a common anchor. | Retrieve actual observations; use one common local target. Its retrieval receives the demonstrated start-to-goal duration, and fixed-span retrieval performs poorly. We cannot supply that oracle duration. Our variable-span retrieval is an unvalidated adaptation. |
| [Hi-LeWM / Mind the Gap](https://arxiv.org/html/2607.12547v1), July; method, failure analysis and empirical-bank solver | Unconstrained coarse action search can exploit model error. Empirical macro-action support reduces this problem; a low terminal cost can coexist with a bad first waypoint. | Coarse candidates come from recorded action routes. Score intermediate compatibility, not only the terminal prediction. PushT results and tuned configurations are not MetaWorld evidence. |
| [SAGE](https://arxiv.org/html/2607.17973v1), July; paper methods and results | A subgoal-conditioned, one-pass Gaussian-mixture action generator can initialize short model-based refinement. Training its action decoder on predicted subgoals addresses a conditioning mismatch. | Use multimodal short action proposals instead of eight flow passes over a large state path. The paper also uses proprioception and known duration, which we do not have. Official code was not verified. Our observed-target training retains a retrieval-target mismatch. |
| [Temporal-Distance JEPA](https://arxiv.org/html/2607.25337v2), July; methods, manipulation results and temporal-loss code | Better temporal ranking can accompany worse contact-rich control. Its pure temporal cost underperforms geometric cost on reported manipulation comparisons. Cross-trajectory negative labels are heuristic. | Keep geometric scoring for local control. Do not add a temporal critic merely because it makes expert paths monotonic. Representation-training improvements are distinct from cost replacement. |
| [Planning Limits](https://arxiv.org/html/2609.39235v1), September 30; paper methods and results | On MetaWorld, short expert subgoals can help much more than extending model rollouts. Proposal quality matters even when the model ranks local alternatives usefully. | Strong task-relevant motivation for short verification. Expert future subgoals in an oracle analysis are unavailable to our deployed controller. No paper reproduction was run. |
| [RC-aux / Predictive but Not Plannable](https://arxiv.org/html/2605.07278v1), May; method/results, repository entry points | Training queries should match planning horizon and reachability questions. Trajectory time separation is only partial reachability evidence. | Align coarse training and inference. Do not treat a slow expert route, a different trajectory, or a low learned energy as proof of physical unreachability. No extra critic in this candidate. |
| [Qantara](https://arxiv.org/html/2607.04978v1), July; targeted training-query ablations and training code | Matching planning-query structure and stabilizing residual prediction can matter substantially. | Use continuous coarse rollout supervision and zero-initialized residual output. Its end-to-end representation training is not a drop-in change under our frozen-cache constraint. |
| [HWM](https://arxiv.org/html/2604.03208v2), April; paper and public implementation scope | Action-conditioned temporal abstraction provides longer reach with fewer predictor calls. | Train a five-control-step coarse transition in the existing state space. Its public HWM_PLDM repository documents a maze implementation; our old four-dimensional macro-action adaptation is not a faithful robot reproduction. |
| [LeFlow](https://arxiv.org/html/2608.24855v1), August; paper and planner training code | Amortized proposals plus verification are useful. Generated-plan consistency is already part of the paper's formulation. | Keep the proposal/refinement pattern. A full generated latent path is not essential to that pattern or a justified novelty claim. The inspected public training path passes observed states to its consistency calculation. |
| [H-JEPA](https://arxiv.org/html/2610.06805v1), October 5, and [EB-JEPA](https://arxiv.org/html/2602.03604v3), April revision; previous detailed review | Temporal hierarchy, constrained coarse actions and local planning are relevant. Abstract cost benefits vary by environment; changing the optimizer alone has weaker evidence. | Keep one shallow coarse level, original spatial features, and CEM. Do not add representation regularization to frozen tensors or claim expert-trajectory cost trends establish executable proposals. |

Additional relevant papers were screened through targeted sections:

- [Reinforced Planning](https://arxiv.org/html/2608.18669v2): its model-exploitation
  failures and corrective online data collection matter, but collecting new
  failures and retraining the world changes our fixed offline data protocol.
- [Diffusion Subgoal Planning](https://arxiv.org/html/2609.34575v1): support and
  reachability can deteriorate with stronger guidance. Valid-looking states
  need not be reached. Its main state-based evidence and limited visual appendix
  do not justify another unconstrained visual subgoal generator here.
- [LeWAM](https://arxiv.org/html/2609.27455v2): its own future-prediction ablation
  does not establish flow as uniformly better than direct prediction. The larger
  model, task setting and preference-training stage are substantial differences.
- [Flow-JEPA](https://arxiv.org/html/2608.29029v3): stochastic joint prediction
  is relevant, but its stated attribution limits and repeated sampling cost do
  not justify replacing our world for this iteration.
- [FF-JEPA](https://arxiv.org/html/2606.09311v1): a different goal-free setting
  supplies evidence that hierarchy need not require a generative latent path.
- [DDP-WM](https://arxiv.org/abs/2602.01780): abstract-level screening only. Its
  dynamics architecture is a potential future speed direction, not a lossless
  batching change to our already trained model.

## Exact proposed computation

### State, action and frozen components

Use the same normalized static-image V-JEPA 2.1 cache (`image_goals`):
`z, g ∈ R^(32×1024)`. Use the existing per-token cosine distance
`d(x,y) = mean_p[1 − cos(x_p,y_p)]`. One control action is eight-dimensional,
containing two four-dimensional primitive actions. A local chunk contains
five control actions, `U ∈ R^(5×8)`; its physical duration is ten primitive
actions. Continue to execute just one control action before observing again.

Freeze the current encoder, normalization and selected fine model `F`, world
checkpoint SHA-256
`ae3f910da43f0a43dd65945394444a0b11719ed042cdacffacc2faaf824ec5a3`.
No new visual cache, learned state metric, task ID, proprioception, oracle
evaluation duration, validation trajectory or future observation enters control.

### Demonstration memory and route selection

Build memory only from the same **6,222 successful training expert episodes**.
Store pointers to observed states and recorded actions. Candidate route starts
use a five-control-step grid, capped at the episode's recorded first-success
index as in the training sampler. Route spans are `h ∈ {5,10,20,40,60}` control
steps when all required frames/actions exist in that episode. Do not concatenate
unrelated episodes or use evaluation task labels to filter the bank.

For a record `j`, denote its source state `Z_j,0`, route endpoint `Z_j,end`,
five-step observed successors `Z_j,r`, and action chunks `U_j,r`. Retrieve by

`R_j(z,g) = d(z,Z_j,0) + d(g,Z_j,end)`.

The route span is metadata from a training example, not the unknown duration of
the evaluation problem. Search all permitted spans at every decision. This
avoids requiring an oracle countdown, but its effectiveness must be tested:
spatially similar source/goal pairs can still require incompatible actions.

For bounded retrieval cost, use a fixed seed-3072 projection to 256 dimensions
of the flattened, individually unit-normalized tokens. Shortlist 256 records
using projected source-plus-endpoint distance; rerank them with the exact
original-token `R_j` and retain 32. This projection and temporal subsampling are
**approximate algorithm choices**, not lossless throughput optimizations.
Use original features for every subsequent model prediction and goal score.

An upper bound before source filtering is 6,222 × 21 = 130,662 distinct grid
states for 100-control-step episodes. Their float32 projected keys occupy about
127.6 MiB, plus route metadata and actions. The corresponding full float16
latents would occupy about 7.97 GiB if separately staged; use pointers to the
existing cache and a bounded staging policy rather than requiring another full
copy. These are shape-derived estimates, not measured storage or runtime.
When measured peak memory permits, preload this read-only raw float16 grid once
per inference GPU and normalize gathered entries in the original float32 order.
Avoid per-decision HDF5 reads. Keep projected keys and route metadata resident;
measure staging overhead before choosing a fallback. The eight 80-GiB A800s
make this plausible, but memory fit and throughput are not verified here.

### Action-conditioned coarse verification

Train `F_H(z,U)` to predict the next observed five-control-step state in the
same 32×1024 feature space. Start from `Dynamics(dim=1024, action_dim=40,
width=256, depth=4, heads=4)` with its zero-initialized residual output. Input is
the actual flattened action chunk, not a freely optimized four-dimensional
macro-action code. This is temporal abstraction without an additional learned
abstract state space.

For each of the 32 retrieved routes, continuously roll from the live state:

`hat_Z_j,0 = z`

`hat_Z_j,r = F_H(hat_Z_j,r−1, U_j,r)`, for `r = 1,…,m_j`,
where `m_j = min(4, h_j / 5)`.

Choose the route minimizing

`S_j = R_j + (1/m_j) Σ_r d(hat_Z_j,r, Z_j,r)`.

Equal weights are explicit initial design defaults, not validated optimal
weights. The compatibility term checks all predicted coarse waypoints from
the current reset. For longer routes, only the first 20 control steps are
model-checked; the recorded suffix supplies goal connectivity evidence, not a
verified rollout to the final goal. The selected **observed** first successor
`q = Z_j*,1` becomes the common local target. Do not substitute a model-predicted
state and then use self-consistency as proof that it is reachable.

### One-pass action proposals and five-step fine verification

Train a four-component diagonal Gaussian mixture over the entire 40-dimensional
chunk: `πθ(U | z,q,g)`. Use separate state-type embeddings and a three-layer,
width-256, four-head transformer over the 96 current/anchor/goal tokens, followed
by pooled mixture logits, means and scales. Bound means with `tanh` and scales
to `[0.05,0.5]`. These numerical defaults are chosen once, without a sweep.

The first batch contains 24 mixture samples and eight recorded first chunks
from the best coarse-ranked routes, including the selected route. All 32 are
evaluated against the **same** `q`. Clamp actions to the environment's physical
bounds before model prediction. Use two further CEM refinement batches, with
32 candidates, eight elites, scale floor 0.05, and retention of the best
previous candidate. There are **three candidate batches total**, not three
additional rounds after the initial batch.

For each chunk, continuously roll the existing fine world for five steps from
the actual current state. Minimize the terminal-window cost

`J(U;z,q) = 0.5 d(F^4(z,U_1:4),q) + 0.5 d(F^5(z,U_1:5),q)`.

This proposed cost removes the current minimum-over-any-time incentive for a
transient close encounter. It does not impose monotonic approach to the final
goal at every step, which could punish necessary manipulation detours. Whether
the last-two-step window is beneficial remains an empirical question.

Execute the best chunk's first control action, observe again, and replan. Keep
the registered two-primitive-action observation cadence and budget-interruption
semantics. Include encoding, retrieval, transfers and both planning levels in
controller timing; do not stop the clock for the new overhead.

### Training objective and alignment

Keep the original task-balanced, stateless sampled 60-control-step expert
windows and global batch64. Use the same episode/window indices; any goal-offset
sampling gets its own deterministic stream so it does not alter those indices.

For each window, train `F_H` by a **continuous four-step coarse rollout** with
recorded action chunks, supervised at offsets 5, 10, 15 and 20. Do not reset the
prediction to observed states between those steps. Let `ell` be the existing
feature MSE plus token cosine loss:

`L_coarse = (1/4) Σ_r ell(hat_Z_r, Z_r)`.

For the mixture, use current state `Z_0`, observed local target `Z_5`, observed
first action chunk `U_1`, and a goal sampled at offset
`δ ∈ {5,10,20,40,60}` within the same already sampled window:

`L_action = −(1/40) log πθ(U_1 | Z_0,Z_5,Z_δ)`.

Optimize `L_coarse + L_action` in one training job, initializing both modules
from scratch. Reuse the existing optimizer/schedule unless a demonstrated
implementation incompatibility requires a documented change. There is no
extra learned critic, inverse head, representation regularizer, flow sampler,
new world training or generated-plan consistency objective in this candidate.

Observed local-target supervision is valid action evidence. At inference,
retrieved anchors may lie off that conditional training distribution. We retain
this limitation explicitly; SAGE's predicted-target training is not reproduced,
and arbitrarily relabeling an expert action as reaching a different retrieved
state would create false supervision. Recorded proposals provide an additional
supported source, not a guarantee that the new reset can execute them.

## Compute, data and comparison contract

| Item | Fixed requirement |
|---|---|
| Training seed | 3072 only. |
| Data | Existing 7,800 training episodes: 6,240 expert and 1,560 random. New modules/memory use the same 6,222 successful experts as the repaired head, not the validation pool. |
| Observation | Same static-image cache and normalization; same frozen encoder and fine world. |
| New trainable work | Coarse predictor plus mixture jointly: at most 20,000 global-batch64 updates (1.28 million sampled windows), **and** at most 28,800 optimization GPU-seconds, whichever limit is reached first. No separate allowance per component. |
| GPU scheduling | All eight GPUs may accelerate the one candidate and its existing validation schedule, once an iteration is authorized. Eight GPUs would exhaust the optimization GPU-time ceiling in 3,600 seconds; they do not receive 7,200 seconds each. Preserve global batch64 and charge the actual allocation. |
| Registered validation | Same 104 cases, eight per training task, at 5k/10k/15k/20k if reached within the ceiling. Earliest tied best success selects the checkpoint. No hidden extension to reach a missing round. |
| Controller | Same cumulative ten-second allowance, 200 primitive actions, action bounds and observation cadence. Same hardware/timing boundary as reference evaluations. |
| References | Preserve saved same-world CEM22 and historical LeFlow27/HWM8/old-world CEM7; no repeated baseline training or evaluation. |
| Final tests | Reserved. Do not use held-out cases or earlier sealed outputs to design, select or tune this candidate. |

This is an **equal ceiling**, not a promise of identical realized training
FLOPs or GPU-hours. The repaired head used 5.279885 optimization GPU-hours; a new
candidate may consume more or less within the eight-GPU-hour ceiling. Report
actual use for each method. Shared fine-world/encoder pretraining and amortized
reuse remain explicit; they are not new free training. Memory preparation,
preflights and registered evaluation get separate charged ledger entries.
Preparation must not conceal gradient updates or another trained component.
The cumulative known prior campaign subtotal remains 42.713129 GPU-hours.

The existing data are a custom 13-task image-goal MetaWorld-v3 protocol, not
standard MT10/MT50. The 650-episode validation pool supplies the already frozen
104 evaluation cases. The selected world's prediction validation uses 512 fixed
windows. These diagnostics must not be confused with successful task execution.

### Throughput target, not a speed claim

The old planning call has 60 fine rollout stages plus 15 local refinement stages
and eight flow passes. Its 32-candidate work is 2,400 fine transitions.
The proposed call has at most four coarse stages and 15 fine stages plus one
mixture pass: at most 128 coarse and 480 fine transitions. Coarse and fine work
have different input sizes; adding them is not a FLOP-equivalence calculation.
Retrieval, encoding and transfers remain. No measured speedup is claimed.

Before committing a training allowance, an implementation must measure an
end-to-end call on the actual hardware with representative shapes, including
retrieval and a full encoder call. The roughly 100 ms average target follows
from the fixed controller/action limits. If the encoder or memory transfer sets
a higher floor, record that limitation and revise the design explicitly; do not
quietly relax controller time or execute longer chunks to make it fit.
This review did not run that benchmark or consume GPU time.

Use fused candidate batches, preallocated buffers, deterministic cache reads,
and loss-preserving data prefetch where supported. Validate numerical outputs,
candidate ordering and timing boundaries when changing execution. Different
retrieval approximations, precision, horizons, commit lengths, early stopping
or candidate counts are method changes unless equivalence is demonstrated.

## Decisive evidence and failure analysis

The primary outcome is **paired task success on the registered cases within the
fixed controller allowance**. Report per-task successes, CEM-only/ours-only
cases, actions executed, controller time, throughput and actual training spend.
Beating 22/104 would improve this selected-world development reference; beating
27/104 would improve the historical LeFlow score. Neither alone establishes
statistical reliability or superiority to a corrected published SOTA method.
One seed, reused validation and historical implementation confounds remain.

Log, within those same registered episodes:

- Encoder, retrieval, coarse, mixture, fine-refinement and transfer latency;
  controller p50/p95, action counts and timeout conditions.
- Retrieved episode/start/span, source and endpoint distances, coarse waypoint
  errors, chosen anchor, anchor changes and local candidate costs before/after
  refinement. Record metadata without feeding task labels to the planner.
- Actual next observation compared with the predicted result of the **executed**
  first control action; distance change relative to the previous fixed anchor,
  success and task return. Keep comparisons tied to that old anchor when the
  next decision selects a different one.
- Mixture component usage, clipping and candidate diversity, including which
  source supplied the executed action. This can expose an ignored goal/anchor
  or a collapsed proposal distribution without adding training variants.

Because only one control action executes before replanning, these logs do not
observe the unexecuted five-step suffix or counterfactual candidates. Do not
label them as full-chunk executability or candidate-rank calibration. A direct
ranking study needs separately budgeted counterfactual executions; none is
included or silently scheduled. Expert-path monotonicity remains secondary.

| Plausible failure, not yet established | Evidence to seek in the registered run |
|---|---|
| Retrieval matches appearance but not controllable configuration | Large realized one-action error, poor source coverage, frequent anchor switching, or repeated no-progress actions despite low retrieval cost. |
| Coarse model exploits its own errors | Low predicted route mismatch with poor observed first-action progress; this is a warning, not full five-step verification. |
| Local proposer ignores the target or encounters unfamiliar retrieved anchors | Similar action distributions across changing anchors; recorded-action candidates repeatedly outperform mixture candidates. |
| Fine-world cost misranks useful contact actions | Predicted improvement diverges from executed progress and task success, especially the existing CEM-only cases. Video would be needed for a physical contact diagnosis. |
| Encoder/retrieval dominates the time allowance | Component timings explain insufficient action opportunities despite fewer predictor stages. |
| Memory lacks support for new task configurations | Poor retrieval coverage; known-task gains cannot establish held-out-task generalization. Keep test sealed. |

## Implementation boundary and novelty

The next implementation would add a train-only route-index builder/loader,
`CoarseDynamics` and `ChunkMixturePolicy`, their joint loss, and one controller
entry point using the formulas above. Existing `models.py:Dynamics`,
`data.py:Segments`, `evaluate.py`, budget handling and comparison reports are
the integration points. Add meaningful checks for data-split exclusion,
action/state alignment, common-anchor scoring, continuous rollout, fixed global
batch and aggregate budget enforcement. No runtime configuration or launcher
for this design exists yet, and no baseline code should be overwritten.

The existing completed-run hold remains in force. The user's present request
is to research and design the next formulation; it does not queue a fresh
experiment. The concrete next stage is implementation and a charged throughput
check when continuing into an authorized iteration, followed by exactly one
from-scratch candidate and its four registered validations. Analyze those first
results before deciding whether any ablation is worth its cost.

Retrieval anchors, empirical action support, coarse dynamics, mixture proposals
and local verification all have precedents. The potential contribution is a
validated combination that improves this task under its real resource limits,
with honest failure evidence. Do not claim invention of these ingredients,
generated consistency, or a universally best formulation.

## Inspected code revisions and verification scope

These repositories were read, not executed or reproduced. Paper-only and
targeted-review limitations are marked above.

| Repository | Pinned revision | Inspected portion |
|---|---|---|
| [Anchored Planning](https://github.com/daybraeklaxry/Aim-Short-to-Reach-Far) | `984327424ca7a2e52c75a8279197750d9d0b7c8e` | Observation bank, shared-anchor planners and observed-target solver. |
| [Hi-LeWM](https://github.com/NiccoloCase/Hi-LeWM) | `4bb21a2888e8f22b8d084762c80361e398968775` | Empirical macro-action bank and residual CEM policy. |
| [TD-JEPA](https://github.com/HKBU-KnowComp/TD-JEPA) | `b4c17ca4649c9bf47272fa66c38da7a684f2a020` | Temporal-distance loss, positive/negative construction and calibration comments. |
| [RC-aux](https://github.com/Guang000/RC-aux) | `cbdf3786b149df8145d6c7314f32f460d43c9695` | Repository inventory and training entry points; not a full implementation audit. |
| [Qantara](https://github.com/corl-team/qantara) | `abdbcb23235e33c3bd83c2c669f5285e9883badf` | Training time-pair masks and rollout sections. |
| [HWM_PLDM](https://github.com/kevinghst/HWM_PLDM) | `e197375b844692a0a2e1342889f95a78edced07a` | Public implementation/benchmark scope. |
| [LeFlow](https://github.com/hsiangwei0903/LeFlow) | `f1fe192e41ec6f20de25cd1054ede40a8cbdcfa5` | Planner training and consistency inputs. |
| [H-JEPA](https://github.com/kevinghst/H-JEPA) | `840e76b5894d9acd335785518318d17814918265` | Hierarchical costs, action constraints and loss/regularization code; prior review. |
| [EB-JEPA](https://github.com/facebookresearch/eb_jepa) | `966e61e9285b3a876f49b9774e9720d9a99a7925` | Planning costs/optimizers; prior review. |

Verification for this deliverable: re-read the final saved formulation, checked
its arithmetic, repository-relative references, source identities and diff.
No candidate implementation, latency benchmark, training outcome, physical
failure diagnosis or upstream reproduction is claimed.
