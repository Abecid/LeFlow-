# Candidate-conditioned action flow: completed run and failure analysis

October 10, 2026. **The authorized fresh run completed 20,000 updates and four
fixed 104-case validations: 79, 75, 76, 76.**
The registered selection rule retains **5,000 updates, 79/104
(75.96%)**. The strongest preserved GMM controller is **80/104
(76.92%)**. This first flow-reasoning candidate does not improve on it.

The code now implements candidate-conditioned reasoning and flow action generation.
It does **not** yet demonstrate useful test-time scaling, causal reasoning gains,
or a publication-ready superiority claim.

## What was implemented

The previous reasoner completed its internal updates before candidate search.
This implementation closes that gap: a four-slot latent workspace conditions
flow-generated action chunks, observes their frozen-world predicted consequences,
and revises its context before the next generation round. Past observed execution
errors initialize the workspace. Physical JEPA states and internal thoughts remain
separate. This is the accepted research formulation, not a full LARC/BAGEL replica.

A fresh 5,445,162-parameter policy uses a conditional rectified flow over ten
primitive actions (five two-action blocks), with eight Euler evaluations per
proposal round, unrestricted Gaussian source noise and final-only action clipping.
There are eight targets (seven retrieved and the direct goal), four candidates
each and three rounds. Each
round preserves candidate/action/outcome pairing. The same candidate noise across
rounds makes generated revisions depend on the changed context. The first round
also retains recorded-route and observed-before-reuse plan proposals.

A separate scoring context stays fixed throughout a decision. Revising a thought
cannot improve an unchanged plan merely by changing its scorer. The controller
retains the best candidate across rounds, uses prefix/terminal cost corrections
and causal stall penalties, and executes the first two primitive actions.
Exactly 480 predicted world transitions are allowed per complete decision; flow,
reasoning, retrieval, encoding and scoring all count in the controller clock.

Training cycles through one, two and three rounds. Its objective combines mean
flow-matching error across visited depths, a 0.25-weight paired regression hinge
and 0.10-weight factual calibration. The teacher target, Gaussian noise and flow
time are shared across depth losses. Planning noise is independent; current
recorded actions and prefix/terminal outcome labels never enter deliberation.
Supplied goal/subgoal states remain legitimate conditioning inputs.
Imagined evidence is detached and the world is frozen. Generated actions are
never assigned demonstration outcomes as physical ground truth.

[Registered equations, architecture and checks](FLOW_REASONING_RUN.md),
[implementation](../flow_jepa/execution/flow_reasoning.py),
[controller](../flow_jepa/execution/flow_controller.py), and
[primary-literature/code review](REASONING_DIFFUSION_REVIEW_20261010.md).
LARC motivates downstream-usefulness supervision; GPC, LeWAM and recurrent VLA
work limit generic novelty claims. A propose/predict/revise loop alone is prior art.

## Task, data and comparison contract

This is custom image-goal control on 13 MetaWorld v3 manipulation tasks, with a
frozen V-JEPA 2.1 ViT-L encoder and separately trained frozen action-conditioned
world. It is not the published Flow-JEPA dynamics model or an unchanged LeFlow
checkpoint. The complete offline training split contains 7,800 episodes
(1,560,000 recorded primitive transitions); this policy and retrieval bank use
the same 6,222 successful expert episodes as the preceding policies.

One fresh seed 3072 used all eight A800 GPUs, global batch 64, and the same
20,000-update/28,800 aggregate optimization-GPU-second ceilings. All learned
modules count inside that budget. The run sampled 1,280,000 windows with
replacement, not a conventional number of epochs. No extra training stage or
simulator-based training was added. World weights and bank tensors are identical.

The four checkpoint evaluations use the same 104 development cases, eight per
task, selected from the existing 650-episode validation split. Reset identities,
goal episodes and seeds match all saved references. The main metric is task-macro
environment success within **200 primitive actions and 10 seconds of cumulative
controller computation per episode**. Goal construction is expert-screened; this
is not an unconditional reset-distribution result. The 3,200-case final test
remains sealed. No baseline, additional seed or follow-up variant was trained.

Equal training/update ceilings do not imply equal realized FLOPs, convergence or
controller runtime. The [corrected baselines](RELEASE_BASELINE_RESULTS.md) are
frozen shared-world ports on this custom benchmark. LeFlow uses released modules
and CEM the released solver; HWM is a paper-based port because robot code is
unavailable. HWM/CEM are strongly constrained by the controller time allowance.
These scores do not establish native published-benchmark SOTA superiority.

## Results

| Checkpoint | Successes | Timeouts | Training flow loss | Initial final-round diversity | Online corrected-prefix MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5,000 | 79/104 | 10 | 0.17613 | 0.05505 | 0.002123 |
| 10,000 | 75/104 | 10 | 0.11552 | 0.03478 | 0.002023 |
| 15,000 | 76/104 | 9 | 0.09066 | 0.02580 | 0.001897 |
| 20,000 | 76/104 | 10 | 0.09284 | 0.02263 | 0.001793 |

Training loss averages ten logged minibatches in the preceding 500 updates; it
is not a fixed held-out denoising evaluation. Online MAE comes from policy-induced
states. Initial diversity instead pairs the same reset/goal/seed with empty
history and no reused plan; target-route costs were checked identical. It is a
candidate dispersion statistic, not likelihood entropy or causal attribution.

| Selected method | Successes /104 |
| --- | ---: |
| Preserved execution-revision GMM | 80 |
| New candidate-conditioned action flow | 79 |
| Preserved controller-grounded GMM | 79 |
| Preserved latent-revision GMM | 77 |
| Corrected released-module LeFlow port | 21 |
| Released CEM solver | 24 |
| HWM paper port | 5 |

Against the strongest 80/104 reference, the selected new model wins five paired
cases and regresses on six: a one-case net deficit. Wins are dial-turn6,
door-open0/1 and reach4/7; regressions are assembly5/6/7, dial-turn1/2 and
faucet-open0. Single-seed selected development scores cannot establish general
superiority. All reference pairing identities and stored report hashes passed.

| Task | 5k /8 | 10k /8 | 15k /8 | 20k /8 |
| --- | ---: | ---: | ---: | ---: |
| assembly | 0 | 1 | 0 | 0 |
| button-press-topdown | 8 | 8 | 8 | 8 |
| coffee-button | 8 | 8 | 8 | 8 |
| dial-turn | 6 | 6 | 8 | 8 |
| door-close | 8 | 8 | 8 | 8 |
| door-open | 8 | 5 | 6 | 7 |
| drawer-close | 8 | 8 | 8 | 8 |
| drawer-open | 8 | 8 | 8 | 8 |
| faucet-open | 2 | 2 | 4 | 2 |
| handle-press | 8 | 8 | 8 | 8 |
| pick-place | 1 | 1 | 0 | 0 |
| plate-slide | 8 | 8 | 8 | 8 |
| reach | 6 | 4 | 2 | 3 |

## Observed failure modes and their limits

**More training improves demonstration fit without improving task success.**
From 5k to 20k, matched-initial final-round proposal dispersion falls
58.90%, decreasing in all 104 paired cases. Successful-expert
action supervision can favor increasingly narrow alternatives without teaching
failed-contact recovery. The shared data and objective make this a plausible
explanation, not proof that diversity loss caused the success decline. The
previous GMM iteration showed a similar pattern. Pairing 5k with 20k gives three
improvements and six regressions; the decline is not monotonic across checkpoints.
Training longer is not justified by these results alone.

**Prediction remains too optimistic on some failed executions.** At the selected
checkpoint, 15/25 failures have positive mean predicted local
progress but nonpositive actual progress over their final 20 decisions;
9 retain this pattern after correction. Corrected-prefix MAE is
0.002123, versus raw 0.002812. Lower average error does
not guarantee reliable ranking near contact or outside expert-like states.
Only executed prefixes have factual outcome labels; rejected candidates do not.

**The new controller is slower.** Mean decision latency at the selected
checkpoint is 99.42 ms (p95
104.78), versus 86.74 ms for the strongest GMM reference.
Ten episodes exhaust controller time after 188–198 primitive actions, versus
zero GMM timeouts. This does not establish that extra actions would solve them.
Only 71 cases succeed within 100 actions, versus 77 for the GMM. The method needs
better execution efficiency, not just lower imagined cost.

**The feedback path is active, but its benefit is unisolated.** Later rounds supply
the chosen candidate in 1829/5103 decisions
(35.84%). Round-two-to-three candidates change by mean
0.01560 in the logged action-change statistic. The first
round transition also removes recorded/reused proposal overrides, so its change
cannot be attributed solely to reasoning. Monotonic incumbent proxy cost is
guaranteed by retention and is not evidence of physical improvement. There is
no same-weight depth or equal-runtime counterfactual in this run.

## Inspected saved trajectories

All ten complete [selected-5k contact sheets](reports/20261010-flow-reasoning/contact-sheets-step-5000/)
were visually inspected at their original 1536×326 resolution. Each contains
five uniformly sampled rollout frames and the registered image goal, with
readable case/outcome/action labels and no clipping or overlap. The renderer
checked exact initial RGB and saved trajectory lengths. These are measured
rollout images, rendered with zero new model or simulator calls.

- Assembly0 approaches the ring, then moves back; the ring remains on the table.
  It fails after 196 actions with controller time exhausted.
- Pick-place0 approaches the red object, then moves toward the blue goal marker
  while the object remains on the table. It fails at 200 actions.
- Faucet-open3 approaches and changes the handle position, but does not meet the
  success criterion by 200 actions. Reach3 approaches, remains offset and fails.
- Coffee-button0 (35 actions), dial-turn0 (77), door-close0 (63), drawer-close4
  (65), handle-press0 (13) and reach5 (46) are the six saved successes.

These views support an object-completion/recovery failure pattern. They do not
measure grasp force or establish a causal representation/contact-model defect.
The illustrative case IDs were fixed before this run.

## Publication readiness and next research question

This is a concrete implemented research direction, **not yet sufficient evidence
for a strong method paper**. Generic flow-plus-JEPA, recurrence and prediction-
conditioned refinement already have close prior work. The defensible target is
a distinct way to learn revisions that increase executable progress under a
bounded controller. Our current teacher-action flow objective is only a proxy
for that target, and the new candidate has not surpassed our strongest reference.

The next diagnostic would reuse these weights to compare one/two/three rounds
and an equal-measured-time sampling allocation. That would test whether evidence
feedback earns its latency before investing in another training variant. A future
training change should target observed recovery/ranking failures and preserve
useful alternatives; arbitrary generated branches must not receive invented
physical labels. Subsequent causal checks and a separately registered sealed
evaluation would be needed for publication. No such follow-up runs were started.

## Compute and verification

Optimization used **3.165801 GPU-hours**, registered validation
**2.080300**, and training-only preflight
**0.005523**, totaling **5.251625 measured
GPU-hours**. Other held occupancy has a separately reported
0.163174-GPU-hour upper bound, not a cluster bill.
Known cumulative campaign components total 70.395501 GPU-hours;
shared world/cache costs are not charged twice. The run stopped at the update
limit with zero optimization-budget overrun.

All **21 behavioral checks** passed on the execution source. Full-size preflight
verified finite gradients, frozen world weights, exact registered RGB and the
original latency gate. The final read-only audit verifies all 134 runtime hashes,
clean frozen source, data/world/bank identity, finite best/final checkpoint
tensors, selected-checkpoint identity, 401 unique training logs, four validation
records, paired cases, causal history/reuse, fixed scoring and compute limits.
It made zero model or simulator calls. W&B independently reports `finished` at
20,000 updates. All owned workers exited, and all eight GPUs were empty/idle at
2026-10-10T07:22:07.795332+00:00. Training plus validation took about 40 minutes;
no further experiment is queued.

Frozen execution source: `ca8430ecac26ecf11fdae4307981e095f61c937c`.
Selected checkpoint: `/home/mtxu/adam/LeFlow-experiments/20261010-flow-reasoning/campaign/runs/flow_reasoning_3072/best.pt`.
SHA256: `88d7ba394d7c7050c11d5263f917030ddf8bf9f217bf8c7982f44a3442ddbd3e`.

[Full numeric review](reports/20261010-flow-reasoning/validation-review.json),
[compressed raw records](reports/20261010-flow-reasoning/run/),
[postrun audit](reports/20261010-flow-reasoning/postrun-audit.json),
[compute ledger](reports/20261010-flow-reasoning/compute-accounting.json),
[shutdown readback](reports/20261010-flow-reasoning/final-health.json), and
[W&B](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/6orbayjr).
