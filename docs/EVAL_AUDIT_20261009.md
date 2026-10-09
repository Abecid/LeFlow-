# Validation and implementation audit — October 9, 2026

**The current run does not establish an advantage for our proposed method. A
confirmed architectural limitation affects both flow planners. These are custom
method-family adaptations, not faithful published-SOTA reproductions.**

This audit responds to the user's request for current metrics, failure modes,
implementation fidelity, and evidence behind the proposed approach. It uses the
01:03 UTC validation snapshot and a read-only CPU checkpoint diagnostic at 01:06
UTC (October 8, 18:03–18:06 PDT). Active execution SHA is
`56419ed14cf127d4b7a8ab09a9d68e6681d2edc0`. No training configuration, checkpoint,
active source, seed, or evaluation outcome was changed; no ablation was launched.

## Measured results

All rows use the same 104 validation resets, eight per training task. HWM remains
in training. The final 3,200-reset test is not included in this audit.

| Method | Success at 5k / 10k / 15k / 20k | Selected success | Mean controller-call latency | Failed episodes that time out |
| --- | --- | --- | --- | --- |
| Ours | 23 / 17 / 16 / 19 | 23/104, 22.12%, step 5k | 173.47 ms | 81/81 |
| LeFlow adaptation | 26 / 20 / 25 / 27 | 27/104, 25.96%, step 20k | 212.34 ms | 77/77 |
| HWM adaptation | 7 / 2 / pending / pending | 7/104, 6.73%, step 5k so far | 156.89 ms | 97/97 |
| CEM, shared selected world | World changes during its validation curve | 7/104, 6.73%, world step 20k | 280.27 ms | 97/97 |

Ours trails the primary baseline by 3.85 percentage points: 7 paired resets favor
ours, 11 favor LeFlow, 16 succeed for both, and 70 fail for both. This small,
checkpoint-selected validation set cannot establish a statistically reliable
advantage or disadvantage, nor generalization to held-out tasks. Ours is about
18.3% faster per controller call, but that efficiency has not translated into
higher success.

Our selected model succeeds only on dial-turn (4/8), door-close (6/8),
drawer-close (8/8), and handle-press (5/8). Nine tasks have zero successes. LeFlow
adds coffee-button (4/8) and reach (1/8), with 3/8 dial-turn, 7/8 door-close,
4/8 drawer-close, and 8/8 handle-press. Eight observations per task are insufficient
for strong per-task conclusions.

The common world model does learn action-dependent short-range dynamics:
validation prediction loss is 0.01751 versus 0.06280 for persistence, and it
identifies recorded action chunks among 16 choices at 88.28% versus 6.25% chance.
These checks do not validate generated-state extrapolation or 30/60-step rollouts.

## Confirmed architectural limitation: injected noise cannot be removed

`JointPlanner.forward` predicts each 1,024-dimensional state velocity through
`Linear(256,1024)`, without a full-dimensional input skip or noise-cancellation
term. Each velocity is `W h + b`. For a fixed trained checkpoint, every Euler
update therefore lies in `span(W,b)`, of dimension at most 257. At least 767
directions of each initial Gaussian sample remain unchanged throughout sampling.
Conditioning, attention, more Euler steps, and more candidates cannot remove
this fixed-checkpoint constraint.

This affects **both** `joint_flow_consistent` and `leflow_adapted`. Read-only CPU
checks on their selected real checkpoints used the actual eight-step sampler:

| Checkpoint | RMS change in noise projected outside span(W,b) | Maximum absolute change |
| --- | --- | --- |
| Ours, step 5k | 7.40e-7 | 5.36e-6 |
| LeFlow, step 20k | 4.88e-7 | 3.81e-6 |

The residual is unchanged to floating-point precision. This is a structural
support restriction, not an inference from poor success. It makes the current
flow comparison unsuitable for concluding that a properly parameterized latent
flow method succeeds or fails. Existing small-fixture gradient tests verify
gradient plumbing but do not exercise this production dimensionality failure.

## Additional failures and design risks

1. **Controller time exhaustion.** Failed episodes stop after median 114 primitive
   actions for ours, 92 for LeFlow, 126 for HWM, and 70 for CEM, below the common
   200-action ceiling. Equal controller-time allowances are enforced, but the
   experiment is strongly limited by computation per action. A timeout records
   how the episode ended; it does not prove that extra time would solve it.
2. **Better surrogate metrics do not mean better behavior.** Ours' generated
   consistency training loss falls from 1.948 at 5k to 1.765 at 20k, and observed
   subgoal cosine distance falls from 0.310 to 0.252, while success falls from
   22.12% to 18.27%. HWM has much smaller observed distance (~0.029) but poor
   success. The observed metric follows repeated replanning, so it is not a
   controlled replay test of an unchanged proposed action chunk. These results
   show a weak proxy, not proof of a particular exploitation mechanism.
3. **Local consistency is not a full reachable trajectory.** Ours scores each
   five-step bridge by restarting the world model at a generated state, then
   averages bridge discrepancies. Only the first bridge starts at the actual
   observation. It does not rank a continuous action rollout from the real start
   by final-goal progress. LeFlow's adaptation does use such a continuous rollout
   for ranking. Locally consistent hypothetical states may still compose poorly.
   This is faithful to our specified local regularizer, but a weakness of the
   design; its causal contribution has not been isolated.
4. **Goal representation mismatch.** The fine predictor learns causal video-history
   features, while final goals use one image repeated through the clip. The last
   generated-consistency bridge is penalized directly against that static-goal
   feature. A small CPU cache diagnostic across 13 validation tasks and three
   timepoints each measured mean same-endpoint history/static cosine distance
   0.1092, versus 0.0340 for an ordinary five-control-step history change. This
   establishes a representation gap; it does not quantify its effect on success.
5. **Training/deployment discrepancy.** Generated consistency uses four Euler
   steps, while evaluation uses eight. All generated endpoints receive input
   gradients through the frozen world; there is no accidental detach of the
   main consistency mechanism. Nevertheless, the sampler discrepancy and
   out-of-distribution generated states weaken its calibration.

No videos were reviewed in this audit. Claims about grasp misses, collisions,
object dropping, or specific physical behavior would require rollout inspection.

## Fidelity and novelty correction

The proposed implementation does contain joint state/action flow matching,
generated-path consistency with a frozen differentiable world, and short-range
CEM refinement. The principal algorithmic ingredients are wired, but the output
parameterization prevents calling the implementation sound as a full-dimensional
flow generator.

The LeFlow adaptation follows the public code's observed-path inverse-dynamics
consistency pattern. However, **LeFlow's paper itself already describes consistency
on generated transitions and shaping the planner's distribution**. Its currently
published `step_batch` instead passes observed encoded `z_path` to the consistency
function. Our earlier distinction between generated consistency and LeFlow must
therefore be narrowed to the inspected implementation, not asserted as novelty
over the paper. The local adaptation also adds short CEM refinement and changes
encoder, architecture, horizons, and sampling budgets. Sources:
[paper §3.5](https://arxiv.org/html/2608.24855v1#S3.SS5),
[official training code](https://github.com/hsiangwei0903/LeFlow/blob/main/train_latent_planner.py).

HWM here uses fixed five-step chunks, an MLP macro-action encoder, and a simpler
coarse predictor. The robot method in the paper uses variable temporal segments,
a transformer action encoder, and waypoint/history conditioning. Its low score
here cannot be presented as the published method's performance.
[HWM](https://arxiv.org/html/2604.03208v2)

The common predictor also uses one causal-history embedding rather than the
Planning Limits paper's four-state history. Our CEM starts afresh each replanning
call rather than shifting the previous solution, and executes its best sampled
candidate rather than the final elite mean. Matching the task list and nominal
horizon does not reproduce the paper's comparator.
[Planning Limits](https://arxiv.org/html/2609.39235v1)

## What the evidence supports next

There is no evidence that the current joint flow design is the best available
approach. LeFlow is the strongest completed learned method in this campaign's
validation. The current run is useful diagnostic evidence, not a SOTA result.

The literature still supports useful, locally reachable subgoals: Planning Limits
finds a large benefit from oracle nearby subgoals, and HWM identifies local
executability as a bottleneck. Neither result establishes that our particular
joint flow objective is optimal. FF-JEPA obtains strong results with a
deterministic hierarchy as well as a generative one. Qantara emphasizes matching
training queries to inference queries. LeWAM's direct future-feature predictor
slightly outperforms its flow alternative; Flow-JEPA explicitly does not isolate
joint-trajectory prediction from stochastic flow training. Thus adding flow or
more joint prediction alone is not a supported guarantee of improvement.
[FF-JEPA](https://arxiv.org/html/2606.09311v1),
[Qantara](https://arxiv.org/html/2607.04978v1),
[LeWAM](https://arxiv.org/html/2609.27455v2),
[Flow-JEPA](https://arxiv.org/html/2608.29029v3).

Priority should be a valid, efficient latent-subgoal planner: remove the confirmed
noise-support restriction for both flow methods, align reachable-state and image
goal representations, and validate that proposal ranking predicts real progress.
Preserve the shared data and resource limits. A full-dimensional skip or suitable
noise-cancellation parameterization may address the support issue without simply
quadrupling transformer width; the exact repair needs verification. Fixing known
defects takes precedence over longer training, extra seeds, or ablation sweeps.
This audit does not silently change or restart the frozen campaign.

Evidence: [validation summary](reports/20261007-joint-flow/audit-20261009/validation-summary.json),
[checkpoint and feature diagnostic](reports/20261007-joint-flow/audit-20261009/structure-check.json),
[read-only diagnostic source](reports/20261007-joint-flow/audit-20261009/check_structure.py).
