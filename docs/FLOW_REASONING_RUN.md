# Candidate-conditioned action flow: registered first run

October 9 Pacific / October 10 UTC, 2026. The user authorized implementation,
one fresh training run and periodic evaluation. This registration precedes any
validation results. Method: `flow_reasoning`;
configuration: `config/flow_metaworld_flow_reasoning.json`.

## Hypothesis and contribution boundary

The prior best execution-revision controller scores 80/104 development cases
with a GMM head. Its recurrent workspace is computed before candidate search
and stays fixed throughout that search. The new hypothesis is that inspecting
predicted candidate consequences can train useful revisions of a flow action
proposal under the same physical controller allowance.

This is an adaptation informed by LARC, recurrent VLA work, generative predictive
control and JEPA/flow methods, not an exact reproduction of any paper. See the
[primary-source review](REASONING_DIFFUSION_REVIEW_20261010.md). There is no
language CoT, GRPO, RL interaction training, VLM teacher, or inherited pretrained
policy. Recurrence, flow matching and a propose/check/revise loop alone are not
new contributions. The candidate's claimed mechanism is learned generation
revision from imagined action consequences plus past observed execution errors.

The formulation supports a research experiment, not yet a publication claim.
Publication evidence would need a clear distinction from close prior work,
task-success gains at matched measured inference cost, evidence that candidate
feedback itself is useful, and generalization on a separately registered sealed
evaluation. The existing repeatedly used 104 development cases cannot supply
that final estimate. This first run does not authorize an automatic ablation
sequence or final-test access. A test-time scaling curve is not yet measured.

## Implemented mechanism

- Keep the frozen V-JEPA 2.1 encoder, spatial normalization, separately trained
  frozen world, training-only supported routes, five-block action horizon,
  two-primitive-action execution prefix, and causal history/stall/plan reuse.
- Replace the GMM and CEM parameter update with a conditional rectified-flow
  velocity over a complete 5-by-8 action chunk. A width-256 action head has three
  residual MLP blocks. Euler integration uses eight evaluations from a full
  Gaussian source, with clipping only after integration. Velocity is unbounded;
  the world sees completed actuator-bounded action candidates, never noisy ones.
- Initialize four latent workspace slots with one attention update over the
  observation/target/goal and up to four factual past prediction residuals.
  Use up to three proposal rounds, with a shared recurrent update between them.
- Each candidate evidence token contains its 40 action coordinates, ordered
  five-by-32 spatial goal-cost map, two factual-error-trained corrections, and
  scalar selection cost. Costs are scaled by 100, consistently with the
  inherited history summaries. A learned MLP projects each complete paired
  token; attention treats candidate order as exchangeable. The candidate set is
  not collapsed into independently averaged actions and outcomes.
- Preserve a separate causal scoring context fixed at initialization. Revising
  a thought cannot change the score of an unchanged action/prediction pair.
  The score keeps the earlier equal-weight prefix/terminal cost, conservative
  positive corrections, route cost and observed-stall penalty. Keep the best
  candidate across rounds without extra world calls.
- Use independent Gaussian candidates within each target, and common source
  noise across rounds. Thus the pure generated candidates change because their
  reasoning context changes. In the first inference round retain the earlier
  recorded-route candidate and observed-before-reuse shifted plan. Later rounds
  generate fresh conditional solutions from the same noise; saved incumbents
  remain available. There is no guaranteed monotonic physical progress.

Physical states and internal thoughts are distinct. Imagined candidate outcomes
are detached before the reasoning update. The reasoner and action generator
receive gradients from supervised losses; the world receives none. No direct
world-gradient action optimization or old generated-action/expert-outcome
consistency loss enters this candidate.

## Training objective and leakage boundary

For recorded chunk A, independent Gaussian teacher noise E and uniform time t,
use X=(1-t)E+tA and velocity target A-E. Each visited reasoning depth receives
mean squared velocity error on the same X, t and target. The objective is:

```text
mean_depth(flow_error)
+ 0.25 * mean_deeper_depth(relu(flow_error - stop_gradient(shallow_error)))
+ 0.10 * factual_calibration_error
```

The shallow comparison receives its own supervised loss. The paired term does
not send a gradient rewarding a worse shallow reference. This is a supervised
usefulness proxy, not LARC's frozen-reference GRPO reward or physical success.
Calibration is smooth-L1 error on the factual prefix/terminal correction divided
by the inherited 0.02 scale, with corrections bounded at 0.1.

Training cycles through depths 1, 2 and 3, synchronized across ranks. Each deeper
context sees only self-generated candidates from planning noise independent of
the teacher noise. The current recorded action and actual future prefix outcome
are never deliberation inputs. Recorded future states supplied as the legitimate
goal/subgoal remain allowed conditioning. The factual calibration branch alone
queries recorded actions, with their recorded outcomes used only as labels.
Its supported-target relabeling remains the previous endpoint/future-goal/other-
example-endpoint mixture. All trainable components are initialized from scratch.

There is no extra warmup run or data collection. The balanced depth schedule,
eight solver steps, 0.25 usefulness weight and unchanged 0.10 calibration weight
are fixed before validation, not selected by a validation sweep. Flow's initial
Gaussian candidates may be poor, and expert-only data may not teach recovery;
the first evaluation determines whether this design is useful.

## Comparison and compute contract

- One model seed: 3072. Global batch 64 on all eight available A800 GPUs, physical
  batch eight per rank. Same AdamW schedule/learning-rate ceiling 1e-4, clipping
  and weight decay. Stop at 20,000 updates or 28,800 aggregate optimization GPU
  seconds, whichever comes first. Candidate rollouts, factual calibration and
  variable-depth computation all count inside that ceiling.
- Same 7,800-episode training split; policy and bank use the same 6,222 successful
  expert episodes. Use the execution-revision start/goal sampler, including its
  late-window coverage. Frozen world SHA256:
  `ae3f910da43f0a43dd65945394444a0b11719ed042cdacffacc2faaf824ec5a3`.
- Four fixed 104-case validations at 5k/10k/15k/20k. If the compute ceiling binds
  between milestones, evaluate the last trained checkpoint in the next remaining
  slot, without extending training. There can consequently be fewer than four
  validations; report the actual steps. Select highest task-macro success and
  earliest checkpoint on ties. Restore training RNG after validation.
- Per episode: 200 primitive actions and 10 seconds cumulative controller time,
  including encoding, retrieval, flow integration, reasoning, scoring and world
  prediction. Eight targets by four candidates by three rounds by five blocks
  equals 480 predicted world transitions per decision. Equal world-call counts
  are not equal FLOPs or actual runtime; measure controller time directly.
- Reuse corrected frozen LeFlow 21/104, released CEM 24/104 and HWM paper-port
  5/104; also compare execution revision 80/104 and controller-grounded 79/104.
  These are custom shared-world fixed-budget comparisons, not native published
  benchmark or matched-convergence superiority. See
  [baseline fidelity/results](RELEASE_BASELINE_RESULTS.md).
- No baseline retraining, additional seeds, automatic variants, simulator-based
  training or sealed final test. The 3,200-case final test remains reserved.
  A same-weight inference-only depth/compute-allocation study follows review of
  the primary result rather than silently expanding this run.

## Verification and failure analysis

Before training, test unrestricted-source flow integration and final bounds,
candidate/outcome pairing, sensitivity to imagined evidence, fixed score context,
permutation behavior, teacher-action/future-label exclusion, frozen-world
gradients, all trained depths, observed-only reuse and exact executed-prefix
handoff. Check exactly 480 transitions and evidence updates after the first 160
and 320 transitions. Run the existing controller/history/sampler tests as well.

Training-data-only preflight checks unchanged episode draws, late-window bounds,
finite full-size forward/backward, frozen-world tensors and the original 120 ms
full-controller latency gate. No optimizer update or policy simulator episode is
needed for preflight. Verify renderer identity on the registered training reset.
Use separate preparation/preflight, optimization and validation compute ledgers.

Log flow errors at shallow/deeper depths, paired improvement/regression, factual
calibration, actual success and per-task outcomes. Retain earlier stall/progress
and prefix-error diagnostics; add per-round candidate changes, proposal diversity,
incumbent cost and selected round. Monotonic incumbent *proxy cost* is a selection
property, not proof of useful reasoning. Generated/rejected actions have no
measured physical-outcome labels. Preserve the same ten trajectory case IDs.

Compare paired wins/regressions against saved references, controller timeouts,
optimistic nonprogressing failures and training-length behavior. Later rounds
improving flow loss but failing to change actions or success is a possible
negative result, to be reported rather than repaired with an automatic next run.

## Verified launch

Coordinator 2439186 launched at October 9, 23:38:55 Pacific (October 10,
06:38:55 UTC). Optimization launched at 23:40:53 Pacific on all eight A800 GPUs.
Frozen source: `ca8430ecac26ecf11fdae4307981e095f61c937c`, in
`/home/mtxu/adam/LeFlow-experiments/20261010-flow-reasoning/repo`.
The sparse execution checkout preserves the exact published commit; all 134
runtime file checksums were independently compared with that commit.

All 21 behavioral checks passed. The registered training reset rendered bitwise
identically. Preflight checked 10,000 unchanged task/episode draws, feasible
windows/goals, eight late real examples and causal histories. Full-size
forward/backward preserved the world. The 5,445,162-parameter policy's complete
controller averaged 98.39 ms across five measured decisions, with 9.49 GiB peak
GPU allocation. Preflight used 0.005523 GPU-hours, zero optimizer updates and
zero policy simulator episodes. The route bank reuses its original tensors;
only its protocol/manifest metadata changes.

The first live readback reached update 582 with finite training metrics and
all eight workers active. Evaluation results were still pending. At the update
cap, this run samples 1,280,000 windows with replacement; this is not an epoch
count. Launch/health evidence does not establish task success or useful reasoning.

[Online W&B run](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/6orbayjr).
[Verified launch and configuration](reports/20261010-flow-reasoning/).
