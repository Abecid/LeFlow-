# Execution-calibrated action-flow guidance, trained for adaptation

October 10, 2026 (Pacific). The user requests returning to our stronger action-flow
framework, applying QGF-style guidance and training the mechanism to benefit from
test-time training. This authorizes **one fresh `guided_ttt` candidate** under the
existing comparison contract. It does not authorize a sweep or baseline rerun.

## What is being claimed and tested

TTT **does train during inference**: temporary weights are fitted from observations
within the current episode, then reset. Offline meta-training can make those fits
useful for subsequent decisions. A closed-form ridge fit is still a learned-weight
update; it need not call an optimizer object's `step()`.

Our prior `progress_ttt` already combined offline meta-training with episodic TTT
on our action-flow framework. It scored78/104, below the earlier80/104 controller.
Direct LeFlow-TTT later scored29/104 versus its preserved21/104 port. Returning to
our framework is therefore a continuation, not the first application of TTT here.

The specific new hypothesis is: **a factual execution-error fit can improve
generative action refinement when its outer training objective explicitly tests
the resulting action gradient, and its online use is gated by predictive evidence
from earlier observations.** The module is an execution-calibrated prefix-cost
critic, not an RL Q-function. Its success is unknown until measured.

This remains action generation plus JEPA verification. LeFlow instead generates
latent paths and decodes actions through inverse dynamics. We should describe the
method as a JEPA-guided action-flow planner, not an unchanged LeFlow extension.

## Literature: what is adopted and what is different

- [QGF](https://arxiv.org/html/2606.11087), official
  [sampler at60ca92e](https://github.com/zhouzypaul/qgf/blob/60ca92e70aa59c23cd016e5fd27f1c99ed00984f/agents/qgf.py):
  evaluate a critic gradient at the clipped clean-action Euler estimate and use
  the identity-Jacobian approximation. We adopt that sampling mechanism with a
  cost-minimizing sign. We do not reproduce its IQL reward/value training, target
  critic, or claim exact sampling from a tilted distribution.
- [PreferenceFlow](https://arxiv.org/html/2609.36872): guidance gradients need
  explicit constraints, including bounded magnitude and suppression near expert
  behavior. We use a drift cap and an expert-gradient penalty. Its human
  intervention preference pairs are unavailable here and are not fabricated.
- [LaCT](https://tianyuanzhang.com/projects/ttt-done-right/): training through
  fast-weight updates motivates the outer/inner split. We use a small exact ridge
  memory, not its nonlinear large-chunk implementation or throughput claims.
- [SCOUT](https://arxiv.org/html/2609.36107): an observed-outcome inner objective
  and future-action outer objective support learning adaptation for decisions.
  Our scalar progress critic and guided action flow differ from its belief model
  and GMM policy. Transfer to our contact/stall failures is a hypothesis.
- [Feedback World Model](https://arxiv.org/html/2605.15705) already combines
  observed-error correction with diffusion guidance; [AdaJEPA](https://arxiv.org/html/2606.32026)
  already adapts JEPA during MPC. Generic feedback, JEPA, diffusion or TTT is not
  our novelty claim. A distinctive effective decision-training mechanism and
  supporting evidence would still have to be established.

## Mechanism

Keep the frozen V-JEPA2.1 encoder, frozen spatial world, retrieved training routes,
two candidate/evidence/revision rounds, five-block action flow and two-primitive
executed prefix. All new learned parameters start fresh; no previous policy
checkpoint is used for this comparison.

For a current state `z`, supported target `g`, and executable two-action prefix
`a`, a small network produces spatial features `h(z,g,a)`. State projections are
cached across denoising steps. A base head predicts each patch's **world-model
prefix cost**. A unit-norm key `k(h)` queries a rank32 fast correction vector.
The scalar guidance energy is the mean of predicted patch costs plus bounded,
signed error correction, divided by the existing0.02 error scale.

The base is trained on the frozen world's prediction for recorded actions. The
correction is trained on actual minus world-predicted patch cost for those same
actions. Generated actions never inherit the demonstration's outcome. The cheap
base is an additional surrogate, so surrogate error is explicitly logged and is
a possible failure mode. Actual candidate ranking still runs the full frozen
world and uses conservative positive-only correction penalties.

**Online inner fit.** The last four completed transitions give at most128 patch
observations. Fit ridge weights about a learned prior to their observed world
prediction errors, using the same prefix-sensitive keys as the guidance energy.
For trust, separately fit only the earlier support and predict the most recent
observed residual. Compare that prediction with the prior. The nonnegative relative
MSE improvement gates the full-support weight delta; fewer than two transitions
or a worse prediction yields zero trust. The denominator has a0.01 floor in
normalized error units. Trust is detached during outer optimization.

This is a causal historical reliability heuristic, **not calibrated uncertainty**.
It can reject useful updates after distribution shifts or accept a correction
that fails on the next state. It does not establish counterfactual accuracy.
The same trusted weights condition generation, guidance and prefix ranking.
There is no cross-episode replay or parameter accumulation.

**Guided generation.** At flow time `t`, compute

`a_clean = clip(a_t + (1-t) v_theta(a_t,t), -1, 1)`.

Evaluate the compact energy gradient at `a_clean` with no derivative through the
velocity or solver. Subtract the gradient from velocity, with per-prefix RMS
bounded at0.25. Apply this on the last four of eight Euler steps, only to the
executed prefix and two of four proposal slots. Keep the other two slots as
ordinary flow samples within the existing candidate count. Retrieved actions and
shifted previous plans retain their existing slots. Suffix generation remains
flow-conditioned; the guidance never assigns an outcome to an unexecuted suffix.

The direct added drift integrates to at most0.125 prefix RMS across the four
steps. This does **not** bound final deviation from an unguided trajectory because
the flow velocity can respond to changed intermediate states.

**Offline outer training.** Preserve the original flow and factual calibration
losses. Add base-cost distillation, a next-recorded-action denoising loss after
the critic-gradient update, an adapted-versus-prior action-regression penalty,
and a small gradient penalty at recorded expert actions. The action loss
differentiates through the energy's action gradient and the causal ridge fit.
The clean estimate is detached from the actor for this branch, avoiding a
second derivative through the full flow/world. The actor is still trained by
flow matching through the adapted context.

The denoising target is **behavioral supervision**, not an observed physical
outcome for the perturbed action. It can teach imitation-compatible gradients;
it cannot prove that they improve actual recovery or task success. The raw
expert-gradient anchor may also conflict with useful goal-cost gradients.
These tradeoffs must be evaluated rather than hidden behind prediction MAE.

A training-data-only timing check put the three-round controller at112–115ms
per decision, above the100ms average needed to use200 actions in10 seconds.
The registered candidate therefore uses two rounds, with training depths1/2,
to reserve time for guidance and observed-error fitting. This changes the compute
allocation; it is not a claim that fewer rounds are lossless or equally effective.
No environment evaluation or extra training run was used to make this choice.

Registered extra loss weights: base surrogate0.1, guided action MSE0.25,
adaptation regression0.25, expert-gradient penalty0.01. No hyperparameter sweep.
Training and deployment both include guided candidates in deliberation.

## Fixed run and measurements

- One seed3072; all eight available A800 GPUs non-preemptively; global batch64.
- Same6,222 successful expert episodes, train-only route tensors, frozen world
  hash and case inventory as the existing action-flow comparisons.
- Stop at20,000 updates or28,800 aggregate optimization GPU-seconds. All new
  critic, policy and meta-training work counts inside that cap. Equal ceilings
  do not imply equal consumed compute or convergence.
- Four fixed104-case development evaluations at5k/10k/15k/20k; a binding compute
  cap triggers evaluation of the final checkpoint without extending training.
- Primary success: task completion within200 primitive actions and10 cumulative
  controller seconds, including fitting and guidance. Keep highest success,
  earliest checkpoint for ties. All3,200 final cases remain sealed.
- Reuse80/104 execution revision,79/104 flow reasoning,78/104 progressTTT,
  LeFlow21/104, CEM24/104 and HWM5/104. Do not retrain baselines.

Track success, per-task and paired outcomes, timeouts and latency, proposal
dispersion, trust acceptance, prior/adapted observed error, surrogate mismatch,
guidance gradient magnitude, and adapted/prior/unguided action-denoising errors.
World verification uses320 predicted transitions per decision; cheap critic
gradients add8 batched evaluations. Counts alone do not establish equal runtime.

This first run does not establish a test-time scaling curve or isolate each
component's effect. Analyze its failures before any further authorized study.

## Are the existing results already enough for a paper?

The80/104 versus21/104 result is **76.92% versus20.19%, a56.73 percentage-point
development gap** under the registered controller limits. That is substantial
preliminary evidence for this system on this benchmark. It does not establish
that reasoning or TTT helped: their best results79/104 and78/104 are below80/104.

Publication readiness remains conditional:

1. LeFlow is a release-module/shared-world MetaWorld port, not the authors' native
   benchmark/backbone. Its held-out losses were still declining; matched
   convergence is unproven. HWM and CEM are heavily limited by the10-second cap.
2. These same104 development cases have guided repeated iteration and checkpoint
   selection. They are not an independent final generalization result.
3. Our strongest controller includes retrieval, execution correction, stall memory
   and action refinement. A large gap cannot be assigned to a new TTT mechanism.
4. Useful inference scaling requires measured success versus actual compute;
   more gradient/TTT/reasoning steps alone do not demonstrate it.

The current honest claim is a promising fixed-budget development result for a
custom JEPA planning system. A paper about execution-calibrated generative
reasoning needs the proposed mechanism to improve decisions and held-out results
to support its stated scope. More seeds and task breadth can strengthen later
claims, but are not automatically launched in this compute-limited iteration.

Source: `flow_jepa/execution/guided_ttt.py`; config
`config/flow_metaworld_guided_ttt.json`; launcher
`scripts/operations/launch_guided_ttt.sh`. Training/evaluation use the existing
isolated coordinator. See the run report for verified launch status and results.
