# Research angle after the first progress-TTT result

October10,2026. Recommendation: prioritize **reliable execution-grounded guidance
of the generative flow**, measured by task success at a fixed controller allowance.
This is a design judgment based on the current evidence, not a demonstrated gain.

## Current implementation and thesis

The current candidate uses a frozen JEPA visual encoder and action-conditioned
predictor, retrieved targets, an action-flow policy, a candidate-conditioned latent
workspace, and episodic fast weights correcting goal-relative execution errors.
Its thesis is that recent observed errors can improve future action proposals and
their selection without changing the backbone. The [completed run](PROGRESS_TTT_RESULTS.md)
reduces local error but scores78/104 versus the strongest80/104 reference.

[LeFlow](https://arxiv.org/html/2608.24855) generates latent state paths, decodes
them through inverse dynamics and verifies the resulting actions with world
rollouts. Our current generator instead produces actions directly. Both exploit
JEPA prediction and generative planning, but they are distinct formulations.
A direct LeFlow extension would retain its latent-path generator and guide it
through the inverse-decoder/world composition. That is not what we have trained.

The sharper research question is: **Can observed execution errors teach a
JEPA-guided flow planner when and how to refine a plan, so additional inference
improves actual progress rather than exploiting an inaccurate latent score?**
Reasoning and adaptation should support that objective. Merely adding either
module, or combining flow with JEPA, does not establish novelty.

## Recent methodological anchors

- [QGF, June2026](https://arxiv.org/html/2606.11087) is the best immediate sampler
  reference for our action-flow formulation. Estimate the clean endpoint with
  one Euler extrapolation, evaluate a critic's action gradient there, and add it
  to the velocity. The default drops the velocity Jacobian. I inspected
  [release code60ca92e](https://github.com/zhouzypaul/qgf/blob/60ca92e70aa59c23cd016e5fd27f1c99ed00984f/agents/qgf.py):
  it clips the endpoint estimate, uses the target critic, and defaults to
  `apply_jacobian=False`. Our cosine progress score is not its IQL critic.
  Adopting its sampler does not reproduce its RL method or guarantee exact
  sampling from an exponentially tilted policy.
- [PreferenceFlow, September29](https://arxiv.org/html/2609.36872) explicitly
  extends QGF with preference gradients and regularization. Its experiments show
  large degradation when gradient bounds/near-expert suppression are removed;
  fitting scores alone does not ensure useful guidance. Its human-intervention
  supervision is absent from our fixed data. Paper reviewed; no release code
  inspected. Do not fabricate equivalent preference labels for generated actions.
- [TraceFlow, September17](https://arxiv.org/html/2609.20646) uses bounded,
  progress-aligned attraction/repulsion from retrieved success/failure traces.
  Its gains are task-dependent, and repeated trace accumulation adds deployment
  experience. That cross-episode protocol is outside our independent-case test.
  Paper reviewed; no release implementation inspected.
- [Feedback World Model, May2026](https://arxiv.org/html/2605.15705) already
  corrects latent predictions with observed errors and uses action-aware diffusion
  guidance. [FBFM, July2026](https://arxiv.org/html/2607.29235) already injects
  aligned execution feedback into active flow generation. These are direct
  novelty constraints, not evidence that generic feedback-guided flow is new.
- [One-Step Flow Policy, March2026](https://arxiv.org/html/2603.12480) is an
  efficiency reference: interval self-consistency, EMA self-distillation,
  flow anchoring and self-guidance support few/one-step actions. Its from-scratch
  setup avoids a separate pretrained teacher. Paper reviewed; no author training
  release verified. Reported generation speedups are not our controller speedups.

## Proposed next mechanism, not implemented

Use a supported action proposal, estimate its clean form, and evaluate its
short executed prefix with the JEPA predictor. A goal-relative correction learned
from recorded transitions and updated only from this episode's observations
adjusts the guidance energy. Bound the action correction and reduce guidance
when the current context lacks supporting evidence. Feed the resulting evidence
back into generation. The unresolved contribution is a reliable way to learn this
decision-sensitive correction and its trust, not the QGF update itself.

Our existing ridge memory and scalar MAE do **not** establish calibrated uncertainty
or correct action gradients. Training supervision must remain factual: recorded
actions have outcomes; arbitrary candidates do not. Successful expert data can
anchor supported behavior, but it cannot supply counterfactual recovery labels.
Validation failures remain diagnostics, not added training examples.

For a direct latent-path LeFlow formulation, the analogous objective would inspect
the outcome of decoding and executing the prefix, rather than the generated goal
endpoint, which is fixed by construction. Decoder/world gradients would need
their own reliability checks. This transfer is a research hypothesis and should
not be mixed into the current action-flow claim.

## Priority and evidence required

1. Research priority: make guidance improve executable decisions, with JEPA
   providing compact consequence predictions and flow supplying multimodal plans.
2. Immediate engineering priority: profile total controller time and remove the
   overhead crossing the approximate100ms/decision threshold. Distillation is
   justified only if flow generation contributes enough of the cost; the current
   controller also performs480 predicted transitions per decision.
3. After a quality gain, measure success against actual latency/world calls across
   inference budgets. More solver steps, candidates, reasoning rounds and online
   weight updates are different compute axes, not interchangeable “scaling.”
4. Use distillation to preserve an effective planner's decisions cheaply once
   useful refinement is demonstrated. Extra teacher/adapter training must count
   against the method budget.

A publication claim needs a distinct mechanism, improvement beyond the strongest
reference, held-out success and a measured success/compute curve. Existing104-case
development results establish none of those claims for the new fast memory.
No new run, ablation, baseline retraining or scaling sweep was launched in this
research-review turn.
