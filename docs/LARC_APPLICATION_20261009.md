# Applying latent reasoning to the JEPA planner

October 9, 2026. Research/design review only; no additional model or simulator
calls. **A compact recurrent latent workspace is compatible with our frozen
JEPA features and action planner. BAGEL and its mixture-of-transformer experts
are not prerequisites.** The useful research target is plan revision driven by
execution evidence, not latent recurrence by itself.

## What transfers, and what does not

[LARC](https://rootyjeon.github.io/latent-reasoning-umm/assets/larc.pdf) feeds
continuous transformer hidden states back as reasoning inputs and conditions
generation on that context. Its supervised curriculum is followed by a GRPO
stage using paired improvement in a frozen reference generator's prediction
loss plus a diversity term. This suggests learning internal computation from
its downstream usefulness. It does not show that image-prediction improvement
implies executable robot progress. Its text/image training data and large-model
training recipe are outside our allowance; the full recipe is not a drop-in.
The project page's code, arXiv and model buttons resolve to placeholders as of
this check. We inspected its paper, not a released LARC implementation.

Official [BAGEL code](https://github.com/ByteDance-Seed/Bagel/blob/a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f/modeling/bagel/bagel.py)
uses separate token/image projections, generation conditioning through the
transformer/cache, and flow-velocity output. Its standard `generate_text` loop
decodes token logits; it does not itself implement LARC's new feedback loop.
[Coconut's code](https://github.com/facebookresearch/coconut/blob/27273cb8cca4bb763c041a63b036d0c3b7cbbb48/coconut.py)
does explicitly replace latent-token input embeddings with the preceding hidden
states. These are useful interface precedents, not code to copy unmodified.

Our physical state representation `z` (32 spatial JEPA tokens), an internal
reasoning workspace `h`, and an action proposal are different objects. Arbitrary
thought vectors must not be fed to the frozen world as purported physical
states. A trainable interface must connect `h` to the action proposer/scorer.
The world remains `F(z,a)`. A flow head could be conditioned as
`v_theta(u_tau, tau | z, q, goal, h)`; our currently measured best head is a
Gaussian mixture and can accept the same context. Adding recurrence does not
require simultaneously changing the current action head.

JEPA's predictive features are not a pretrained language reasoning engine.
Our small head would learn the usefulness of internal iterations from the
permitted trajectories. More thought steps and more action-denoising steps
are distinct compute choices; neither automatically extends physical horizon.

## Closer robotic prior art

[RD-VLA](https://arxiv.org/html/2602.07845v1), sections III-B/C, already uses a
weight-tied recurrent action head, repeated observation conditioning, randomized
training recurrence and truncated backpropagation. It stops inference based on
action changes. This supports recurrence without explicit text, but action
convergence is not proof of progress. Its changing execution horizon is not
compatible with our fixed two-primitive-action interface without a protocol
change. We should not borrow its reported latency or depth gains as estimates
for our hardware, model or benchmark.

[MPCoT](https://arxiv.org/html/2606.06245v1), sections 3 and appendices A–C,
already combines multiple recurrent latent hypotheses, confidence-weighted
aggregation and action/progress/success preferences. Generic multi-path latent
planning, execution-aware scoring and depth/width scaling therefore cannot be
our novelty claim. Its paper describes training-side branch evaluation absent
at inference. We have no recorded physical outcomes for arbitrary generated
action branches, so those labels cannot be assumed available offline.

The inspected [MPCoT code](https://github.com/Scout-UCAS/MPCoT/blob/7b3759ffed960c19f8abac512f802df927b16c64/prismatic/models/action_heads.py)
implements shared refinement and supports optional external branch rewards.
When these are absent, `_compute_auto_progress_reward` uses cumulative action
agreement and `_compute_auto_success_reward` uses terminal cumulative-action
agreement with the demonstration. These are explicitly offline proxies, not
measured environmental progress/success. The training caller passes optional
batch reward fields. We must audit that data path before treating the public
implementation as reproducing the paper's branch-outcome supervision. Its
default code also differs from paper settings; no reproduction was run.

The earlier [contribution audit](CONTRIBUTION_AUDIT_20261009.md) still applies:
HAC and failure-aware routing already motivate controller-achievable goals.
An execution-error-conditioned recurrent correction is a research hypothesis,
not a certified first use or an established paper contribution.

## Proposed central mechanism

**Use an observation of how the previous action actually failed its prediction
to revise the next plan in a small recurrent workspace.** Keep the existing
supported routes and short controller verification, which already improve our
validation results. Do not merely add a generic reasoner rewarded for making
the same frozen world increasingly optimistic.

For control decision `t`, let the available context contain current image
features `z_t`, goal `g`, retrieved route targets, previous selected target and
executed action prefix, the previous prefix's predicted/observed residual, and
remaining controller time. A small recurrent core updates:

```
h_t^0     = initialize(previous memory, current observed context)
h_t^(k+1) = R_theta(h_t^k, z_t, g, supported routes,
                    bounded-controller response summaries, residual history)
actions, score corrections = decode(h_t^K, current candidates)
```

Thought updates happen inside the decision. Physical progress occurs only
after action execution. The next observed residual refreshes memory; it must
never leak a future observation into the current decision. The first decision
uses an explicitly empty-history mask. Inactive slots are masked, and memory
resets on every episode. Preserve the exact chosen response-prefix handoff.

Use a few width-256 workspace slots with shared weights and a short training
depth schedule, initially up to four iterations. Cross-attend to cached spatial
features, rather than repeatedly encoding images. Reuse current controller
batches as evidence; do not silently multiply the 480-transition search by
every reasoning iteration. Any extra world calls or recurrent work count in
the same controller clock. This is a proposed starting configuration, not a
profiled runtime or an optimized architecture claim.

The distinct intended behavior is: an apparently attractive target that has
repeatedly produced no observed progress should change the internal plan and
the next proposed action/target. It should not simply trigger more samples
scored by the unchanged optimistic model. This must still allow intermediate
setbacks needed for a supported longer route; requiring positive final-goal
distance reduction at every step would introduce a different greedy failure.

## Grounding the learning signal

Use the same training episodes/windows and their actual recorded transitions.
For a recorded action `a*` and a supported target `q`, valid observed labels are:

```
Delta_observed = d(z_t, q) - d(z_(t+1)^recorded, q)
error_recorded = representation_difference(F(z_t, a*), z_(t+1)^recorded)
```

A small action-conditioned progress/error readout can be trained from `h`
against those factual labels. Historical residual inputs come from earlier
recorded transitions within the same available window. The current transition's
outcome is a training target only. Keep demonstrated-action likelihood as the
behavioral anchor; sample reasoning depths during training so an early exit is
also supervised. Initialize the combined head/reasoner from scratch under the
same total method allowance, rather than adding free training to the selected
5k model and calling the comparison equally budgeted.

The transferable LARC idea is a paired usefulness signal: compare the error
with and without the refined workspace for the **same recorded action, target
and transition**. Treat this as prediction-loss reduction, not mutual
information or task reward. The frozen world does not accept `h`, so the new
readout/conditioned policy is essential; merely adding thought vectors leaves
the original world prediction unchanged. Warm up a useful supervised interface
before using any fixed reference comparison. A deterministic recurrent module
can learn through ordinary backpropagation; GRPO requires an explicitly
stochastic policy with valid probabilities and is not automatically appropriate.

For generated actions, observed outcomes remain unknown. Neither copying the
expert next state onto a different generated action nor rebranding frozen-world
predictions as real progress solves that problem. A learned factual error
readout can still fail outside demonstrated support. These are central limits
of the proposed offline adaptation, to be assessed with closed-loop validation.
Validation rollout traces diagnose this design; they must not become training
examples. The current expert-heavy subset also offers limited recovery coverage.

## Why this direction fits the measured failure

The [completed first run](CONTROLLER_GROUNDED_RESULTS.md) reaches 79/104, while
19 of its 25 failures end with positive predicted but nonpositive observed mean
progress over their last 20 decisions. This supports investigating contradiction
between imagined and observed execution. It does not show that missing latent
reasoning caused the failures, nor rule out spatial precision, action proposal,
retrieval coverage or metric deficiencies.

The next decisive outcome remains task success within 10 seconds and 200
primitive actions. Supporting diagnostics should ask whether late stalling,
repeated optimistic prefixes and regression cases decrease, and whether
memory changes the candidate eventually executed. We still cannot claim
counterfactual ranking accuracy without outcomes for alternative candidates.

Test-time depth of 1/2/4 is a future scaling axis, not an experiment launched
by this review. Evaluate any scaling at matched total wall-time allowances and
against spending the same time on ordinary candidate search. More iterations
may converge to the same wrong action, suppress useful diversity or leave too
little time to act. A larger reasoning depth is not automatically better.
Our current 13 single-task MetaWorld setting cannot establish long-horizon
compositional generalization; that would require a separately registered task
suite after the current mechanism works.

## Verified source scope

Primary LARC paper/project, RD-VLA method, MPCoT method and reward appendices,
and the official BAGEL, Coconut and MPCoT source interfaces were inspected.
Pinned code revisions are in the links above. No LARC weights/code were found
through its resource buttons, and no upstream model was run. The answer is
therefore architectural feasibility plus an evidence-led proposal, not a
claimed reproduction or a result for latent reasoning in this project.
