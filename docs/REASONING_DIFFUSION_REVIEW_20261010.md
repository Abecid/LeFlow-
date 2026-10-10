# Reasoning for generative control with JEPA and test-time scaling

October 9 Pacific / October 10 UTC, 2026. Literature review and proposed design,
not a registered or executed experiment. The current best checkpoint remains
execution revision at 80/104 development success, with a GMM action head.

The user's earlier LARC suggestion motivated our latent-workspace experiment.
That experiment transferred recurrence and paired calibration supervision; it
did not implement LARC, a diffusion head, or a test-time scaling evaluation.
This proposal explicitly returns to a flow-based action generator.

## Recommendation and actual implementation gap

Adopt a small **reasoning–generation–prediction loop**. A latent workspace should
inspect the consequences of generated action candidates, then condition the next
generation round. JEPA provides compact observations and predicted consequences;
flow matching models alternative action chunks; reasoning updates the generation
context from those consequences and earlier observed execution discrepancies.

In the current code, `RevisionPolicy.prepare` performs every workspace iteration
before candidate search. `AlignedController.plan` then reuses the same thought
vectors in all three search rounds. The reasoner sees prior real execution errors,
but does not inspect fresh candidate rollouts between its thought steps. A
rollout-conditioned update would change this interface concretely.

The hypothesis is that additional learned computation can produce more useful
action proposals per unit of controller time than extra independent samples or
unchanged CEM refinement. This is a research hypothesis, not established novelty
or evidence that the method will beat the existing controller.

## Relevant primary sources and transfer limits

| Source | Applicable idea | Boundary for this task |
|---|---|---|
| [LARC paper](https://rootyjeon.github.io/latent-reasoning-umm/assets/larc.pdf) and [project](https://rootyjeon.github.io/latent-reasoning-umm/) | Hidden-state reasoning conditions the continuous generator; paired prediction-loss improvement measures its usefulness | BAGEL, text/latent curriculum and GRPO are not required interfaces for our small controller. Its image-generation results do not demonstrate physical control |
| [RD-VLA](https://arxiv.org/html/2602.07845v1), [MPCoT](https://arxiv.org/html/2606.06245v1) | Shared recurrent refinement, depth variation and multiple latent hypotheses | Existing prior art; our offline data do not supply true outcomes for arbitrary branches |
| [ELASTIC](https://arxiv.org/html/2606.31132) | Allocate compute between denoising depth and parallel candidates | Its learned scheduler uses a verifier and offline/hybrid RL rollout training. It is not free offline-only scheduling for us |
| [Iterative Partial Refinement](https://arxiv.org/html/2605.19317) | Re-noise and regenerate selected portions while retaining the rest | Requires mixed-noise conditional training. Structured image-generation evidence does not establish robot recovery |
| [ThinkJEPA](https://arxiv.org/html/2603.22281v2) | Couple a reasoning representation to JEPA prediction | Its VLM guidance and forecasting results are a larger alternative, not proof of closed-loop MetaWorld improvement |
| [LeWAM, Hegde et al.](https://arxiv.org/html/2610.12407) | Flow policy plus JEPA; search within policy-supported generations | Generic JEPA/flow integration and policy-noise search are already explored |
| [Generative Predictive Control](https://arxiv.org/html/2502.00622) | Improve generative proposals using predicted consequences | A propose–predict–refine loop alone is already prior art |
| [Compositional Visual Planning](https://arxiv.org/html/2603.02646) | Couple short segments through estimated clean boundaries | A later long-horizon direction; requires a suitable trajectory generator and a compositional evaluation protocol |

Other close precedents remain [Feedback World Model](https://arxiv.org/html/2605.15705),
[FBFM](https://arxiv.org/html/2607.29235), and the separate
[Flow-JEPA dynamics paper](https://arxiv.org/html/2608.29029v3). They already cover
feedback-guided generation and flow-based JEPA prediction. The specific learning
mechanism and evidence for useful deliberation must carry any contribution claim.

LARC's practical transferable training signal is the paired difference in a
reference generator's flow losses with and without reasoning, under shared noise
and flow time. For our adaptation, call this a supervised usefulness proxy. It is
not automatically physical reward, calibrated information gain, or an exact
likelihood difference. LARC's text spans provide stochastic policy probabilities
for GRPO; deterministic latent recurrence alone does not provide that interface.

## One proposed first candidate

Retain the frozen encoder/world, cached spatial states, supported route bank,
five-block action horizon and two-primitive-action execution prefix. Replace the
GMM proposal/CEM-update mechanism with a compact conditional action flow plus
shared workspace updates. Do not simultaneously replace the world, introduce a
large VLM, or generate pixels. All proposed new trainable modules start from
scratch and count within the same method allowance.

For each supported target `q`, let `c = (z_t, q, g, observed_history)` and keep
physical state `z`, action plan `A`, and internal workspace `h` distinct.
The history contains only already observed transitions and their prediction
residuals. Initialize `h_0` from `c`. At round `k`:

```text
A_k       = FlowSample_theta(noise_k | c, h_k)
Zhat_k    = FrozenWorld(z_t, A_k)
evidence  = spatial_summary(A_k, Zhat_k, q, calibrated_costs)
h_(k+1)   = R_phi(h_k, c, stop_gradient(evidence))
```

`FlowSample` denotes a complete numerical integration to action space, not an
untrained intermediate noisy action passed to the world. Each round has several
independent candidates; their individual identities and action/outcome pairing
must survive summary construction. Predicted consequences remain explicitly
imagined. Only execution followed by observation supplies factual feedback.

Use the existing prefix/terminal selection and causal history semantics. Retain
an incumbent so later rounds can return an earlier candidate. Its score must
remain comparable across rounds: use a fixed scoring context derived from the
current observation and real history, separate from the evolving deliberation
workspace. This prevents a candidate from appearing improved merely because its
scorer changed. Training still updates this scorer from factual transitions.
The world is frozen, and the first candidate detaches imagined evidence rather
than optimizing actions through a world-model gradient for a better proxy score.

Begin with whole-chunk flow generation. Selective suffix repair from IPR is a
follow-up if logs show that full regeneration destroys useful partial plans. It
requires explicit per-block noise/masking training, with the complete action
sequence rolled out again after an edit. A preserved suffix need not remain
valid when an earlier action changes.

## Supervision for useful reasoning

For recorded action chunk `A*`, draw `s` and Gaussian `epsilon`, and construct:

```text
A_s = (1-s)*epsilon + s*A*
ell_k = mean_squared_error(v_theta(A_s, s | c, h_k), A* - epsilon)
```

Train the shallow and deeper contexts on the same recorded target, noise and
flow time. A candidate supervised objective is:

```text
L = mean_k(ell_k)
  + lambda_use * mean_{k>0} max(0, ell_k - stop_gradient(ell_0))
  + lambda_cal * L_factual_error
```

This adapts LARC's downstream-usefulness principle using ordinary backpropagation.
It is not a reproduction of its frozen-reference GRPO objective. The shallow
context is supervised too; the hinge provides no gradient that rewards worsening
the shallow reference. Depths one through three proposal rounds should be covered
in training; any initial shallow warmup is included inside the fixed update/GPU
ceiling. Loss coefficients and solver settings remain preflight/registration
choices, not validation-sweep parameters or values established by this review.

Critical separation: construct `h_k` through self-generated candidates from
independent planning noise and causal context. Never feed the teacher-forced
`A_s`, recorded current action, or recorded future outcome into its deliberation
history. The flow loss legitimately sees noisy target actions; the reasoner must
not obtain that privileged denoising view at training but lose it at deployment.
The supplied goal and supported target remain allowed conditioning inputs.

Factual calibration queries use recorded actions with their actual recorded
prefix/terminal outcomes. Generated candidates cannot inherit those outcome
labels. Initially the same 6,222 successful expert episodes supply policy
training. Recovery coverage remains limited; adding random-episode calibration
or interaction data would be a disclosed later change, not silently free data.

The usefulness loss can improve demonstration fitting without improving recovery.
No guarantee of monotonic task improvement follows. Avoid a latent cosine-diversity
reward as a substitute for useful action alternatives: different thought vectors
can decode to the same action, and varied actions can all fail.

## Test-time scaling without unbounded compute

Distinguish three axes: deliberation rounds `K`, flow solver evaluations `S`, and
parallel candidates `N`. These differ from the physical action horizon. A control
policy must trade them against latency and the time remaining to execute actions.

The existing maximum is eight targets, four candidates each, three world-search
batches and five blocks: `8*4*3*5 = 480` predicted world transitions per decision.
Reuse that allowance for at most three proposal–prediction rounds, not three
reasoning rounds each containing a full 480-transition search. Reasoning, flow
integration, scoring, retrieval and encoding all remain in the controller clock.
Equal world-transition counts alone do not imply equal wall time or FLOPs;
preflight must measure the added flow work on the actual hardware.

A future training run would preserve one seed (3072), global batch size 64,
at most 20,000 updates or 8 aggregate optimization GPU-hours, the same
train/validation cases, and the four primary validations with a 10-second
controller allowance and 200 primitive actions per episode. Extra training-side
candidate/world computations are charged inside that allowance, so fewer updates
may fit if the GPU-time cap binds. The frozen 80/104 GMM system remains a reference.

After a promising primary result, a small inference-only study can use the same
selected weights at one/two/three deliberation rounds. Hold solver accuracy and
candidate width fixed for that depth curve. Then compare an equal measured time
allocation spent on ordinary sampling/refinement; otherwise more computation
could explain any gain. Existing baseline weights can be reevaluated at any
new common allowance without retraining. Extra evaluations still consume compute
and must be reported. Do not open sealed tests during design.

The target evidence is higher environment success per controller budget, with
fewer optimistic stalls and better recovery on paired cases. Record which round
changes the executed action, useful candidate diversity, chosen-prefix calibration,
timeouts, actual controller time and per-task failures. Lower flow loss, lower
imagined goal cost, thought-vector changes, or a wider search alone are not
evidence of effective reasoning. An adaptive halting rule or learned scheduler
comes later, once a fixed-depth curve supports it.

## Source and code verification

- Rechecked the LARC PDF, including its latent/text interface, supervised
  curriculum, shared-noise reward and GRPO formulation. Project code/model/arXiv
  buttons still point to a placeholder. No released LARC implementation was run.
- Inspected the official IPR sampler and independent noise-time sampler at
  commit `63bd4e2353f31b44f098a8d487e65b67de5a9f21`:
  [sampler](https://github.com/ahn-ml/IPR/blob/63bd4e2353f31b44f098a8d487e65b67de5a9f21/src/sampler/ipr_sampler.py).
  Random region resets and conditional regeneration are present. The code also
  contains task-validity checks and optional immediate stopping; its default is
  false. That optional oracle stopping must not enter robot inference.
- The official compositional-planning repository at
  `197383d497fdd6fb1f4558618a4ca9e659e07ba3` contains only
  [a README with release TODOs](https://github.com/yzhang4179/comp_visual_planning_release/blob/197383d497fdd6fb1f4558618a4ca9e659e07ba3/README.md).
  Its method was reviewed from the paper, not available algorithm code.
- Read ELASTIC's compute-allocation objective and offline/hybrid-RL requirements;
  reviewed ThinkJEPA's primary paper and official release scope. These were not
  reproduced. HF Markdown lookup was unavailable for ELASTIC/IPR; primary arXiv
  text was used. Earlier BAGEL/Coconut/MPCoT code checks remain documented in
  [the LARC applicability review](LARC_APPLICATION_20261009.md).
- Rechecked our actual `RevisionPolicy.prepare` and `AlignedController.plan`
  source to establish the missing candidate-to-workspace feedback path.

This review changes the proposed research direction and documentation only.
It does not claim an implemented flow reasoner, a trained model, a scaling curve,
or authorization to launch an automatic sequence of experiments.
