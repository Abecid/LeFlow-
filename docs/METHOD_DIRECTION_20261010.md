# Current method, flow matching and the next research direction

October 9 Pacific / October 10 UTC, 2026. Research review only. No new training,
controller evaluation, ablation or final-test access. Current source/report head
at review start: `95e8ca07b0387f147c0676c56449bc612d9f8ea1`.

## What is better, and what remains unproved

| Selected method | Fixed development success |
|---|---:|
| Our execution revision | 80/104 (76.92%) |
| Our earlier controller-grounded method | 79/104 (75.96%) |
| Our earlier latent revision | 77/104 (74.04%) |
| Corrected LeFlow release modules + spatial port | 21/104 (20.19%) |
| Released CEM, shared world | 24/104 (23.08%) |
| HWM paper port, published smaller planner | 5/104 (4.81%) |

These results establish the highest measured score for our full pipeline on the
fixed 104-case, 13-task MetaWorld-v3 development suite. They do not establish
superiority over native published systems or independently confirm the benefit
of the new reasoner. The simpler internal method is only one case behind.
The development cases have repeatedly informed method design and checkpoint
selection; the 3,200 final cases remain sealed. HWM/CEM are heavily limited by
the 10-second cumulative controller allowance, whereas selected LeFlow and ours
have zero timeouts. Training ceilings match; actual optimizer GPU-hours differ.
Convergence is not established. See [the completed baseline report](RELEASE_BASELINE_RESULTS.md).

## Actual implemented formulation

The selected policy contains **no diffusion or flow-matching head or loss**.
The `flow_jepa` package name is historical. The GMM replacement began with the
controller-grounded iteration; the latest run retained it. This result therefore
does not validate the original JEPA-plus-flow research hypothesis.

Frozen V-JEPA 2.1 features encode the current observation and goal as spatial
tokens, `z = E(o)` and `g = E(o_goal)`. A separately trained, frozen predictor
`F(z, A)` predicts consequences of action chunks. Seven retrieved training-route
targets plus the direct goal supply eight choices `q`.

For each target, four width-256 workspace slots repeatedly attend to current,
target and goal tokens, and up to four previous observed transitions. The history
contains prediction residuals `r_i = z_(i+1) - F(z_i, a_i)`, observed movements,
and progress summaries. Schematically:

```text
h^0       = initialize(z, q, g)
h^(k+1)   = R_theta(h^k; z, q, g, causal execution history)
A         ~ four-component GMM_theta(z, q, g, h^K)
```

The workspace is rebuilt each decision; physical observations remain separate
from internal planning vectors. It is not an indefinitely persistent memory.
The workspace does not itself receive a fresh world rollout at every iteration.
Candidate predictions enter the downstream correction readouts and search.

Each candidate has five control blocks, each containing two primitive actions.
Let `C1` be predicted target distance after the first block and `C5` the average
at blocks four/five. Learned signed errors `e1` and `e5` estimate observed cost
minus predicted cost. The actual selection score is:

```text
J(A,q) = route_cost(q)
       + 0.5 * (C1(A,q) + max(0, e1(A,q)))
       + 0.5 * (C5(A,q) + max(0, e5(A,q)))
       + temporary_observed_stall_penalty(q)
```

Three batches of four candidates per target use 480 world transitions in total.
Shifted-plan reuse preserves a previously selected suffix after its prefix has
been observed. The selected first block executes, and the next real observation
updates history. Positive-only score corrections are conservative engineering;
they are not certified confidence bounds.

## Reasoning training really was included

The run trains with uniformly sampled depths one through four and evaluates
with four iterations. Ordinary backpropagation optimizes action likelihood,
world-predicted matching to recorded prefix/endpoint targets, and factual cost
error calibration. There is no language chain-of-thought, GRPO or reward-based
reasoning training, and no measured test-time depth-scaling result.

Let `E_K` denote smooth-L1 calibration error after refinement and `E_0` the
initial-workspace error for the same recorded action, target and outcome.
Errors are normalized by 0.02. The calibration term is:

```text
0.1 * mean(E_K + 0.25*E_0 + 0.25*max(0, E_K - stop_gradient(E_0)))
```

It supervises both shallow and refined predictions and penalizes refinement
that worsens the supervised calibration loss. Action imitation and this paired
term train useful internal computation indirectly; they do not supervise a
successful recovery decision or prove that four thoughts improve control.
Calibration labels use recorded actions and their recorded outcomes. The
separate generated-action auxiliary uses world predictions, which remain proxy
outcomes. Rejected generated actions have no observed physical labels.

Verified implementation: `flow_jepa/execution/revision.py`, `aligned.py`,
`model.py`, training dispatch, and `config/flow_metaworld_aligned.json`. These
three model/controller files and the configuration have no differences from
the frozen execution source `8be2808eaa09e539d30d07ee3760fc7b59717793`.

## Difference from LeFlow

[LeFlow](https://arxiv.org/html/2608.24855v1) transports Gaussian noise to latent
path interiors conditioned on start/goal, decodes transitions into actions,
then ranks frozen-world rollouts. Its flow objective learns the velocity
`u_data - epsilon` along `(1-s)*epsilon + s*u_data`. Thus JEPA supplies
representations/dynamics and flow supplies candidate paths; it is not pixel
diffusion. LeFlow already includes rollout verification and a consistency term.

Our current method retrieves observed route targets, directly proposes short
actions, and conditions subsequent proposals/scoring on observed execution
errors. Both methods replan from new observations. The distinction is our
explicit residual-conditioned workspace and scoring, not that LeFlow lacks
feedback or verification altogether. The baseline port contains none of our
retrieval, reasoning, corrections or stall heuristics.

## Recent primary literature changes the novelty boundary

The review found several close precedents beyond the earlier LARC/RD-VLA/MPCoT
review. These are source findings, not additional experimental comparisons.

- [Feedback World Model, May 2026](https://arxiv.org/html/2605.15705): an online
  latent observer corrects predictions from execution discrepancies; corrected,
  action-weighted energies guide diffusion. Generic execution-feedback guidance
  is already prior art.
- [FBFM, July 2026](https://arxiv.org/html/2607.29235): measured states and committed
  overlapping actions guide a frozen world-action flow during asynchronous
  execution. Feedback-conditioned flow correction alone is insufficient novelty.
- [Flow-JEPA, September revision](https://arxiv.org/html/2608.29029v3): conditional
  flow predicts action-conditioned future JEPA trajectories. Merely making JEPA
  dynamics stochastic is already explored. This is a separate project from ours.
- [LeWAM by Hegde et al., October 8](https://arxiv.org/html/2610.12407): JEPA
  forward/backward/inverse/policy learning with flow actions and search in policy
  noise space. It motivates keeping search near policy-supported actions;
  its reported benchmark results are not interchangeable with ours.
- [Action-to-Action Flow Matching](https://arxiv.org/html/2602.07322v2): historical
  proprioceptive action embeddings initialize future-action flow. Using a
  previous action sequence as a structured source is also not new by itself.
- [RD-VLA](https://arxiv.org/html/2602.07845v1) and
  [MPCoT](https://arxiv.org/html/2606.06245v1) already study latent refinement and
  computational scaling. See [our earlier code-level analysis](LARC_APPLICATION_20261009.md).

Several distinct projects use the name LeWAM. The Hegde paper above is different
from [Fu et al.'s end-to-end JEPA project](https://le-wam.github.io/) and from
the September paper *Latent evolving World Action Model*. Do not merge their
methods, code or reported numbers. The new papers above were reviewed through
primary arXiv text; no new author-code reproduction was performed. HF lookup
returned 404 for four new IDs, so the primary arXiv fallback was used.

## Recommended capability and research hypothesis

**Learn to recognize which candidate plans have been contradicted by execution,
and redirect the action distribution toward effective alternatives within the
same controller allowance.** This is a candidate scientific question, not an
established distinct algorithm or a claim of first invention.

For a future flow variant, a concrete interface would be an action-chunk flow
conditioned on JEPA state/goal, a supported target, the remaining previous plan,
and causal error workspace:

```text
dA_s/ds = v_theta(A_s, s | z_t, q, g, h_t, previous_plan)
```

This interface is standard conditional flow. The unresolved contribution is the
learning/selection rule that uses contradictory execution evidence to change
probability across plan alternatives, preserves effective alternatives, and
improves actual recovery. Replacing the GMM with this equation is not sufficient.
JEPA would supply physical context and outcome checks, flow would represent
alternative action sequences and their revision, and reasoning would integrate
observed discrepancies. Its benefit must exceed the existing GMM controller at
matched training and inference allowances. None of this variant has trained.

The evidence motivating this direction is specific: proposal diversity falls
about 51% between 5k and 20k while action likelihood improves, and 17/24 selected
checkpoint failures retain optimistic late predictions without actual local
progress. Neither observation proves a cause or guarantees a flow remedy.

Two related capability directions deserve priority over increasing model size:

1. **Object/contact-sensitive progress.** Pick-place is 1/8 and assembly 3/8 at
   the selected checkpoint. Examine whether a small temporal/relational adapter
   can distinguish moving the gripper from actually moving/attaching the object.
   The current static-image features and mean cosine costs may be insufficient,
   but that diagnosis is not yet established. Action-controllability weighting
   alone could still overemphasize the moving arm. Any learned adapter counts
   inside the method budget.
2. **Useful adaptive computation.** Allocate refinement/search to decisions
   where observed error or candidate disagreement suggests it can change the
   outcome; stop when more computation has no demonstrated value. The target
   is success versus controller time and recovery after stalls, not merely lower
   latent cost. Recurrent depth, flow integration steps, search width and physical
   planning horizon are distinct quantities. Long-horizon skill composition
   needs a separate task protocol; current scores do not demonstrate it.

Expert-only histories leave a major supervision gap for recovery. Recorded
random episodes in the original training inventory could provide factual
calibration pairs, but lack demonstrated successful repairs and would change
policy-training exposure. Generated failures or alternative actions cannot be
assigned demonstration outcomes. Any new interaction data must be explicitly
budgeted and must never come from validation/final-test cases.

Preserve the current checkpoints and fixed references. A future registered
candidate should target one of these failures, then assess success and paired
failure cases before adding further mechanisms. No candidate or new run is
registered or launched by this review.
