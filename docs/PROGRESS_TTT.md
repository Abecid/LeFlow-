# Execution-grounded progress TTT: literature, formulation and implementation

October 10, 2026. **Implemented; training-data-only verification completed. No new
full training run or environment evaluation has started.** The prior candidate's
completed authorization is preserved. This request asks for literature research,
formulation and implementation; the new config therefore leaves training and
sealed-test execution disabled. There is one candidate, no sweep or ablation queue.

The best-supported next hypothesis for our measured failures is to **meta-train a
small, goal-relative execution-error memory that changes both action-flow proposals
and their ranking**. “Best-supported” is a design judgment, not evidence that it
outperforms the current 80/104 reference. Online world-model adaptation itself is
already established; novelty and task benefit remain unproven.

## Evidence driving the choice

Our [completed flow run](FLOW_REASONING_RESULTS.md) scores 79/104 at 5k and 76/104
at 20k, below the preserved GMM controller's 80/104. Its corrected-prefix error
improves while task success declines. Proposal dispersion drops 58.90% on matched
initial cases; this is not proof of loss of useful physical alternatives. Ten
5k episodes hit the controller time limit. Assembly and pick-place remain weak.
These observations argue for decision-relevant adaptation and bounded overhead;
they do not prove that fast weights will solve the problem.

The current framework uses a frozen V-JEPA 2.1 visual encoder, a separately trained
frozen action-conditioned world model, retrieval from recorded training routes,
and a reasoner conditioning rectified-flow action generation. It is a custom
MetaWorld image-goal system, not an unchanged published LeFlow model.

## Primary literature and released-code audit

Sources were checked on October 10. The following are methodological inputs, not
additional experiments run in this project.

| Source | Relevant finding and decision |
| --- | --- |
| [LaCT / Test-Time Training Done Right](https://arxiv.org/abs/2505.23884), [project](https://tianyuanzhang.com/projects/ttt-done-right/) | Fast-weight memory can be trained end to end and updated in large GPU-efficient chunks. Causal application must respect what was observed. Our four transitions times 32 patches are 128 observations, far below its large-chunk regime. We adopt batched, causal writes, not its full architecture or speedup claim. |
| [Fast Spatial Memory / Elastic TTT](https://arxiv.org/abs/2604.07350), [project](https://fast-spatial-memory.github.io/) | A direct LaCT follow-up addresses drift/forgetting with importance-weighted consolidation and evolving anchors. This supports constraining adaptation. Our fixed isotropic ridge prior is simpler; it is not their Fisher-style elastic update. |
| [REFINE](https://arxiv.org/abs/2602.16704) | A direct fast-weight follow-up uses next-sequence objectives and reinforcement learning. Downstream usefulness matters beyond fitting the current context. We supervise future recorded actions/outcomes; we do not add its GRPO stage or reward budget. Paper reviewed, release implementation not audited. |
| [AdaJEPA](https://arxiv.org/abs/2606.32026), [project](https://agenticlearning.ai/adajepa/) | Updates a JEPA model within MPC from executed transitions and resets between episodes. This directly rules out claiming episodic JEPA adaptation as new. We preserve the cached encoder/world and adapt progress-error weights instead. |
| [Sandwich-Residuals](https://arxiv.org/abs/2609.21740) | A recent AdaJEPA-related method adapts small residuals around a frozen world model, with episode resets. Parameter-efficient TTT alone is also prior art. Our correction lives in goal-relative progress space, not predictor input/output latent space. Paper reviewed, release implementation not audited. |
| [SCOUT](https://arxiv.org/abs/2609.36107), [project](https://liy1shu.github.io/SCOUT/) | Shares an outcome-adapted latent belief between policy and dynamics and meta-trains adaptation for future expert actions. This motivates training through the inner fit and conditioning the generator. Its histories include random/incorrect actions and privileged supervision unavailable in our success-only policy data. We do not silently import those assumptions. |
| [JEPA-TTT](https://arxiv.org/abs/2610.00722) | Adapts a latent predictor with fixed encoder/reward head and persistent replay across deployment episodes. Its 500-episode setting is different from our independent-case benchmark. We reset all adaptive state each episode and use no cross-case replay. |
| [World-Coherent Decoding](https://arxiv.org/abs/2609.02159) | Uses observed imagination/execution mismatch to learn candidate reliability. Online reliability-based ranking is already prior art. Its persistence and video/action-surprisal machinery differ from this small episodic progress fit. |
| [Beyond Visual Quality](https://arxiv.org/abs/2609.24745) | Its controlled candidate analysis finds useful oracle opportunities that proxy selectors often fail to recover. Action spread is not sufficient evidence of useful candidate diversity. We retain success, latency and observed-progress diagnostics as the meaningful tests. |
| [R2D2 / differentiable closed-form solvers](https://arxiv.org/abs/1805.08136) | Establishes differentiating through regularized closed-form adaptation. This is the solver foundation, not a claimed invention. We use a batched Cholesky solve of a small positive-definite system rather than an explicit inverse. |

Reviewed release code, pinned for reproducibility:

- [LaCT a648340](https://github.com/a1600012888/LaCT/tree/a648340f9798f173227a0626fde66a6e9b65879a):
  `minimal_implementations/causal_lact_with_sliding_window_attn.py` and
  `bidirectional_lact_layer.py`. The minimal bidirectional Muon branch assigns
  normalized updates to weight variables, while the causal branch normalizes
  update variables before adding them. We did not copy that branch or transplant
  Muon; this discrepancy was not experimentally characterized.
- [FSM 499464e](https://github.com/Mars-tin/fast-spatial-mem/tree/499464ecd971dc096cc9a27d197aa0b5995f123a):
  `fsm/model/model_lact.py` and `model_lacet.py`; inspected detached importance
  statistics, prior shrinkage, channel renormalization and optional re-anchoring.
- [AdaJEPA 51d8665](https://github.com/agentic-learning-ai-lab/adajepa/tree/51d8665b7978824bd218decab9e05ddb6eb1f47b):
  `planning/adajepa.py` and `adajepa_mpc.py`; checked parameter snapshots/reset,
  observed-segment adaptation, stop-gradient targets and the per-sample loop.
- [R2D2 fc0c13e](https://github.com/bertinetto/r2d2/tree/fc0c13ec991bb9f84395cb12de57cb150ce76f8d):
  `fewshots/models/r2d2.py`; checked support/query separation and regularized
  standard/Woodbury solves. Our code is independently implemented in current PyTorch.
- [SCOUT c4b69fa](https://github.com/liy1shu/SCOUT/tree/c4b69fa6743df0e44ad51c85b07c289d96025973)
  contains the project page and media; its page says code is coming soon. No
  training implementation was available in that inspected tree.

## One coherent mechanism

The research question is: **Can a fast memory learned for future action selection
correct the progress errors that matter to a compute-limited JEPA/flow controller?**
The fast memory is the central mechanism. Retrieval provides supported targets;
the frozen world supplies imagined consequences; flow matching proposes action
chunks; candidate-conditioned reasoning revises subsequent proposals.

Let `z_i` be the encoded state before an executed two-action prefix `a_i`,
`z_i+` its subsequently observed outcome and `hat_z_i+ = f(z_i,a_i)` the frozen
world prediction. For a legitimate current target `q` and patch `p`, define

```
y_i,p(q) = d_p(z_i+, q) - d_p(hat_z_i+, q),
d_p(u,q) = 1 - cosine(u_p,q_p).
```

Positive error means the world was too optimistic about reaching that target.
This is a factual label for an executed action. A target may be relabelled from
supported training states; an unexecuted action may not inherit this outcome.

A learned key `phi_i,p(q)` sees only the pre-outcome state, executed action,
frozen prediction, target and patch position. Its 32 coordinates include a bias
coordinate and are normalized to unit norm. With `s=0.02`, the fast weights solve

```
w(q) = argmin_w sum_{i in last4, p} (phi_i,p(q)^T w - y_i,p(q)/s)^2 / P
               + 0.125 * ||w - w0||^2 .
```

`w0` is the offline-learned prior. The solution uses a batched 32-by32 Cholesky
solve in float32, differentiated during offline training. No support returns
exactly `w0`. At most 128 patch rows per target are available; these correlated
patches are not 128 independent physical transitions. Regularization is a fixed
predeclared choice, not a tuned optimum or a Fisher approximation.

The readout `0.1*tanh(s*phi^T w/0.1)` bounds a patch correction to ±0.1. The inner
solve fits the unbounded linear surrogate; the deployed correction is the bounded
readout. The average prefix correction enters the existing conservative score:

```
score = 0.5 * raw_prefix_cost + 0.5 * raw_terminal_cost
        + 0.5 * max(mean_patch_correction, 0)
        + 0.5 * max(offline_terminal_correction, 0).
```

Negative estimated error does not make a counterfactual candidate cheaper than
the raw world predicts. Existing route costs and causal stall penalties remain.
The fast delta `w-w0` also projects into the workspace before flow generation.
Thus newly observed evidence can change both proposed actions and their ranking.
The reasoner then inspects each round's imagined action/cost pairs to revise its
next proposals, as in the preceding flow implementation.

**One fit per decision, fixed through all three search rounds.** An unchanged
candidate cannot improve its score merely because the reasoner changes. Adapted
weights are ephemeral context tensors; the offline checkpoint, encoder and world
are never modified. History enters only after the selected prefix is executed
and observed. New episodes clear action/outcome history and all transient weights.

## Offline objective and data integrity

Each example contains a current state, its past four recorded transitions, a
five-block expert action chunk, its recorded prefix/terminal outcomes, and supported
goals. The inner fit sees only the past. Current teacher actions and future outcome
labels never enter candidate generation or the support fit. The outer loss is

```
L = mean_depth(flow_matching_loss)
    + 0.25 * paired_depth_regression_hinge
    + 0.10 * 0.5 * (prefix_patch_smoothL1 + terminal_smoothL1).
```

The two calibration losses use scale 0.02. Prefix supervision averages patches so
its weight does not grow 32-fold. Gradients pass through the fit into key features,
prior and the flow-conditioning projection. The shared frozen world provides
only detached imagined evidence. No generated action receives a demonstration's
physical outcome label. This trains adaptation for the next recorded decision,
not merely for reconstructing the support observations.

The existing one/two/three-round depth cycle, full Gaussian flow source, eight
Euler steps, final-only action clipping, candidate count and incumbent retention
are unchanged. No entropy bonus, extra rollout, full-world tuning or auxiliary
RL stage is bundled into this candidate.

The config preserves seed 3072, global batch 64, maximum 20k updates / 28,800 aggregate
optimization GPU-seconds, the same 6,222 successful expert episodes, frozen world,
and four fixed 104-case validations. The complete world-training split remains
7,800 episodes. Sampling remains 1.28M windows with replacement at 20k, not a new
epoch schedule. Adaptation, encoding, retrieval and search all count within the
same 10-second cumulative controller allowance and 200 primitive actions.
No cross-evaluation-case learning is permitted. Sealed tests stay unopened.

## Verification and status

See [verification evidence](reports/20261010-progress-ttt/). The implementation has
5,577,448 parameters versus 5,445,162 in the preceding flow candidate (+132,286).
Eight new behavior tests cover analytic optimality/differentiation, masked padding,
example isolation, exact empty-history prior, no checkpoint mutation, no teacher
leakage, suffix-independent prefix correction, fixed scoring during revision,
flow influence, action-ranking influence, causal action/outcome alignment,
episode reset, and one batched fit / 480 world transitions per decision.

The A800 train-data-only preflight uses a fresh untrained model, eight real training
examples and the unchanged frozen world and bank tensors. It checks 10,000 sample
draws, including late windows, and performs forward/backward plus controller timing.
Five measured full-controller calls are 93.307, 91.813, 91.223, 94.690, 92.735 ms:
**92.754 ms mean**, below the existing 120 ms readiness gate. Peak allocation is 9.493 GiB.
This is a short preflight, not a paired rollout-level speedup or timeout guarantee.
The fitted untrained prefix MAE was 0.002199 versus raw 0.001827 on this tiny batch;
this is not performance evidence for the trained method.

The eight-GPU backward check passed at global batch 64 for all three reasoning
depths. Gradients were finite and exactly synchronized; model weights stayed
unchanged and the world received no gradients. The 21 inherited regression tests
and eight final TTT tests passed (29 distinct behavior tests). These checks made
zero optimizer updates and opened zero simulator episodes. The final audit verified
all 72 runtime-file hashes and exact identity of every preserved bank tensor and
training inventory. All eight GPUs were idle afterward. The timed GPU check
regions total 0.010649 GPU-hours; process startup and CPU preparation are excluded.
There is no new task-success score, active training run or test-time scaling curve.

## What would constitute progress, and what may still fail

The decisive result is environment task success under the fixed allowance against
our strongest 80/104 reference and the saved corrected baselines. Fitting recent
errors, lowering flow loss or increasing action spread cannot substitute for it.
Recorded diagnostics expose the prior and adapted prefix correction, fast-weight
change, support count, actual executed progress, selected round, proposal spread,
timeouts and per-task outcomes.

Main risks are transfer from successful demonstrations to stalled/contact states;
weak goal features for object attachment and contact; locally accurate but
counterfactually wrong corrections; useful alternatives missing from the proposal
pool; and latency consuming the remaining action allowance. This implementation
does not establish a causal remedy for the observed diversity decline.

TTT here means fitting fast weights from newly observed execution. Reasoning means
revising proposals from imagined consequences. Test-time scaling means measured
improvement as inference resources increase. The first two are implemented;
useful scaling remains unmeasured. More iterations can amplify model errors and
exhaust the clock, so no automatic scaling sweep is queued.

A publication would need a meaningful measured gain, mechanism evidence, a locked
held-out evaluation and comparison with relevant adaptive methods. AdaJEPA and
Sandwich-Residuals now matter in addition to LeFlow/HWM/CEM; SCOUT matters with its
data assumptions made explicit. Those are prospective comparisons, not newly
queued baseline runs. The potential contribution is the jointly trained,
goal-relative progress adaptation feeding flow generation and decision scoring,
not the invention of fast weights, ridge regression, JEPA adaptation or generic
execution-aware reasoning.
