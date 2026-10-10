# LeFlow with episodic nonlinear test-time memory

October 10, 2026. One authorized seed3072 candidate, `leflow_ttt`. This is a
new direct LeFlow extension, distinct from the completed action-flow `progress_ttt`.
Implementation/preflight complete. Frozen source70c19323351b7fe2756f52ec6968b75f65abe7a2
was dispatched at16:33:27 UTC; coordinator3060649. Optimizer/evaluation evidence
is tracked in PROGRESS.md and the run reports. Dispatch alone is not training proof.

## Thesis and learning mechanism

A visual latent path can be plausible yet decode into actions whose executed
outcomes differ from the frozen world's predictions. Learn a small memory whose
updates on **observed action outcomes** help the next latent-path generation,
inverse decoding and outcome ranking. Training differentiates through those
updates using later recorded actions/outcomes. This is a proposed mechanism,
not a demonstrated novel contribution or state-of-the-art claim.

Keep the released LeFlow flow/inverse modules, horizon5,64 Gaussian candidates,
16 Euler steps and10 primitive actions executed per decision. A small shared
SwiGLU fast-weight memory M(k)=(SiLU(kW0)*(kW2))W1 conditions both the flow's
512D context and the inverse decoder's first hidden layer. Its output also
predicts a bounded correction to the full spatial world prediction for ranking.

Each episode starts from a learned prior. Before each decision, replay the last
four completed10-action chunks; encode each spatial patch's start, predicted
outcome, position and executed actions as a32D key. Values encode the observed
minus predicted outcome. Fit three32/64-dimensional matrices using four inner
MSE updates, a learned bounded step, gradient clipping and importance-weighted
shrinkage to the prior. There are at most128 support rows, not LaCT's large-token
regime. We do not claim LaCT's large-context throughput or reproduce its Muon
optimizer. No cross-episode weights, reward labels or privileged state are used.

Offline query windows exactly match LeFlow's sampler. Support chunks strictly
precede the query. The outer objective retains native flow matching, inverse
MSE and0.1 world consistency, plus0.1 recorded outcome loss and0.05 residual
reconstruction loss. Outcome residuals are scaled by0.1. Training cycles inner
update depths1/2/4; the first formal evaluation fixes4. Predicted outcomes of
newly generated actions are not training ground truth. No generated candidate
receives an expert transition's outcome label.

The baseline simulator clips commands. This candidate records the selected
actually clipped actions and recomputes their expected outcome for its factual
support. This adds5 world transitions:325 versus baseline320 per decision.
All adaptation, correction and encoding time counts toward the10-second budget.
The candidate retains raw-action ranking as in the release before this selected-
action history correction; measure clipping as a possible remaining mismatch.

## Literature and precise attribution

- [LaCT / Test-Time Training Done Right](https://tianyuanzhang.com/projects/ttt-done-right/),
  [paper](https://arxiv.org/abs/2505.23884): nonlinear fast weights and batched
  updates. Official code inspected at a648340f9798f173227a0626fde66a6e9b65879a,
  including minimal causal/bidirectional implementations. We implement an
  independently checked exact gradient of our MSE, without its Muon path.
- [Fast Spatial Memory / Elastic TTT](https://arxiv.org/html/2604.07350): importance-
  weighted protection from destructive adaptation. Code inspected at
  [499464e](https://github.com/Mars-tin/fast-spatial-mem/blob/499464ecd971dc096cc9a27d197aa0b5995f123a/fsm/model/model_lacet.py).
  Our bounded episode-local variant anchors to the learned prior; it does not
  reproduce the paper's streaming EMA anchor or all normalization rules.
- [SCOUT](https://arxiv.org/html/2609.36107): meta-train outcome-based adaptation
  for later expert action quality, rather than optimizing reconstruction alone.
  Official project repository c4b69fa6743df0e44ad51c85b07c289d96025973 exposed
  project content, not released training code. Our data are successful expert
  episodes and lack SCOUT's broader interaction coverage.
- [AdaJEPA](https://arxiv.org/html/2606.32026): plan/execute/observe/adapt;
  official planner code inspected at51d8665. More inner updates need not help.
- [Sandwich Residuals](https://arxiv.org/html/2609.21740): online residual
  adaptation around frozen models is prior art. We cannot claim that alone.

These are selected relevant sources, not proof that this formulation is optimal.
The hypothesis is **meta-learning outcome memory for both LeFlow proposal and
verification**. No separate candidate-deliberation loop, useful reasoning gain,
or inference-scaling curve has yet been demonstrated by this candidate.

## Fixed comparison and budget

Frozen LeFlow:14/21/19/14 out of104, selected10k21/104, zero controller timeouts.
It uses pinned released flow/inverse code f1fe192e41ec6f20de25cd1054ede40a8cbdcfa5,
a paper Appendix-E spatial adapter, and our shared V-JEPA2.1/MetaWorld world.
Its adapter decoder and schedule are explicit port choices. Training losses
were still decreasing; convergence is not established. This is a stable audited
**fixed-budget released-code port**, not a native-benchmark reproduction or
matched-convergence reference. See RELEASE_BASELINE_RESULTS.md.

Reuse its exact frozen adapter after the common2,000-update prefix, charging
its original193.79706083983183 aggregate GPU-seconds. Adapter tensors were
verified identical at5k/10k/15k/20k. Only adapter weights are loaded from best.pt;
flow, inverse and new memory parameters start fresh with seed3072. New candidate
optimization is18,000 updates, yielding20,000 total charged updates. The same
28,800 aggregate optimization GPU-second ceiling includes the common prefix.
This matches caps and total update count, not necessarily consumed GPU time:
the baseline used0.8524 optimization GPUh; meta-training adds work per query.
Report both actual totals and avoid claiming equal consumed compute.
Baseline policy weights and optimizer state are never inherited.

Same6,222 successful expert episodes across13 MetaWorld tasks; unchanged
encoder, world, normalization, packed features and query-window draws. Batch64
on8 nonpreemptively acquired A800s, AdamW1e-4, weight decay1e-4, gradient clip1.
Dataset sampling is with replacement; an epoch count is not the protocol.
There are1,152,000 new planner query windows plus128,000 common adapter samples.
Additional causal support observations come from the same training episodes.

Evaluate the same104 development cases at5k/10k/15k/20k (or registered GPU-cap
fractions if the cap binds), using200 primitive actions and10 cumulative
controller seconds. Select highest success, earliest tie. Primary metric is
success within both limits. Paired failures, task scores, latency/timeouts,
clipping, diversity, fast-weight changes and observed prediction errors are logged.
These104 cases have been repeatedly used for development: final3200 cases remain
sealed. Beating21/104 supports only this benchmark/port/budget claim, not broad
publication superiority. No baseline retraining, new seed, variant, ablation,
automatic inference-scaling sweep or sealed test is included.

## Verification and execution

Seven behavior tests passed, including exact analytic inner gradients checked
against autograd, empty/NaN-padded support, finite meta-gradients, example isolation,
no parameter mutation and equivalence to native modules at zero conditioning.
Training-data-only GPU checks verified256 identical baseline query windows and
causal history. All trainable gradients are finite and synchronized exactly
across8 GPUs at inner depths1/2/4, global64; no optimizer update or simulator
evaluation was used for these checks. Warm single-GPU planner46.56ms excluding
encoding; formal controller timing includes encoding. Preflight records are in
reports/20261010-leflow-ttt/. The journal lock retry is an explicitly recorded
filesystem runtime repair, not an algorithm change.

Execution root: `/home/mtxu/adam/LeFlow-experiments/20261010-leflow-ttt`.
Run: `campaign/runs/leflow_ttt_3072`. Source is a clean frozen Git checkout.
Launcher: `scripts/operations/launch_leflow_ttt.sh`. The coordinator schedules
all four validations and online W&B logging. Never edit the running checkout.

After completion, audit source/data/checkpoint hashes, common-adapter identity,
case pairing and budget; compare all four milestones and selected checkpoints.
Inspect whether lower support error transfers to actual subsequent outcomes,
and whether any gain comes with candidate collapse, action clipping or timeouts.
Do not automatically tune or retrain after seeing the first result.

## First completed evaluation; run still active

At5k:27/104(26.0%), versus selected LeFlow10k21/104(20.2%) and
matched LeFlow5k14/104(13.5%). Against the selected reference,9 paired wins
and3 regressions. Zero controller timeouts; mean decision104.74ms. Optimization
charged through this checkpoint0.449786 GPUh versus0.400696 GPUh for LeFlow's
selected10k checkpoint (and0.188169 GPUh for baseline5k). Thus update-matched
and selected-checkpoint comparisons have different actual training costs.

Faucet-open8/8 and drawer-close5/8 drive the gain. Eight tasks remain0/8.
On executed chunks with prior history, adapted full-outcome MSE is0.013016
versus learned prior0.013129:only0.86% lower. Raw-versus-corrected ranking changes
7.44% of decisions, but that includes the static learned correction and does
not isolate the online update. The completed action-flow/GMM project reference
remains80/104, far above this direct LeFlow extension.

Four saved trajectories were visually inspected. An assembly failure still
expects progress while moving away in latent goal distance; the drawer and door
failures show little useful completion despite positive predicted progress.
The faucet success reaches its task at action126. See the saved
[visual review](reports/20261010-leflow-ttt/round1-visual-review.json).
No causal adaptation gain, monotonic inference scaling, native LeFlow benchmark
improvement or broad state-of-the-art claim follows from this first round.
The10k/15k/20k evaluations continue automatically on the frozen run.
