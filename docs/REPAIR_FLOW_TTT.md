# Execution-conditioned flow repair: evidence and registered experiment

October11,2026. Registered fresh candidate `repair_flow_ttt`, pending verified
launch. No result is claimed. The strongest measured reference remains the
execution-revision GMM controller,80/104. The question for this iteration is
whether execution-conditioned local flow repair can make additional planning
computation useful within the same physical controller allowance.

## What failed

| Preserved method | Four development scores /104 | Selected | Selected timeouts | Mean decision ms |
| --- | --- | ---: | ---: | ---: |
| Execution revision, GMM + CEM refinement |80/78/79/76|80|0|86.74|
| Reasoning-conditioned action flow |79/75/76/76|79|10|99.42|
| Flow + episodic adaptation |77/76/78/74|78|21|102.50|
| Guided flow + adaptation |72/70/72/75|75|29|103.46|

These repeatedly reused104 development cases do not establish independent
robustness. The GMM controller has the best selected score and highest minimum
checkpoint score among these four. Its search explicitly refines elites. The
flow variants instead regenerate proposals from global Gaussian noise after a
workspace update. Merely adding a larger/recurrent generator did not preserve
that effective local search behavior.

The latest guided model loses7 cases and wins2 against the80/104 checkpoint.
Losses: assembly6/7, door-open4, faucet-open0, pick-place4, plate-slide1 and reach1.
Selected task scores include assembly1/8, faucet-open2/8, pick-place0/8 and reach3/8.
Saved videos show unsuccessful object interaction/contact: e.g. the pick-place
hand travels while the object remains on the table. Image similarity and predicted
progress can improve without manipulating the object. All29 failures exhaust
controller time after182–198 primitive actions; this does not prove extra time
alone would solve them.20/29 have raw optimistic stalls in their last20 decisions;
8 remain after correction. See [full outcome analysis](GUIDED_TTT_RESULTS.md).

The adapted prefix predictor improves MAE7.92% over its learned prior, yet the
guided critic's observed-outcome MAE(.002479) is worse than the corrected world
score(.001632). Scoring accurately on recorded actions does not certify useful
gradients at generated actions.

A no-update diagnostic on the first32 fixed TRAIN windows and128 generated
proposals found57.8% surrogate/raw-world pairwise ordering agreement (192 pairs),
mean gradient cosine.204 and30.5% opposite gradients between guided energy and the
full world with the SAME correction. The world is itself imperfect: these are
model-to-model checks, not physical derivatives or success measurements. Small
sample; do not generalize the percentages. Timing the actual8-target batch gave
9.89ms guided proposal vs3.86ms unguided generation, before world verification.
Two diagnostic invocations cost about.000945 GPUh in total, with zero updates,
simulator calls or final-test access. [Raw checks](reports/20261011-repair-flow-ttt/diagnostics-corrected.json)
and the exact diagnostic script are preserved.

## Bugs, mismatches and limits

- Confirmed logging alias: every50th update recorded only depth2 of the alternating
  depth1/2 schedule. New logs use offsets0 and1; analyze depth groups separately.
- Guided candidate slot1 was subsequently overwritten by a previous-plan prefix
  in round1. Some guidance computation was discarded. No such gradient loop in
  this candidate.
- Old short preflight underpredicted full-evaluation latency. The new probe includes
  observation bookkeeping, warm starts and populated memory, with12 measured
  decisions after2 warmups and a95ms launch gate. It still uses training-image
  timing probes; physical rollout latency is measured by the registered evaluation.
- No evidence of an ODE sign, full-Gaussian source or action-dimension defect in
  the previous guided sampler. QGF's clean-action estimate and detached identity-
  Jacobian approximation were implemented. The main mismatch is the critic:
  a jointly learned expert-transition surrogate is not QGF's offline-RL critic.
- Training demonstrations do not label outcomes of generated actions or contact
  failures. We must not attach demonstration outcomes to generated actions, train
  on validation failures, or claim that lower teacher action loss proves execution.
- The new training proposal pool still lacks deployment's retrieved/warm-start
  slots. Its local teacher target may saturate when the proposal is far from the
  demonstration. Both are explicit remaining risks, measured where possible.

## Formulation and implementation

Let C be the causal workspace containing observation, target, final goal and
recent executed transitions. Fit the existing regularized episodic error memory
using only observed next-state minus predicted prefix cost. The fast fit resets
per episode; the offline action/error losses differentiate through the fit.
Generate four global action chunks per target using an8-step conditional flow.
Verify them with the frozen action-conditioned world and fixed corrected scoring
context. Retain the lowest-cost incumbent a*. Revise the workspace from detached
candidate/action/predicted-outcome tokens.

A separate conditional flow now produces a bounded local residual:

    r* = clip((a_recorded - stop_gradient(a*)) / rho, -1, 1)
    r_t = (1-t) epsilon + t r*                  epsilon ~ N(0,I)
    L_repair = ||v_repair(r_t,t,C_revised,a*) - (r* - epsilon)||²
    a_repaired = clip(a* + rho clip(r_generated,-1,1), -1,1)

rho=.25,4 Euler steps,3 repair candidates plus one EXACT incumbent. Full-world
verification selects between the initial and repaired pools using the same scoring
context. A deterministic first-minimum tie rule aligns the retained incumbent
with the repair anchor. No physical outcome is assigned to a repaired action.
The factual calibration branch sees only recorded actions and outcomes.

Total objective: global flow loss every update, repair loss at the alternating
deep updates, plus.1 factual calibration loss. The global and repair heads and
all new trainable modules initialize fresh. The old surrogate/gradient guidance
is absent. Global solver8 + repair solver4 =12 velocity evaluations per decision;
2 rounds ×8 targets ×4 candidates ×5 transitions =320 world transitions, versus
480 for the GMM reference. This is a new formulation, not a lossless optimization
or an isolated ablation. Fewer predictions do not by themselves imply better plans.

Five new behavioral tests passed: detached best-anchor/permutation consistency,
full Gaussian source and exact incumbent/radius bounds, absence of future-teacher
leakage, gradients through both trained depths with frozen world, and controller
accounting/observed-only memory/reset. The launch additionally runs inherited tests,
real-data preflight and a no-update4-rank gradient synchronization check.

## Literature and research judgment

[QGF and official sampling code](https://github.com/zhouzypaul/qgf/blob/60ca92e70aa59c23cd016e5fd27f1c99ed00984f/agents/qgf.py)
justify evaluating clean-action critic gradients, but do not validate our critic.
[PreferenceFlow](https://arxiv.org/html/2609.36872) highlights harmful gradients and
regularization; its human-intervention labels are absent here. Increasing guidance
strength without reliable gradients is not the next justified step.

[Fisher Decorator](https://arxiv.org/html/2604.17919) motivates local residual
transport with constrained policy refinement. Our action box is not its Fisher
constraint, KL guarantee or RL algorithm. Residual flow refinement itself is
therefore not a novelty claim.
[TraceFlow](https://arxiv.org/html/2609.20646) already uses retrieved traces and
bounded flow guidance, and [SCOUT](https://arxiv.org/abs/2609.36107) already learns
belief adaptation from execution errors. Generic feedback + retrieval + flow,
or adding TTT, is insufficient novelty.
[LaCT](https://tianyuanzhang.com/projects/ttt-done-right/) concerns fast-weight
learning and large-chunk efficiency; our small episodic fit is an adaptation,
not its demonstrated scaling result.

Best current engineering choice: execution-revision, because it wins this measured
comparison and meets latency. Preferred research direction: **learning how observed
execution errors should change the next locally refined generative action**, with
world verification and retained incumbents. It gives a concrete mechanism to study
rather than relying on a workspace being implicitly useful. A publication requires
showing that this mechanism improves physical success and that additional inference
helps at controlled cost. Neither distinct novelty nor test-time scaling is yet
established. Prioritize the mechanism and evidence over adding more modules.

## Registered comparison and continuation

Same13 training tasks,6222 successful expert episodes, V-JEPA2.1 ViT-L static
32-token cache, frozen selected action-conditioned world and train-only route bank.
Seed3072,global64,20k updates OR28800 aggregate optimization GPU-seconds, validations
at5k/10k/15k/20k (or final trained step at a binding cap).10 cumulative controller
seconds/200 primitive actions; fixed104 development cases,3200 final tests sealed.
Training windows are resampled;20k×64=1.28M window presentations is not20k epochs.
AdamW1e-4,weight decay.05,1000-update warmup/cosine decay and clipping1 unchanged.

Use4 idle A800 GPUs while others are occupied. Aggregate ceilings/data/cases are
matched; rank RNG assignment and floating-point reductions differ from8-GPU runs,
so no bitwise-equivalence claim. Actual consumed compute must be reported, including
separate preflight/diagnostic/evaluation costs. Frozen baselines are never retrained.

Record: `/home/mtxu/adam/LeFlow-experiments/20261011-repair-flow-ttt`.
Before launch freeze source and runtime hashes; after launch append actual identity,
PIDs and W&B. On completion preserve every periodic metric and104-case report,
audit hashes/cases/budgets, analyze paired wins/losses, timeout/stall/contact behavior,
incumbent-selection fraction and teacher-target clipping. Publish to origin/main.
Only then decide the next specific correction under the user's iterative request.
No blind queue, extra seeds, hyperparameter sweep, baseline rerun or sealed test.
