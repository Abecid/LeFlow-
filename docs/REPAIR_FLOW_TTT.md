# Execution-conditioned flow repair: evidence and registered experiment

October11,2026. Active candidate `repair_flow_ttt` scored **80/104 at5k and
73/104 at10k**, both without timeouts. The selected5k checkpoint ties the preserved
execution-revision GMM controller. Training has reached15k; two registered
evaluations remain. This is an interim result, not a completed run. The question
is whether execution-conditioned local flow repair makes additional planning
computation useful within the same physical controller allowance.

## Second registered evaluation:73/104 at10k

Readback October11 13:30 Pacific: source481bb767 and all150 runtime hashes remain
unchanged; coordinator739421 and launcher741958 retain their recorded process
identities. Optimization reached15k under the original cap; the15k evaluation is
in progress. W&B records73/104 at10k. This is a scientific regression with healthy
execution, not an operational failure requiring a restart.

Relative to5k,10k loses seven successes and gains none: assembly5, door-open1,
faucet-open1/4, pick-place3/4 and reach7. Task scores become assembly0/8,
faucet-open0/8, pick-place0/8, reach3/8, dial-turn7/8 and door-open7/8; the remaining
seven tasks stay8/8. All31 failures reach200 primitive actions, zero timeouts;
mean decision latency86.27ms. Five-k remains selected at80/104, with its paired
reference comparisons unchanged. Every case/reset/model-seed/episode identity is preserved.

| Saved measurement |5k|10k|
| --- | ---: | ---: |
| Development successes /104 |80|73|
| Repairs selected, % decisions |55.85|49.92|
| Corrected executed-prefix MAE |.001688|.001680|
| Optimistic failure tails, raw / corrected |15 /5|22 /9|
| Deep-update global flow loss |.1537|.1033|
| Deep-update repair flow loss |.4393|.3341|
| Deep-update anchor action MSE |.02870|.01563|
| Deep-update teacher residual clipping, % |7.79|3.81|

Training numbers average the last500 updates before each checkpoint, grouped by
depth. Lower action losses, less clipping and slightly lower rollout calibration
error accompany worse success; they are not reliable success certificates here.
Teacher-interpolated action loss still contains target information. Rollout MAEs
come from different executed trajectories and are not a controlled calibration test.

At the identical first reset/goal/routes with empty history, global-pool proposal
dispersion falls40.34% and repair-pool dispersion36.71%; both decrease in all104
cases. Across all decisions, the mean selected action displacement from the
incumbent falls from.1107 to.08291. These saved associations motivate checking
proposal coverage in the final analysis, but do not prove that diversity loss
caused the failures. The dispersion metric mixes generated/recorded proposals
or incumbent/residual proposals and is not policy entropy. No additional model,
simulator or inference sweep was run to obtain these findings.

[Two-round analysis](reports/20261011-repair-flow-ttt/analysis-round-2.json),
[full10k evaluation](reports/20261011-repair-flow-ttt/run/validation/step_0010000.json.gz)
and [interim identity verification](reports/20261011-repair-flow-ttt/verification-round-2.json)
preserve all outcomes and checkpoint hashes. Periodic metrics are now saved through
15k. Cumulative compute at15k is1.209956 optimization GPUh plus.923527GPUh for the
two completed evaluations, with zero optimization overrun. The in-progress15k
validation is not yet included in that completed-evaluation charge; no final total
or completion audit is claimed. Preflight and diagnostic charges remain separate.
Keep the original run undisturbed through its registered evaluations; defer any
next correction until completed analysis and verified publication.

## First registered evaluation:80/104 at5k

Readback October11 13:14 Pacific: training healthy atstep8668, with source and
process identities unchanged. The5k report contains all104 fixed cases:80 successes,
zero timeouts,86.46ms mean decision time (89.88ms p95). All24 failures use the full
200-action allowance. This ties the GMM reference and improves over the selected
flow79, progress78 and guided75 references; it does not exceed the stopping-for-review
threshold. The existing run continues to its remaining registered evaluations.

| Preserved reference | Successes | Repair-flow paired wins | Paired regressions |
| --- | ---: | ---: | ---: |
| Execution revision |80|6|6|
| Flow reasoning |79|4|3|
| Progress TTT |78|7|5|
| Guided TTT |75|7|2|

Case IDs, reset/model seeds, episode hashes and tasks match every reference.
The six regressions against GMM are assembly6/7, dial-turn2, faucet-open0/7 and
reach1; the six gains are dial-turn6, door-open0/1, faucet-open1, pick-place3 and
reach7. Task success remains weak for assembly1/8, faucet-open2/8, pick-place2/8
and reach4/8; dial-turn7/8, and the other eight tasks are8/8.

Local repairs are selected on2889/5173 decisions (55.85%); the exact incumbent
is retained on44.15%. All saved decisions preserve the incumbent, keep action
corrections within.25, and use12 flow evaluations/320 world transitions with fixed
scoring context. Selected repairs and incumbents have similar mean observed latent
prefix progress (.002913 and.002908), but these selected groups cannot establish a
causal repair benefit. Full-world ranking guarantees a non-worsening predicted
score, not physical progress.

Corrected executed-prefix MAE is.001688 versus.002637 for the prior (35.97% lower).
Fifteen of24 failures have optimistic raw predicted progress with nonpositive
observed latent progress over their final20 decisions; five remain after correction.
These are latent proxies. No new contact-video review or physical-cause conclusion
is claimed. In the deep-update logs immediately before5k, repair-target coordinates
clip at7.79%, repair-flow loss averages.4393, and anchor action MSE.02870. Analyze
depths separately: shallow updates have no repair loss. Teacher-interpolated action
MSE includes target information and does not certify generated-action quality.
One checkpoint cannot establish that calibration or teacher loss predicts success.

[Saved-record analysis](reports/20261011-repair-flow-ttt/analysis-round-1.json),
[complete first evaluation](reports/20261011-repair-flow-ttt/run/validation/step_0005000.json.gz)
and [interim verification](reports/20261011-repair-flow-ttt/verification-round-1.json)
preserve negative outcomes and checkpoint/source/data identities. At the first
publication, the metric snapshot coveredstep9251 and the compute snapshotstep9296,
optimization.747117GPUh plus first validation.470445GPUh. These are cumulative
within this run and are not final totals. Preflight.005795GPUh, distributed check
.002990GPUh and earlier diagnostic costs remain separate; no allowance was reset.
W&B reports the first80/104 while training remains active. Source481bb767 and all
150 runtime hashes are checked without modifying the execution checkout.

The104 cases are repeatedly reused development data, with one training seed and
checkpoint selection. Four versus eight training ranks changes RNG assignment.
This new formulation and latency change do not isolate a component's effect or
establish independent generalization, novelty, useful inference scaling or SOTA.
No further candidate is selected or launched before this run completes and is
fully analyzed and published. Final tests remain sealed.

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

## Launch record

October11 12:54 Pacific: deployed immutable source
`481bb7679aa3f43e938fd3991f3be57ba59f0ed0`;150 runtime file hashes verified.
Coordinator PID739421 entered pre-training checks. This is not yet confirmation
of optimizer updates. Existing15-minute follow-up now handles the user-authorized
sequential failure-analysis/implementation/training loop and publication, with
unchanged per-run constraints. The first numerical result remains pending.

Verified at12:56:56 Pacific: optimizer step353, worker741958, W&B
[r3y7nepq](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/r3y7nepq),
no failure. All34 CPU tests passed;4-rank no-update gradient synchronization exact,
all parameters unchanged by checks, world gradients absent. Preflight mean79.77ms
(12 measured probes),9.50GiB peak including frozen encoder. Preflight.005795GPUh
and distributed check.002990GPUh are recorded separately. Healthy training logs
include both depths. No development evaluation has completed at this snapshot.

Read-only follow-up helpers added outside the frozen execution source:
`scripts/analysis/repair_flow_ttt_details.py` analyzes all saved evaluations and
paired reference outcomes; `scripts/analysis/audit_repair_flow_ttt_run.py` verifies
source, data, route-bank tensors, checkpoints, both logged depths, causal histories,
incumbent/radius bounds,12 flow/320 world accounting and compute. Run them on saved
records after completion; their final checks are not yet claimed to have passed.

Interpretation guard: `repair_teacher_action_improvement` evaluates a clean
estimate from a teacher-interpolated point that already contains part of the
action target. It is positive even in preflight before any update; it is not a
generation-quality gain. Analyze it with that baseline limitation, the residual
clipping fraction, selected-repair frequency and actual registered task success.
Comparisons between selected incumbent/repair decisions are observational, not
randomized evidence of the repair's causal effect.
