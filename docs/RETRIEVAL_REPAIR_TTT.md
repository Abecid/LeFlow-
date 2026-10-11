# Completed: 75/77/79/80, selected 20k80/104

[Final findings and preserved evidence](RETRIEVAL_REPAIR_TTT_RESULTS.md) supersede
all interim status below. The run is complete, tying the preserved80/104; all416
episodes have zero timeouts. Source/data/checkpoint audit passed, W&B6wpx3jcv
finished, and owned processes exited. No new run is active at this completion
snapshot. Preserve source and baselines; publication precedes a next decision.

# Second evaluation: 77/104; run remains active

At the 21:42 UTC collection, the registered development curve is **75 / 77**
at 5k / 10k. The 10k checkpoint gains 8 cases and loses 6 versus 5k;
it is the provisional selection, still below the preserved 80/104. Both rounds
have zero controller timeouts. All 27 second-round failures reach the 200-action
ceiling. Training has reached 15,000 updates and the 15k evaluation is in
progress; the 15k and 20k results and final audit remain pending at this snapshot.
The run continues unchanged.

| Update | Successes | Timeouts | Repair selected | Corrected prefix MAE |
| --- | ---: | ---: | ---: | ---: |
| 5,000 | 75/104 | 0 | 53.37% | 0.0015608 |
| 10,000 | 77/104 | 0 | 44.21% | 0.0016705 |

Corrected executed-prefix error increases about 7% from 5k while success rises
by two cases. Across these checkpoints, lower calibration/action losses do not
consistently track task success. This association does not isolate a cause.

| Frozen reference | Successes | Current 10k wins | Regressions |
| --- | ---: | ---: | ---: |
| repair_flow_ttt | 80 | 2 | 5 |
| execution_revision | 80 | 5 | 8 |
| flow_reasoning | 79 | 4 | 6 |
| progress_ttt | 78 | 5 | 6 |
| guided_ttt | 75 | 6 | 4 |

[Exact paired case IDs and diagnostics](reports/20261011-retrieval-repair-ttt/analysis-round-2.json)
and [5k-to-10k changes](reports/20261011-retrieval-repair-ttt/checkpoint-changes-round-2.json)
are preserved. The same case IDs, episode hashes, reset seeds, model seed and
controller accounting passed checks against every frozen reference. Comparisons
remain selected results on repeatedly used development cases, with one seed and
baseline port limitations; no component causality or independent generalization
is established.

The 10k repair-anchor counts are generated 1,495, shifted 2,313, retrieved 1,394. These are
incumbent sources before residual repair. Shifted plans remain present during
deployment but absent from this training correction; selected-source counts
do not prove a cause of failure. Over each last-500-update window, deep retrieved
anchor fractions change 30.94% → 30.16%; teacher clipping
6.87% → 4.06%, repair-flow loss
0.427065 → 0.291928. Teacher targets include recorded
actions and do not measure generated action quality. The saved traces contain
19/27 raw-score and 10/27 corrected-score optimistic, nonpositive-observed
latent-progress failure tails. These are proxy diagnostics, not physical contact
or stall-cause measurements. [Anchor groups](reports/20261011-retrieval-repair-ttt/anchor-outcomes-round-2.json).

Source remains clean at 3bdfb49 with all 137 runtime hashes verified. Coordinator
819165, launcher 821683 and their four training ranks match the recorded start
identities; W&B 6wpx3jcv is running and reports the same 10k score. All available
periodic metrics and both full evaluations are preserved losslessly. Checkpoint
paths/hashes and online evidence are in the [health snapshot](reports/20261011-retrieval-repair-ttt/interim-health-round-2.json).
The compute snapshot records 1.312060 optimization GPUh and
0.934417 registered evaluation GPUh, plus the separately charged startup checks
and retrieval setup. [Known cumulative measured components](reports/20261011-retrieval-repair-ttt/compute-accounting-round-2.json)
are 93.193763 GPUh; this partial snapshot is not a final run cost or full
cluster bill. No extra candidate, baseline work, evaluation repeat or sealed test.

# First evaluation:75/104 at5k; training continues

Verified October11 at21:28UTC: the first fixed104 development round scores
**75/104**, below the preserved repair-flow and execution-revision references
at80/104. Against repair-flow there are **three paired wins and eight regressions**;
against execution-revision, four wins and nine regressions. All29 failures reach
200 primitive actions, with **zero controller timeouts**. Mean decision latency
is85.65ms. The remaining10k/15k/20k rounds are pending at this snapshot; this is
an interim result, not completed training or evidence to change the active run.

Repairs supply2,963/5,552 decisions (53.37%). Repair-anchor provenance is1,586
generated,1,482 retrieved and2,484 shifted (44.74%). Among failure decisions,
1,476/2,900 anchors are shifted (50.90%); episode lengths and visited states
confound that proportion. This confirms deployment uses shifted anchors, but
does not establish that missing shifted-plan training caused failures.
Retrieved actions are selected as training anchors in30.94% of deep-update
examples in the final500-update window before5k; eligible fraction is87.5%.
Deep teacher residual clipping is6.87%, repair-flow loss.427065, and anchor/teacher
action MSE.032702/.021947. The teacher contains the recorded action label;
its lower MSE is not measured action-generation quality.

Corrected executed-prefix MAE is.0015608 versus prior.0025887, and is
7.55% lower than old repair-flow's5k corrected MAE, while success is five
cases lower. Better calibration does not establish better control. Raw model
scores have optimistic/nonpositive-observed latent-progress tails in20/29
failures, corrected scores in12/29. These are latent proxies, not contact-force
or physical-cause measurements. Per-task successes are preserved in the full
analysis: assembly3/8, faucet-open0/8, pick-place1/8 and reach1/8 remain weak.

[Full first report](reports/20261011-retrieval-repair-ttt/run/validation/step_0005000.json.gz),
[paired cases and training diagnostics](reports/20261011-retrieval-repair-ttt/analysis-round-1.json),
[anchor observations](reports/20261011-retrieval-repair-ttt/anchor-outcomes-round-1.json),
and [source/process/online verification](reports/20261011-retrieval-repair-ttt/interim-health-round-1.json).
All case identities match the frozen references; source remains clean at3bdfb49
with137 matching runtime hashes. Owned coordinator819165, launcher821683 and
four ranks remain verified; W&B6wpx3jcv is running and publishes the same5k score.
At the compute snapshot, training had9776 updates with.850860 optimization GPUh
and.504677 evaluation GPUh. All391 periodic training records through9751 and the
registered evaluation record are preserved losslessly. Collection spans advancing
updates: health observed9772 and the subsequently copied compute record9776.
The5k checkpoint remains
on the server at`campaign/runs/retrieval_repair_ttt_3072/checkpoints/step_0005000.pt`,
SHA256`36e54249892eb0c8778a586076fd5de113935333490e5775ddeec6377ec992c4`.

[Measured cumulative components](reports/20261011-retrieval-repair-ttt/compute-accounting-round-1.json)
now total92.302822 GPUh at this partial snapshot, including all
recorded preflight, no-update checks and retrieval setup. Allowances remain
unchanged. Reused development cases, one seed, checkpoint selection and baseline
port limitations prevent independent generalization, causal-component or SOTA
claims. No baseline rerun, new candidate or sealed test was launched.

# Training verified: retrieval-conditioned residual flow

Readback 2026-10-11T21:08:12.257335+00:00: source `3bdfb49e7c9d4952da7494e50c8dc2530197641e` is clean and all137
runtime file hashes match. Coordinator819165/start1007321350 and launcher821683/
start1007334803 own four training ranks821698–821701/start1007334944 on GPUs0–3.
The run reached at least421 updates, with finite periodic losses and unchanged
global64/20k-or28,800 optimizer-GPU-second caps. W&B
[6wpx3jcv](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/6wpx3jcv) is running. The first fixed104 evaluation
is pending; no task-success improvement is claimed. At21:09:13UTC, a later read
confirmed local step1164 and online W&B step751/loss.688086; upload lag is explicit
in [online progress](reports/20261011-retrieval-repair-ttt/online-progress.json).
The zero retrieval-anchor metric at751 is a shallow update; deep updates must
be analyzed separately.

All39 final-source tests pass. Training-cache preflight checks unchanged sampled
bytes,10,000 task/episode draws, exclusion of the query episode and controller
latency79.61ms. The four-rank check verifies synchronized gradients at both
depths, unchanged parameters and no world gradients. Preflight.00597479GPUh,
no-update distributed check.00453653GPUh and retrieval setup.00228342GPUh are
recorded separately. At the compute snapshot, optimization.03855183GPUh;
subsequent periodic metrics are later snapshots of the same advancing run.
[Launch evidence](reports/20261011-retrieval-repair-ttt/launch-verification.json)
preserves process identities, source/data hashes, tests, allocation and metrics.
Reporting changes after the frozen revision do not change the execution checkout.

# Cross-episode retrieval evidence for residual-flow training

One next candidate, `retrieval_repair_ttt`, is registered under the October 11
sequential authorization. The preceding repair-flow run is complete and remotely
published at `15852d5c4fa550c22c0f6b6d9142bd7435593792`: 80/73/75/75, selecting
5k80/104 and tying the preserved GMM80. Its 146 ledger artifacts and 41 registered
reports were verified on origin/main before this implementation. This candidate
has no evaluation result yet. Training is gated by final-source tests, cached
training-data preflight and distributed checks.

## Specific evidence and hypothesis

The old `FlowReasoningPolicy.deliberate` constructs four generated proposals for
training. Its deployment controller replaces the fourth proposal for seven of
eight targets with a retrieved action chunk, and can shift an earlier plan into
another slot. The residual-flow teacher thus never trains on the retrieved-action
slot that can provide its deployment incumbent. This is a confirmed code-path
difference; it is not a proved explanation of task failure.

Repair-flow's selected model still fails 24/104 cases, all at the action ceiling,
including seven assembly, six faucet-open and six pick-place cases. At 20k,
calibration and teacher action losses improve while success falls to75. Saved
5k traces select retrieved *targets* in4,158/5,173 decisions and exactly match a
shifted prefix in1,206; corresponding failure-only counts are1,411/2,400 and703.
The old logs do not identify the action source of the repair anchor. A retrieved
target is not proof that a retrieved action was selected. The
[saved-record rationale](reports/20261011-retrieval-repair-ttt/rationale.json)
keeps this limitation explicit.

Hypothesis: exposing residual training to a cross-episode retrieved action in
its verified candidate pool reduces one distribution mismatch. This is a single
specified correction, not a diversity/radius/learning-rate sweep. Its result may
be negative. Shifted-plan training coverage and on-policy state distribution
remain unresolved; this is not full deployment-distribution matching.

## Exact correction and causal boundary

For each deep training update, use the same sampled current state, local requested
image target and global image goal. Query only the unchanged training route bank's
five-block routes. Rank an approximate shortlist of256 with exact start/endpoint
cosine distance. **Exclude the entire query episode before shortlisting and exact
reranking**. The lookup receives no teacher action, next-state error label or task
success label. The local requested target is already the existing goal-conditioning
input; no new sensor or outcome field is introduced.

Replace one of four generated proposals with the retrieved five-block action in
seven eighths of global training examples. The remaining eighth has no retrieved
slot, matching the controller's seven retrieved targets plus one direct-goal slot
count; training still uses the original local/global target sampler. Score every
candidate through the frozen world with the existing factual calibration, select
the best incumbent, and train the same clipped action residual:

`delta_target = clamp((recorded_action - stopgrad(incumbent)) / 0.25, -1, 1)`.

The teacher interpolant is used only for the residual imitation loss. Only the
recorded action receives its own recorded outcome-error labels. Generated and
retrieved actions receive model predictions, never inherited physical outcomes.
All learned weights are fresh. No policy/world/encoder/baseline is fine-tuned or
silently inherited. The learned episodic adaptation mechanism is unchanged.

The deployment policy has the same6,466,320 parameters, two rounds, four candidates
per target, exact incumbent preservation, radius.25, eight global plus four repair
flow steps and320 world transitions. It executes one two-primitive-action prefix,
under10 cumulative controller seconds and200 actions. Only diagnostics additionally
record whether the first-round repair anchor was generated, retrieved or shifted.
This provenance labels a construction slot, not causal attribution.

## Literature and implementation grounding

[Residual Policy Learning, Silver et al.](https://arxiv.org/html/1812.06298v1)
adds a learned corrective policy to a base controller and trains the composite
behavior. Its discussion of using the controller on the correction's training
states motivates checking our anchor distribution. Their method is online
model-free RL; this candidate is offline goal-conditioned residual flow imitation,
so it is an adaptation of a general idea, not an RPL reproduction or new claim
to residual control.

[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) and its
[official image-policy implementation](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_unet_image_policy.py)
condition action generation on observations and execute a prefix of the sampled
sequence. We preserve that observation/action timing discipline. This does not
reproduce its architecture or establish that retrieval training is novel. The
existing LeFlow generated-consistency precedent remains acknowledged.

Local code reviewed: `flow_reasoning.py:deliberate`, `flow_controller.py:plan`,
`repair_flow_ttt.py:forward/revise`, `bank.py:RouteBank`, `revision.py` and
`aligned.py` samplers. New isolated code is `execution/retrieval_repair.py`;
historical frozen server checkouts are untouched.

## Fixed contract and completion obligations

- Seed3072 only; same6,222 expert episodes, exact windows/targets/actions and
  fixed104 development cases, global64 and frozen world/encoder/statistics.
- One candidate on four idle GPUs, acquired non-preemptively with advisory locks.
  At most20k updates or28,800 aggregate optimization GPU-seconds. Retrieval queries
  are inside the charged update timer. Bank setup is separately measured and its
  ledger accumulates on resume; prior known campaign compute is90.934490GPUh.
- Four registered evaluations at5k/10k/15k/20k; no extra baseline training,
  seed, inference sweep, new data or sealed final test. Same fixed visual cases.
- Verify source/runtime/config/data/checkpoints, periodic metrics, complete reports,
  process exit and online completion. Preserve failures and costs; publish to
  origin/main and verify remote artifacts before deciding any next correction.
- Compare paired cases to both tied80 references, flow79, progress78 and guided75,
  including whether retrieved incumbents occur and are corrected, saturation,
  actual success, timeouts and action-budget failures. Teacher residual loss and
  latent prediction errors are not physical ground truth.
- Stop new launches for review after an improvement over80 or when no specific
  supported correction remains. Reused development cases cannot establish
  independent generalization, seed robustness, isolated causality or SOTA.

## Verification status before launch

Five new behavior tests pass on the project runtime using CPU only: whole-episode
exclusion with exact reranking, teacher isolation, retrieved candidate world
verification/selection, both-depth gradients with frozen world, and controller
accounting/reset/provenance. The initial test harness incorrectly applied a
proposal-pool assertion to the factual calibration call; that test-only error was
corrected, with no optimizer updates or GPU use. The final coordinator also reruns
inherited behavior checks before the one training-data preflight and no-update
four-rank gradient verification. The cached preflight checks exact old data bytes
for eight examples and10,000 unchanged task/episode draws.

Server record: `/home/mtxu/adam/LeFlow-experiments/20261011-retrieval-repair-ttt`.
Runtime: `/tmp/mtxu-flow-jepa-20261007/env/bin/python`. Current identities and frozen
source are recorded in the active ledger and launch evidence after dispatch.
