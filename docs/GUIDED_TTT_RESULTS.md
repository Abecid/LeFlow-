# Guided TTT: completed comparison and failure analysis

Verified October 11, 2026. The authorized seed3072 run completed at October 10,
16:52 Pacific, with **72/70/72/75 successes out of 104** at 5k/10k/15k/20k
updates. The final checkpoint is selected: **75/104 (72.12%)**. This improves
over the fixed LeFlow shared-world port's 21/104, but is below our preserved
execution-revision controller's **80/104 (76.92%)**. This iteration does not
establish a benefit from guided TTT or useful inference scaling.

[Method and training objective](GUIDED_TTT.md) ·
[Full saved evidence](reports/20261010-guided-ttt/) ·
[W&B, finished](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/dw8gnxtj) ·
[All comparison runs and publication policy](RESULTS_INDEX.md)

## Fixed comparison contract

The task remains image-goal control on 13 MetaWorld v3 tasks. The same 6,222
successful expert episodes train the policy and provide retrieval. The shared
world's training split contains 7,800 episodes; the V-JEPA2.1 encoder and world
remain frozen. Seed3072, global batch64, 20,000 updates or 28,800 aggregate
optimization GPU-seconds, whichever comes first. This run sampled 1.28M training
windows with replacement, rather than a fixed number of epochs.

Each checkpoint uses the identical 104 development cases, eight per task.
Success requires the task criterion within 200 primitive actions and 10 cumulative
controller seconds, including online adaptation and guidance. The largest
development success selects the checkpoint, with earliest ties. The 3,200-case
final suite remains sealed. No baseline, extra seed or additional evaluation was
run for this completion analysis.

The new method generates actions with a flow model and checks them with JEPA
predictions; LeFlow generates latent paths followed by inverse dynamics. This is
a full-controller comparison. The guided candidate uses two search rounds and
320 world transitions per decision; prior action-flow candidates used three
rounds and 480 transitions. It also adds a surrogate critic, guidance and new
training losses. Those joint changes prevent attributing the result to TTT alone.
Equal ceilings and data do not imply equal consumed compute or convergence.
These are shared-world ports on a custom development benchmark, not the authors'
native published benchmark results. HWM is a paper port with release limitations.

## Learning curve and paired outcomes

| Updates | Success /104 | Success % | Controller timeouts | Mean decision latency |
| --- | ---: | ---: | ---: | ---: |
| 5,000 | 72 | 69.23 | 31 | 103.12 ms |
| 10,000 | 70 | 67.31 | 32 | 103.13 ms |
| 15,000 | 72 | 69.23 | 31 | 103.00 ms |
| 20,000, selected | **75** | **72.12** | **29** | **103.46 ms** |

| Preserved comparison | Selected success /104 | Guided-TTT wins | Guided-TTT regressions |
| --- | ---: | ---: | ---: |
| Execution revision | 80 | 2 | 7 |
| Reasoning-conditioned action flow | 79 | 3 | 7 |
| Flow + episodic adaptation (progress TTT) | 78 | 3 | 6 |
| Controller-grounded planner | 79 | 6 | 10 |
| LeFlow release-module/spatial port | 21 | 60 | 6 |
| HWM paper port | 5 | 70 | 0 |
| Released CEM controller | 24 | 52 | 1 |

Episode hashes, reset seeds, model seeds and task identities match. Against the
80/104 reference, gains are dial-turn6 and door-open0; regressions are assembly6/7,
door-open4, faucet-open0, pick-place4, plate-slide1 and reach1. These are selected
checkpoints on repeatedly reused development cases and one training seed; they
do not establish general superiority or isolate component effects.

## Measured failure signals

**The controller allowance binds.** All 29 failures at20k exhaust the 10-second
allowance after 182–198 primitive actions. The previous80/104 controller averaged
86.74ms and had zero timeouts; the79/104 flow controller averaged99.42ms and had
10. The new run's mean is103.46ms, p95 is111.18ms. With two actions per decision,
100ms is approximately the average required to reach200 actions within10s.
The earlier96.94ms five-decision training-image timing check did not predict full
episode latency reliably. Timeouts describe termination, not proof that more
time would solve the task: stalled decisions can also consume the full budget.

**Better prediction accuracy has not delivered better decisions.** At20k, mean
absolute error on the same executed prefixes is .002416 for the raw world,
.001773 with the offline prior and .001632 after adaptation: a7.92% reduction
relative to the prior. This is smaller than the prior progress-TTT run's26.0%
reduction and does not establish counterfactual ranking quality. Only executed
prefixes have physical outcome labels.

The inexpensive guidance critic has .001816 MAE relative to the world cost and
.002479 MAE relative to observed executed-prefix costs, versus .001632 for the
corrected world diagnostic on those observations. The surrogate is therefore a
weaker value predictor on these traces. Scalar MAE cannot tell us whether its
action gradients point toward more executable progress. The trust gate is
positive on57.76% of eligible decisions, with mean trust .088 across all decisions;
it is a heuristic gate, not calibrated uncertainty.

**Optimistic stalls remain.** In the final20 decisions of each failed episode,
20/29 failures have positive mean raw predicted progress but nonpositive mean
observed progress; eight remain optimistic after signed correction. Ranking also
uses other costs and positive-only correction penalties, so these signed
diagnostics are not a complete reconstruction of the selection objective.

**The difficult tasks remain difficult.** Selected success by task is assembly1/8,
door-open6/8, faucet-open2/8, pick-place0/8, plate-slide7/8 and reach3/8; the other
seven tasks score8/8. The final checkpoint is better than the earlier checkpoints
in this run, so this is not monotonic deterioration with training.

All ten saved20k contact sheets were visually inspected. In
[pick-place0](reports/20261010-guided-ttt/contact-sheets-20000/pick-place-00000.png),
the gripper moves toward the goal area while the object remains on the table in
the sampled frames. In
[faucet-open3](reports/20261010-guided-ttt/contact-sheets-20000/faucet-open-00003.png),
the late frames show little visible change near the handle. These illustrate
incomplete object manipulation, not a diagnosis of its cause. The saved
[visual review](reports/20261010-guided-ttt/visual-review-20000.json) records the
successful examples and the limits of uniformly sampled, single-view images.

**Proposal dispersion shrinks, but its causal role is unknown.** At matched
initial states, goals and seeds with empty history, final-round proposal
dispersion falls from .056075 at5k to .021569 at20k (61.5%); all104 cases decrease.
Success nevertheless increases from72 to75. This is not a measure of useful
exploration, policy entropy or proof that collapse caused the failures.

The last500-update window's logged flow loss is .088379, down from .151812 near5k.
Guidance improves the supervised teacher-action MSE by .000343 on those logged
training batches; adaptation's extra gain over prior guidance is only .00000137.
The two-depth flow-loss difference is slightly negative (-.000413). These are
training diagnostics, not task-success gains. Logging every50 updates aliases
the alternating depth schedule: these window statistics cover depth2 only;
depth1 appears only at step1. Both depths were trained and audited, but the
periodic logs cannot characterize the full depth1 learning curve.

## Verification, compute and next decision

The CPU [final audit](reports/20261010-guided-ttt/analysis/final-audit.json)
passed:143 runtime hashes, clean frozen source, config/manifest/world identity,
identical route-bank tensors,6,222 training episode identities, finite best/last
checkpoints, selected milestone tensor identity,401 training records, all four
validation records, matching cases, causal episode-local history/reset, score
reconstruction and320 world transitions per decision. Online W&B is finished;
the coordinator, trainer launcher and analysis worker exited. The saved-record
analysis was independently rerun after download and matched the server result.
All transferred evidence was checked against its source SHA256.

Frozen source: `92fdae6120c568a6e13c7ca2e6013fbc72cb45bf`.
Server record: `/home/mtxu/adam/LeFlow-experiments/20261010-guided-ttt`.
Selected checkpoint: `campaign/runs/guided_ttt_3072/best.pt`,20k.
Optimization consumed4.594631 GPU-hours, evaluations2.165615 GPU-hours, and the
coordinator's no-update preflight .005745 GPU-hours (6.765991 timed GPU-hours
combined). Earlier implementation/eight-GPU checks are separate launch evidence.
Training/evaluation wall time was51.14min; coordinator wall time53.46min.
No optimization-budget overrun occurred. Raw data and model binaries remain on
the server; compressed full evaluation records and training logs are in Git.

The evidence supports addressing controller runtime and validating guidance's
decision quality before adding more inference work. A future candidate should
preserve the controller allowance, measure whether the surrogate's suggested
action changes improve real progress, and log both training depths. These are
recommendations, not newly authorized or queued experiments. Baselines and the
80/104 best reference remain preserved; no new training or sealed test was launched.
