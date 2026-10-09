# Controller-grounded first iteration: completed validation

October 9, 2026. The first candidate completed 20,000 updates and all four
registered validations. **The selected checkpoint scores 79/104 (75.96%) at
update 5,000**, substantially above our saved references. This is development
validation on one model seed, not a final-test or corrected-SOTA claim.

## Measured comparison

Every row uses the same 104 validation case IDs, reset seeds, episode hashes
and model seed 3072. Success within the fixed 200 primitive actions and
cumulative 10-second controller allowance is the primary outcome.

| Method/checkpoint | Successes | Success rate | Comparison qualification |
| --- | ---: | ---: | --- |
| Controller-grounded, selected 5k | 79/104 | 75.96% | New complete pipeline; selected from four rounds |
| Saved LeFlow adaptation | 27/104 | 25.96% | Historical world/representation and sampler confounds |
| Saved same-world CEM | 22/104 | 21.15% | Same selected fine world; different controller |
| Saved repaired flow | 17/104 | 16.35% | Same selected fine world |
| Saved HWM adaptation | 8/104 | 7.69% | Historical world/representation |

The historical old-world CEM scored 7/104 and original flow scored 23/104;
both remain preserved. No baseline was retrained or rerun. Against same-world
CEM, 59 cases improve and 2 regress; against historical LeFlow, 53 improve and
1 regresses. All six saved-reference pairing checks passed. The gain cannot
be attributed to any single component without further experiments.

| Training update | Validation successes | Rate |
| --- | ---: | ---: |
| 5,000 | 79/104 | 75.96% |
| 10,000 | 78/104 | 75.00% |
| 15,000 | 75/104 | 72.12% |
| 20,000 | 78/104 | 75.00% |

The registered earliest-tied-best selection chooses 5k. More training did not
improve the primary outcome despite improving demonstrated-action likelihood.
Mean logged action NLL over each checkpoint's last 500 updates is approximately
−1.387, −1.657, −1.757 and −1.787. This is a likelihood/control mismatch, not
proof of overfitting or mode collapse. These checkpoints follow different
evaluation trajectories; proposal-diversity changes do not isolate a cause.

## Actual formulation and data

The [implemented algorithm](CONTROLLER_GROUNDED_RUN.md) retrieves observed
training routes, proposes short actions with a newly trained four-component
Gaussian mixture, and scores each target using its bounded local controller
response. It jointly selects the target and response, then executes the first
two primitive actions from that exact response. Each decision evaluates 480
fine-world transitions. This first successful candidate uses a **Gaussian
mixture action head**, not a diffusion head or latent-reasoning module.

The new head has 2,716,228 parameters. Cached V-JEPA 2.1 spatial features and
the selected action-conditioned fine world remain frozen. Action likelihood
and a small observed-prefix/endpoint auxiliary loss supervise the policy.
Recorded states are targets; they are not claimed to be observed outcomes of
the newly generated actions used in the auxiliary model rollout.

The custom MetaWorld v3 image-goal dataset covers 13 training tasks, not the
standard MT10/MT50 protocol. Training contains 7,800 episodes: 6,240 expert and
1,560 random. This head and its route bank use the same 6,222 successful expert
episodes as the prior heads. Sampling preserves the original 60-control-step
window draws and global batch 64. Episodes contain 100 control blocks / 200
primitive actions. The validation pool has 650 episodes; the fixed registered
subset is 104, eight per task. The separate 3,200-case, 16-task final-test set,
including three held-out tasks, remains sealed.

## Where it fails

| Task | Selected checkpoint successes / 8 |
| --- | ---: |
| assembly | 3 |
| button-press-topdown | 8 |
| coffee-button | 8 |
| dial-turn | 7 |
| door-close | 8 |
| door-open | 8 |
| drawer-close | 8 |
| drawer-open | 8 |
| faucet-open | 4 |
| handle-press | 8 |
| pick-place | 1 |
| plate-slide | 6 |
| reach | 2 |

All 25 selected-checkpoint failures reach 200 primitive actions; none exhausts
the controller clock. Seventy cases succeed within 100 primitives. Mean
decision latency is 81.42 ms, p95 is 90.04 ms, and mean cumulative controller
time is 4.162 seconds per episode. The weak tasks warrant execution analysis,
not a blanket claim that more available compute will solve them.

Across 5,212 observed executed prefixes, predicted and observed target-distance
progress correlate 0.745. However, **23.86% of predicted-positive prefixes have
nonpositive observed progress**. In **19 of the 25 failed episodes**, the last
20 decisions average positive predicted progress and nonpositive actual
progress. These are latent-distance measurements on dependent decisions;
they do not establish physical success calibration or causal mechanisms.

All six failed reach episodes choose the direct goal for their last 20
decisions; five have nonpositive mean observed progress in that tail. The
inspected reach/00003 contact sheet shows an initial approach followed by a
visibly offset posture with little change late in the rollout. Its tail mean
predicted progress is approximately +0.000955 and observed progress −0.000182.
We cannot infer an exact geometric error or contact mechanism from these views.

Direct-goal choices account for 41.12% of decisions in failed episodes, versus
0.249% in successful episodes. This association can reflect entering difficult
states; it is not evidence that disabling the direct option would help.
Controller scoring changes the retrieval-only target choice in 31.68% of all
decisions. Cross-task anchors account for 5.07% of retrieved choices, but task
labels do not enter the planner and cross-task support is not automatically bad.

The six registered [contact sheets](reports/20261009-controller-grounded/contact-sheets-step-5000/index.json)
were generated from saved actual rollout frames, checked against registered
initial images, visually inspected in full, and byte-verified after copying.
Five show successful rollouts and one the reach failure. No images were saved
for the pick-place/assembly failures, so no grasp/insertion diagnosis is claimed.
The sheets are evidence for specific cases, not a representative success sample.

Only the executed two-action prefix has an observed outcome. We still lack
counterfactual outcomes for rejected candidates and full execution of the
unexecuted five-control-step plans. Consequently **candidate-rank calibration
and real full-chunk reachability remain unresolved**.

## Budget, throughput and integrity

The run used eight A800 GPUs, global batch 64 and one seed. It stopped at the
20,000-update limit after **2.42156 aggregate optimization GPU-hours**, below
the common 8 GPU-hour ceiling. Four registered validations cost 1.94079 GPUh;
the training-data-only preflight cost 0.00557 GPUh. New charged components total
4.36792 GPUh. A further 0.52109 GPUh is an upper bound on reserved startup,
file-IO, renderer-failure and shutdown occupancy, reported separately.

Across the recorded MetaWorld campaigns, known charged components now total
47.08105 GPUh, including 31.79230 optimization and 15.14171 completed-validation
GPUh. This is not the complete infrastructure bill: historical preparation and
interrupted test infrastructure were not comprehensively totaled. Equal run
ceilings do not imply equal realized FLOPs or wall time. See the
[compute ledger](reports/20261009-controller-grounded/compute-accounting.json).

Logged training throughput near checkpoints was approximately 1,260–1,360
examples/second. The 8.003 GiB route cache was built in 28.02 CPU wall seconds
after replacing an unsuccessful 184-second threaded attempt with 16 independent
read processes; sampled features remained bitwise identical. Index approximation
and planner changes are algorithm choices, not described as lossless speedups.

An initial launcher omitted the required Mesa environment. Evaluator startup
failed before any scored episode, and the same 5k checkpoint, optimizer, RNG
and W&B run resumed after restoring the renderer and verifying a training reset
image bitwise. No optimizer update was repeated or discarded. The failure and
occupied resources remain in the [recovery record](reports/20261009-controller-grounded/renderer-recovery.json).

The [final CPU-only audit](reports/20261009-controller-grounded/postrun-audit.json)
verified all 401 unique scheduled training metric entries, four complete
validation records, caps, finite checkpoint tensors, selected/final steps,
unchanged world/manifest/bank hashes and a clean frozen execution checkout.
Four behavioral tests and sampled window/bank equivalence checks passed before
training. The coordinator completed normally at 20:06:38 UTC, W&B synced, and
all eight GPUs were idle with no owned worker remaining at the final inspection.

## Reproduction records and next direction

- Frozen execution source: `74a0a715cc393ba26285deb4571006ee7bc4d286`.
- Server run: `/home/mtxu/adam/LeFlow-experiments/20261009-controller-grounded/campaign/runs/controller_grounded_3072`.
- Selected `best.pt` SHA256: `7a375f211bc49b29ae493a76e6e171f568a30955241d3760a7350d1699123315`.
- Final `last.pt` SHA256: `79bb3bc45cd6a39a31b99bbda317121cb3d0fa58a6dd48c2256419f81bfcd31d`.
- [W&B](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/s12cr0yl),
  [all-round analysis](reports/20261009-controller-grounded/validation-review.json),
  [compressed raw validations and training metrics](reports/20261009-controller-grounded/run).

The current evidence supports investigating execution-error-driven plan revision.
The [LARC applicability review](LARC_APPLICATION_20261009.md) develops a proposed
latent-memory mechanism and checks closer robotic prior art. This is a design
proposal, not an implemented second candidate. No additional training, baseline
rerun, ablation or final test is queued. The central novelty and causal benefit
of any next mechanism remain to be demonstrated.
