# Corrected, isolated comparison baselines

The October 9 user request explicitly authorizes correcting and rerunning the
baselines once, then freezing their checkpoints and case-level outcomes. Historical
LeFlow/HWM numbers are not evidence of beating faithful implementations.

This is a **shared-backbone MetaWorld port**, not a reproduction of the authors'
native benchmarks or pretrained backbones. The shared frozen world is unchanged:
`ae3f910da43f0a43dd65945394444a0b11719ed042cdacffacc2faaf824ec5a3`.
The baseline heads/controllers import no code from our controller-grounded,
latent-revision or execution-revision policies. There is no retrieval, learned
workspace, execution-error correction, stall rule, candidate reuse, endpoint-flow
repair or extra action-refinement stage in LeFlow. CEM is the released solver.

## Source and recipe audit

| Component | Source retained | Necessary port / comparison changes |
|---|---|---|
| LeFlow flow and inverse models | [Author release](https://github.com/hsiangwei0903/LeFlow/tree/f1fe192e41ec6f20de25cd1054ede40a8cbdcfa5), five verbatim symbols with individual SHA256s and license | Compact state512, action8=two physical MetaWorld actions; input backbone differs |
| LeFlow spatial representation | [Appendix E](https://arxiv.org/html/2608.24855v1#A5): learned positions, summary token, two-layer Transformer,512-dimensional encoding; reconstruction MSE+cosine, then frozen | Paper does not release decoder architecture/training schedule. We specify a two-layer512/8-head training-only decoder;2000 adapter updates counted inside the total20000. No proprioception because the benchmark is image-only |
| LeFlow learning | Native velocity flow-matching MSE; native inverse MSE on **recorded** paths;0.1 frozen-world consistency; zero smoothness | Recorded-action normalization comes solely from eligible training episodes. Frozen full-spatial world predictions are compressed for consistency. The release trains inverse/consistency on recorded paths although paper prose describes generated paths; we follow executable release code |
| LeFlow inference | H5,64 candidates,16 Euler steps, five-block receding execution; original spatial-world endpoint MSE ranking | One world block is two primitives here, versus five in native experiments. No action clipping before world scoring; environment retains its required action bounds |
| Flat CEM | `stable-worldmodel==0.0.6`, exact installed `solver/cem.py` SHA256 `d88c86dcd1bd1e6d89221ac22079a3efe296cc7566532de0d36605e2f1536050`;300 samples,30 iterations,30 elites,H5,receding5,unit initial standard deviation | Shared world interface converts normalized actions to physical units and scores full-spatial endpoint MSE. Native optimized elite mean, unbiased elite standard deviation, mean candidate insertion; no our CEM changes |
| HWM | [Author repository](https://github.com/kevinghst/HWM_PLDM/tree/e197375b844692a0a2e1342889f95a78edced07a) explicitly releases only PLDM DiverseMaze. [Paper](https://arxiv.org/html/2604.03208v2) describes robot/PushT variants | Robot code is unavailable. A paper-based port must be labeled as such, with all unspecified choices disclosed. Do not report the old simplified fixed-stride MLP version as faithful HWM. Paper-port architecture and controller below; formal registration follows preflight |

Native LeFlow flow width512/depth4/8 heads and inverse width512/depth3 are retained.
Its512-dimensional output is full rank at initialization; the historical1024-state/
256-width sampler defect is absent without importing our noise-cancellation fix.
The paper's trained spatial compression is the reason to use512 dimensions.


## HWM paper-based port and controller-budget audit

The retained high-level architecture follows the paper's PushT setting:10-layer,
768-width,12-head,3072-MLP causal ViT,4-dimensional learned macro-actions,
5 variable waypoints, and only L1 teacher-forcing loss. The Transformer body is
from [DINO-WM source](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/models/vit.py),
with one recorded device-placement change to its attention mask, verified by
reconstructing the original source hash. No architecture/loss from our proposed
policy is used. AdamW5e-4/weight decay0.01/no gradient clipping follows the released
DINO-WM optimizer (default AdamW weight decay); global64/common step/compute caps
replace its native epoch/batch schedule.

Unreleased details necessarily remain explicit port choices: the action CLS
Transformer is2 layers/256 width/4 heads, with a256-hidden MLP to4 dimensions.
Shared1024-D visual patches project to758 channels and10 embedded macro-action
channels concatenate before the768-D predictor; output projects back to1024.
The shared benchmark has no proprioception. Segments cover13–35 cached blocks
(26–70 primitive steps, aligned to the two-step cache); three random interior
waypoints join the endpoints. These choices are not claimed to be author robotics
code. The frozen fine-world architecture remains the common benchmark model,
not the original DINO-WM or300M V-JEPA2-AC predictor.

A discarded-weight, training-case-only check of the paper's Table12 d25 controller
(high900×20,H2; low300×30,h5) measured **15.678 seconds per warm decision** on an
A800. It cannot act under our entire-episode10-second allowance. No validation
performance was used in choosing a replacement planning setting.

For the fixed-budget run, use only settings listed in the paper's Appendix C d50
compute study: high150 samples/10 iterations/H4/10 elites/standard-deviation
momentum0.4; low150 samples/10 iterations/h5/10 elites/momentum0; execute5 fine
blocks. These are the smallest published sample/iteration counts, selected before
any validation evaluation. Momentum smooths the standard deviation, as stated in
Appendix C; no candidate clipping/variance floor/best-sample replacement is added.
The zero-momentum solver matches the pinned native CEM numerically in tests.
This is a published low-compute configuration, **not the headline Table12 setting**.
A later evaluation with longer common allowance could reuse exactly these trained
weights with the headline planner; it would not require another training run.

Six HWM tests passed: DINO source provenance, padding exclusion, temporal causality
and past context, L1-only recorded-waypoint supervision, native CEM equivalence at
zero momentum, and the published standard-deviation momentum equations. Its
four-GPU native-setting preflight had finite gradients and exact reset RGB,
preserved world weights, used0.091724GPUh and discarded all weights. The smaller
planner passed its four-GPU preflight:4.55s mean warm decision,4.858s cold decision,
3.47GiB peak/rank,0.042500GPUh. It can finish roughly two decisions per10-second
episode, so timeout rates remain essential to interpretation. All17 baseline
unit tests passed on the final source after adding coarse-prediction accounting.

## Locked comparison contract

- One seed3072; no variants, sweeps or extra seeds.
- Same7800 train episodes (6240 expert,1560 random),200 primitive steps each:
  1.56M transitions. Frozen world used the original inventory. Learned heads use
  the same6222 successful expert episodes as our policy;650 validation episodes
  remain disjoint. Baseline sampling is task/episode uniform with the common
  pre-success start limit; no our-policy goal curriculum or recovery weighting.
- Same frozen V-JEPA2.1 ViT-L features,32×1024 patches, and same spatial fine world.
  Source manifest SHA256:
  `ce6e190b9b777ad34807c47c2dbeebae810f9cca52b440dbaf9e05043d69f1bc`.
- Global64; stop at20000 updates or28800 aggregate optimizer GPU-seconds,
  whichever comes first. Adapter learning is included. LeFlow's common-budget
  batch/step schedule differs from native batch128/epoch-based training; this is
  a fixed-budget comparison, not a native convergence reproduction.
- Four fixed104 validation evaluations (8 cases/task ×13 tasks) at quarter-budget
  milestones. Select highest macro success, earliest checkpoint on ties.
- Goal-conditioned success within200 primitive actions **and10 seconds of total
  controller time** is primary. Encoding/planning count; environment stepping and
  rendering do not. Execution prefixes follow each method's published recipe.
- Log success per task, paired case outcomes, success by action count, capped
  steps-to-success, controller latency/timeouts, return and world-call counts.
  Add fixed256-window held-out losses every1000 updates as convergence diagnostics;
  they do not select the checkpoint or establish task success.
- CEM reuses the fixed trained world; it needs **no additional training** and only
  one registered104-case evaluation, since its policy has no learned head.
-3200 final cases remain sealed:200×16 tasks, including3 unseen tasks. No final-test
  data is loaded by packing, training, preflight or development evaluation.

20000×64=1.28M sampled training windows, with replacement; there is no literal
full-dataset epoch counter. LeFlow allocates128k sampled frames to its adapter and
1.152M trajectory windows to its flow/inverse head. Comparing an 'epoch' count to
another paper with different window enumeration would be misleading.

## Verification and reuse

The CPU cache preserves original float16 features and float32 actions exactly.
Every source training/validation HDF5 SHA and copied value was checked; no
precision reduction.6872 successful expert train+validation rows occupy about
42.36GiB of features. Cache construction used CPU workers and no test files.

11 baseline tests passed on the training server: verbatim source hashes, objective
and sampler equations, full-rank output dimensions, frozen adapter, actual rollout
ranking, native CEM equations, variable execution-prefix limits, success within a
chunk, and no action execution after a timeout. A discarded-weight GPU preflight
checks finite gradients, unchanged frozen-world weights, training-reset RGB and
four-GPU stage transition before the formal launch.

Formal runs use clean, immutable Git checkouts. Configuration/data/world hashes,
RNG/optimizer state, per-update budget ledger, per-episode resume journals and
periodic checkpoints are preserved. Validation reports/selected checkpoints are
made durable before advancing resume state. A final registry hashes reports and
checkpoints; launching a sealed baseline verifies its files and exits without
retraining. Preflight and validation compute are reported separately.

## What these results can establish

104 development cases suffice to catch large failures. One case is0.96 percentage
points, and repeated use for method design makes this an unsuitable final test.
The reserved3200-case evaluation can support a stronger comparison after method
selection, with taskwise and paired uncertainty. This custom13-task image-goal
suite does not support an unrestricted MetaWorld/SOTA claim.

Convergence is not yet established. Falling training loss or our own late-stage
regression does not prove all baselines are trained sufficiently. Use the new
held-out losses and success curves to assess each run. If a baseline is still
improving at the cap, report budget-limited results; any convergence extension
needs a common prospectively chosen rule and matched allowance, not a selective
extra budget for our method.
