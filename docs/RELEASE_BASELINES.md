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
| HWM | [Author repository](https://github.com/kevinghst/HWM_PLDM/tree/e197375b844692a0a2e1342889f95a78edced07a) explicitly releases only PLDM DiverseMaze. [Paper](https://arxiv.org/html/2604.03208v2) describes robot/PushT variants | Robot code is unavailable. A paper-based port must be labeled as such, with all unspecified choices disclosed. Do not report the old simplified fixed-stride MLP version as faithful HWM. Implementation/registration pending |

Native LeFlow flow width512/depth4/8 heads and inverse width512/depth3 are retained.
Its512-dimensional output is full rank at initialization; the historical1024-state/
256-width sampler defect is absent without importing our noise-cancellation fix.
The paper's trained spatial compression is the reason to use512 dimensions.

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
