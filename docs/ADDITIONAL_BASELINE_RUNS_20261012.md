# Five baseline training and evaluation runs completed

All five fresh seed-3072 runs completed **20,000 updates and four fixed 104-case development evaluations each**. They used two disjoint four-GPU lanes on the eight available A800 80GB GPUs. Preserved LeFlow/GMM/retrieval references were reused.

| Method | 5k /10k /15k /20k successes | Selected /104 | Selected update | Timeouts |
| --- | --- | ---: | ---: | ---: |
| [Vanilla action flow](VANILLA_ACTION_FLOW_RESULTS.md) | 77 / 84 / 89 / 98 | **98** | 20,000 | 0 |
| [Action flow + world search](ACTION_FLOW_SEARCH_RESULTS.md) | 74 / 79 / 84 / 78 | **84** | 15,000 | 0 |
| [SAGE-inspired port](SAGE_PORT_RESULTS.md) | 62 / 77 / 77 / 75 | **77** | 10,000 | 0 |
| [Diffusion Policy port](DIFFUSION_POLICY_PORT_RESULTS.md) | 43 / 69 / 70 / 74 | **74** | 20,000 | 0 |
| [Feedback-WM-inspired port](FEEDBACK_WM_PORT_RESULTS.md) | 42 / 65 / 75 / 76 | **76** | 20,000 | 28 |

**The plain action-flow baseline is the strongest measured controller at 98/104. Our retained proposed retrieval-repair method remains 80/104 and does not outperform it.** The new world-search controller selects 84; its local-goal/retrieval interface differs from vanilla flow, so this does not isolate a single cause. Do not relabel the vanilla BC control as a novel reasoning/TTT method.

## Fixed experimental setup

- Same 6,222 successful expert TRAIN episodes, frozen V-JEPA 2.1 ViT-L features and shared spatial world, seed 3072 and global batch 64.
- Same ceiling: 20k optimizer updates OR 28,800 aggregate optimizer GPU-seconds per policy. All five stopped at the update ceiling, with zero cap overrun.
- Same 104 development cases: 13 tasks × eight resets, evaluated at 5k/10k/15k/20k; earliest ties select checkpoints.
- Same 200 primitive actions AND 10 cumulative controller seconds per case, including image encoding and planning.
- The reserved 3,200-case final benchmark remains sealed. No extra seed, ablation, prior-baseline retraining or budget extension was run.

Policies are small (about 2.5–5.0M parameters) and train on cached frozen features, which explains the short optimizer time. There are 1,280,000 sampled windows per policy across 20k global-64 updates; sampling is with replacement, so there is no conventional epoch count. Equal ceilings are not equal consumed GPU time, architecture, world calls or convergence. Vanilla and diffusion success are still rising at the last checkpoint; convergence is not established.

## Observed failures and interpretation

Vanilla has six action-limit failures: four pick-place and two reach cases, with zero timeouts. World search has 20 action-limit failures and regresses from 84 at 15k to 78 at 20k. Diffusion Policy has 30 action-limit failures, including all eight pick-place cases. Feedback-WM has 28 failures, all exhausting controller time at 166–198 actions; it also remains 0/8 on pick-place. Faster guidance could recover controller allowance, but these logs cannot tell whether more actions would succeed.

SAGE selects 77/104 with 27 action-limit failures and zero timeouts. Selected per-task counts and all factual case failures appear below and in its report.

The independent diffusion and feedback training checkpoints are close but not bit-identical at any saved milestone (maximum absolute 20k weight difference 0.00053736). Feedback wins five cases and loses three against diffusion; 76 versus 74 is not a feedback-only causal ablation. SAGE and Feedback-WM are paper-inspired ports; Diffusion Policy is an image-goal/V-JEPA adaptation. Exact upstream differences remain in the [method specification](ADDITIONAL_BASELINES_20261012.md).

One seed and repeatedly reused development checkpoint selection do not establish native-benchmark SOTA or independent generalization. The simpler baseline changes the target the proposed mechanism must improve. This batch is a completed comparison, not authorization for another run.

## Selected per-task success /8

| Task | Vanilla flow | World search | SAGE port | Diffusion port | Feedback port |
| --- | ---: | ---: | ---: | ---: | ---: |
| assembly | 8 | 5 | 0 | 1 | 2 |
| button-press-topdown | 8 | 8 | 8 | 7 | 8 |
| coffee-button | 8 | 8 | 8 | 8 | 8 |
| dial-turn | 8 | 8 | 1 | 2 | 3 |
| door-close | 8 | 8 | 8 | 8 | 8 |
| door-open | 8 | 6 | 8 | 8 | 7 |
| drawer-close | 8 | 8 | 8 | 6 | 6 |
| drawer-open | 8 | 8 | 7 | 8 | 8 |
| faucet-open | 8 | 6 | 8 | 7 | 7 |
| handle-press | 8 | 8 | 8 | 8 | 8 |
| pick-place | 4 | 1 | 0 | 0 | 0 |
| plate-slide | 8 | 8 | 8 | 8 | 8 |
| reach | 6 | 2 | 5 | 3 | 3 |

## Actual resource consumption

| Method | Optimization GPUh | Evaluation GPUh | Train + eval wall minutes | Mean selected decision ms |
| --- | ---: | ---: | ---: | ---: |
| Vanilla action flow | 0.613096 | 1.475266 | 31.79 | 55.83 |
| Action flow + world search | 0.601418 | 1.836139 | 37.01 | 91.62 |
| SAGE-inspired port | 0.714042 | 1.807850 | 38.38 | 75.02 |
| Diffusion Policy port | 0.592600 | 2.238335 | 42.87 | 81.37 |
| Feedback-WM-inspired port | 0.581692 | 2.275778 | 43.29 | 107.17 |

Total measured optimization, evaluation, GPU preflight and no-update DDP checks for this batch: **12.774646841 GPUh**. CPU route-bank adoption, audit time and idle allocation are not GPU optimization charges. This is component accounting, not a complete server bill. The common world/encoder was reused rather than trained five times. The two-lane queue completed in about 116.43 wall minutes, including launch/preflight work.

## Verification and preserved evidence

All five CPU audits verify the 214 frozen runtime files, source/config/manifest identities, 6,222 TRAIN route IDs, unchanged bank tensors/world hash, finite checkpoint weights, every saved milestone and selected checkpoint. All four-rank preflight gradient checks pass. All five coordinators exited successfully and all five W&B runs report finished. The persistent queue completed with five zero return codes.

Full periodic metrics, all 20 physical evaluation reports, per-task/paired analyses, process/online identities, checkpoint paths/hashes and actual compute records are preserved in the [batch archive](reports/20261012-additional-baselines/), [selected table](reports/20261012-additional-baselines/selected-results.csv), and [per-case matrix](reports/20261012-additional-baselines/selected-per-case.csv). Large model/data binaries stay on the authorized server.

Fifty contact sheets from the original saved selected-checkpoint trajectories across all five methods were individually visually inspected. They show measured frames and an actual expert goal, with primitive action indices. The saved diffusion pick-place example moves the hand while the object remains behind; its assembly example stays near the starting object location. These observations do not establish a physical cause and are not a representative sample of all cases. Their source trajectory hashes and per-image visual checks are preserved.

Execution source: `0a19c7c67d791fd99376216399b03e8cb71d78c0`. Server batch: `/home/mtxu/adam/LeFlow-experiments/20261012-additional-baselines`. The [ledger](results-ledger.json) and [publication verifier](../scripts/analysis/verify_result_publications.py) cover all 21 retained current comparisons and 81 complete evaluation reports.
