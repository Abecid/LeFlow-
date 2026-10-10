# Corrected baseline results: completed and frozen

Both learned baseline runs completed **20,000 updates and all four fixed 104-case
evaluations**. CEM completed its one evaluation and requires no additional learned
head. All workers exited successfully, all three online runs are finished, and
checkpoint/report registries were independently verified after completion.

| Method | Selected development success | Selected update |
|---|---:|---:|
| Our preserved execution revision | 80/104 (76.92%) | 5,000 |
| LeFlow release modules + spatial port | 21/104 (20.19%) | 10,000 |
| HWM paper port, published smaller planner | 5/104 (4.81%) | 15,000 |
| Released CEM on the shared world | 24/104 (23.08%) | No extra head training |

These are **shared-backbone, fixed-budget MetaWorld ports**. They do not reproduce
the authors' native benchmarks/backbones. HWM's robot code is unavailable; its
result must remain labeled a paper-based port. LeFlow retains the released flow
and inverse models plus a documented port of its paper's spatial adapter; CEM retains its released
solver. Our retrieval, workspace, execution-error correction, stall penalties,
plan reuse and action-refinement machinery are absent from the baseline policies.
Read the [source, adaptation and budget audit](RELEASE_BASELINES.md).

The large external-baseline gap does not establish the benefit of our newest
component. Our preserved internal reference already scored 79/104; the newest
revision's gain over that reference remains one case on reused development data.

## Complete learning curves

Success counts and timeout counts below are out of 104. Selection uses the highest
macro success among the four registered checkpoints, with the earliest tie.

| Update | LeFlow successes | LeFlow timeouts | HWM successes | HWM timeouts |
|---:|---:|---:|---:|---:|
| 5,000 | 14 | 0 | 2 | 102 |
| 10,000 | 21 | 0 | 3 | 101 |
| 15,000 | 19 | 0 | 5 | 99 |
| 20,000 | 14 | 0 | 5 | 99 |

CEM scores **24/104**, with **80 timeouts**. Every CEM failure used only 80–90 of
the permitted 200 primitive actions before hitting the controller deadline.
HWM's smaller published planner takes about 4.55 seconds per decision; its
headline configuration took 15.68 seconds before even one decision completed in
the timing preflight. Thus HWM/CEM scores here measure performance under a tight
planning allowance, not their unrestricted task capability. A longer **common**
evaluation allowance can reuse the saved weights without retraining.

No action produced after the 10-second controller deadline is executed. A running
GPU kernel can finish slightly after the deadline; such rejected attempts and
their timing overruns are recorded rather than hidden.

## Data, metrics and convergence

- **Task:** custom 13-task MetaWorld-v3 image-goal control using frozen V-JEPA 2.1
  ViT-L spatial features and the same frozen spatial fine world.
- **Training inventory:** 7,800 episodes, 200 primitive actions each: 6,240 expert
  and 1,560 random episodes, or 1.56 million transitions. The shared world used
  this inventory; every learned policy head uses the same 6,222 successful expert
  episodes. No evaluation episodes enter policy training.
- **Training allowance:** one seed (3072), global batch 64, stop at 20,000 updates
  or 8 optimizer GPU-hours. All learned adapters count inside that allowance.
  Both runs stopped at the update ceiling. Actual GPU use differs, as shown below.
- **Epochs:** sampling uses replacement; 20,000 × 64 means 1.28 million draws,
  not a literal full-dataset epoch count. LeFlow uses its first 2,000 updates for
  the spatial adapter and the remaining 18,000 for flow/inverse learning.
- **Development:** 650 reserved validation episodes, with the same fixed 104
  cases (8 per task) used for all four controller evaluations. Fixed 256-window
  held-out losses are logged every 1,000 updates as diagnostics.
- **Primary metric:** macro task success within 200 primitive actions and the
  10-second controller allowance. Per-task/paired outcomes, timeouts, controller
  latency and steps to success explain the primary result; return and prediction
  losses are secondary diagnostics.
- **Final evaluation:** 3,200 cases (200 × 16 tasks, including three held-out tasks)
  remain sealed and were not evaluated by these runs.

| Update | LeFlow held-out flow MSE | LeFlow held-out inverse MSE | HWM held-out L1 |
|---:|---:|---:|---:|
| 5,000 | 0.277710 | 0.106041 | 0.063823 |
| 10,000 | 0.114974 | 0.066889 | 0.057281 |
| 15,000 | 0.083181 | 0.055216 | 0.055037 |
| 20,000 | 0.068786 | 0.045309 | 0.052901 |

From 15k to 20k, LeFlow's held-out combined objective changed by
-17.56% and HWM's L1 changed by
-3.88%.
These objectives differ and should not be compared numerically across methods.
LeFlow's adapter objective also fell from 0.074281 at update 1,000 to 0.037185
at its allocated 2,000-update endpoint, so adapter convergence is not established.
The spatial adapter's unreleased training schedule remains a disclosed port
choice; this result must not be presented as the authors' fully trained model.
Neither reaching a common update cap nor falling prediction loss proves all
methods are converged in task success. The current runs establish a fixed-budget
reference, not a matched-convergence result. Any later extension of the unchanged
training recipe should use a prospective common allowance/rule and resume these
optimizer states. Changing the adapter recipe would be a distinct baseline
variant; no extension or variant is queued.

The 104 repeatedly used cases are sufficient to expose large development failures.
They are insufficient for an independent publication superiority claim. Use the
sealed final cases after method selection, preserve paired reset identities,
report per-task uncertainty, and state the single-training-seed limitation.

## Per-task selected-checkpoint outcomes

Each entry is a success count out of eight cases.

| Task | Our preserved model | LeFlow | HWM paper port | Released CEM |
|---|---:|---:|---:|---:|
| assembly | 3 | 0 | 0 | 0 |
| button-press-topdown | 8 | 0 | 0 | 0 |
| coffee-button | 8 | 6 | 0 | 8 |
| dial-turn | 7 | 0 | 0 | 0 |
| door-close | 8 | 0 | 0 | 3 |
| door-open | 6 | 0 | 0 | 0 |
| drawer-close | 8 | 1 | 1 | 3 |
| drawer-open | 8 | 0 | 0 | 0 |
| faucet-open | 3 | 4 | 0 | 0 |
| handle-press | 8 | 7 | 4 | 8 |
| pick-place | 1 | 0 | 0 | 0 |
| plate-slide | 8 | 0 | 0 | 0 |
| reach | 4 | 3 | 0 | 2 |

## Compute and runtime recovery

All values are aggregate GPU-hours. Validation includes held-out-loss diagnostics.
Worker occupancy is elapsed worker time multiplied by its four reserved GPUs;
it is not a measure of utilized GPU arithmetic or a cloud bill.

| Method | Optimization | Completed validation | Other worker occupancy | Total worker occupancy |
|---|---:|---:|---:|---:|
| leflow | 0.8524 | 2.4360 | 2.7427 | 6.0312 |
| hwm | 2.0773 | 1.5413 | 0.8728 | 4.4915 |
| cem | 0.0000 | 0.5147 | 0.3061 | 0.8207 |

Our preserved run used 3.2334 optimizer GPU-hours. The shared world/cache are reused
and are not charged again. Discarded-weight GPU preflights are separately recorded
in the [source audit](RELEASE_BASELINES.md).

At the second evaluation, shared-filesystem lock contention caused two ranks to
enter cleanup while peers waited. The runtime fix retries the intended blocking
journal lock and reports errors before collective teardown. Both methods resumed
their exact 10k checkpoints, optimizer/RNG state and 52 completed case records.
**No optimizer updates were repeated.** Interrupted evaluation, startup, saving
and cleanup stalls are included in other worker occupancy above; that overhead
must not be mistaken for extra learning or silently omitted.

All 20 targeted tests passed. The server filesystem test completed 100 concurrent
lock acquisitions with no lost updates. The method sources remain frozen; the
separate infrastructure guard has SHA256
`3ed0ada4a8b923667cdf13af3280ba0e9451c044db97d497fdf3113facd13f94`. Every final registry file hash was rechecked, and each
selected model's tensors exactly match its selected milestone checkpoint.

## Fixed references and reuse

[Machine-readable fixed-reference index](reports/20261010-release-baselines/fixed-references.json)
contains exact checkpoint paths, hashes, source revisions and case-set identity.

- **leflow:** selected update 10,000; source `564b52ffc86eba212aa6858f3a9f10390905a905`. [Registry](reports/20261010-release-baselines/leflow/registry.json), [verified audit](reports/20261010-release-baselines/leflow/audit.json), [finished online run](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/own982ph).
- **hwm:** selected update 15,000; source `cde9c394c00e5be68348d6e80c1fd22d0e5c518d`. [Registry](reports/20261010-release-baselines/hwm/registry.json), [verified audit](reports/20261010-release-baselines/hwm/audit.json), [finished online run](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/8gpcqqc2).
- **cem:** selected fixed world / no head; source `564b52ffc86eba212aa6858f3a9f10390905a905`. [Registry](reports/20261010-release-baselines/cem/registry.json), [verified audit](reports/20261010-release-baselines/cem/audit.json), [finished online run](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/tupg6zpf).

[Final independent audit](reports/20261010-release-baselines/final-audit.json)
records matched case IDs, episode hashes, reset/model seeds, selection, complete
learning curves, actual compute and online readback. Weights remain on the
authorized server under
`/home/mtxu/adam/LeFlow-experiments/20261010-baseline-release/{method}_3072`.
Use these references for future iterations under the unchanged benchmark.
No new seed, baseline restart, ablation, our-method retraining or final test is
queued. A changed benchmark or controller allowance needs new evaluation, not an
automatic fresh training run.
