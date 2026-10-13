# First two curves: vanilla flow77/84, Diffusion Policy43/69

At the latest preserved readback, vanilla conditional action flow has completed
5k/10k evaluations scoring77/84 out of104; Diffusion Policy scores43/69. All
four evaluations have zero timeouts. Training continues under the original
finite budgets. The vanilla10k score exceeds the preserved80 development
reference by four cases; this is an interim single-seed development result,
not independent proof of superiority. The simple baseline changes which
comparator our proposed mechanism needs to beat. All paired case identities
and outcomes are preserved in the per-method analysis files.

Historical launch snapshots follow.

# Five authorized additional baseline runs — October 12, 2026

The user requested training and evaluation of all five implemented methods.
These are fresh, single-seed comparisons; preserved GMM80, retrieval80 and
LeFlow21 references are reused. No prior baseline is retrained.

| Method | Queue lane / physical GPUs | State |
| --- | --- | --- |
| Vanilla conditional action flow | A: 0–3, first | Training verified;5k evaluation pending |
| Action flow + fixed world search | A: 0–3, second | Queued; source and CPU tests verified |
| SAGE-inspired MetaWorld port | A: 0–3, third | Queued; source and CPU tests verified |
| Diffusion Policy port | B: 4–7, first | Training verified;5k evaluation pending |
| Feedback World Model-inspired port | B: 4–7, second | Queued; source and CPU tests verified |

All eight A800 80GiB GPUs were verified idle before allocation. Scratch has about
1.9TiB free; persistent storage has about305GiB free. Frozen execution source is
`0a19c7c67d791fd99376216399b03e8cb71d78c0` with214 checked runtime files.
Each method has its own frozen checkout, optimizer/checkpoints, W&B identity,
full periodic training metrics and physical evaluation records. Two four-GPU
lanes overlap independent runs; within each lane runs are sequential.

Every run uses the same6222 successful expert TRAIN episodes, original frozen
V-JEPA encoder/shared world, seed3072, global64, 20k updates OR28800 aggregate
optimizer GPU-seconds, fixed104 development cases at5k/10k/15k/20k, and200 actions/
10 cumulative controller seconds. Actual consumed compute is reported separately.
Final tests remain disabled. SAGE/FWM are explicitly labeled adaptations.

Before optimization, the real GPU preflight checks exact consumed TRAIN fields
against the established sampler and exercises Feedback-WM with a factual TRAIN
prefix. All five datasets omit unused execution-history tensors without changing
sampled states/actions. The no-update four-rank check verifies finite, exactly
synchronized policy gradients and no gradients/changes in the frozen world.
The dispatcher also handles a final development checkpoint if a new baseline
reaches its optimizer cap between scheduled milestones.

[Method specifications and upstream differences](ADDITIONAL_BASELINES_20261012.md)
remain authoritative. The search control has480 world transitions per decision,
versus320 for retrieval repair; this is a common-time comparison, not equal world
calls or a pure no-workspace ablation. Diffusion/FWM share the same architecture
and DDPM objective but are registered here as separate fresh runs; checkpoint
identity will be examined before interpreting the guidance comparison.

Server batch record: `/home/mtxu/adam/LeFlow-experiments/20261012-additional-baselines`.
Queue execution, source hashes, preflight/DDP reports, process identities and W&B
links will be added after launch verification. All results and failure findings
must be published to origin/main as each run completes. Registration alone is
not evidence of successful training or completed evaluation.

## Verified first two launches

At the03:00UTC snapshot (October12 20:00Pacific), vanilla flow and Diffusion Policy
have both reached5,000 updates, with finite metrics and their first evaluation
in progress. Their exact consumed TRAIN fields match eight original examples.
All four ranks have synchronized finite gradients, no unused policy parameters,
unchanged preflight weights and no world gradients. Measured full-controller
preflight means are51.66ms and76.39ms respectively. The other three methods
remain in the persistent queue; their GPU preflight results are not yet claimed.

- [Vanilla flow live metrics](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/j1m6jzog).
- [Diffusion Policy live metrics](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/cnlxvbpf).
- [Source-verified snapshot](reports/20261012-additional-baselines/live-snapshot.json).
- [Exact launch/preflight/DDP and metric archive](reports/20261012-additional-baselines/).

All13 CPU tests passed in the established runtime. First5k optimization consumption
is0.15778/0.15836GPUh; the8GPUh ceilings remain unchanged. Updated snapshots are
sequential readbacks rather than an atomic global training checkpoint. At the
training readback all eight GPUs had owned workers; disjoint device groups avoid
sharing the physical GPU used for per-case controller timing. No task-success
result is inferred from falling training loss.
