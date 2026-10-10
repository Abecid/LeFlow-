# Corrected baseline results and remaining work

The corrected **released CEM solver scores 24/104 (23.08%)** on the same development
cases where the preserved execution-revision checkpoint scores 80/104 (76.92%).
LeFlow is training; HWM's paper-based port is finishing preflight before launch.
Do not substitute historical simplified LeFlow/HWM results into this comparison.

This is a shared-backbone, fixed-budget MetaWorld comparison, not a reproduction
of the authors' original benchmark results. Read the
[source, adaptation and budget audit](RELEASE_BASELINES.md).

## Completed CEM baseline

- Frozen source: `564b52ffc86eba212aa6858f3a9f10390905a905`.
- [Exact case records and metrics](reports/20261010-release-baselines/cem/validation.json).
- [Checkpoint/report registry](reports/20261010-release-baselines/cem/registry.json).
- [Independent comparison audit](reports/20261010-release-baselines/cem-audit.json).
- [Finished online run](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/tupg6zpf).

The local report's SHA256 matches the sealed server registry. All 104 case IDs,
episode-file hashes, reset seeds and model seeds match our preserved evaluation.
There are 23 successes shared by both methods, 57 successes unique to our method,
one unique to CEM, and 23 failures shared by both. These are development outcomes,
not an independent final-test significance claim.

| Task | Our preserved 5k checkpoint | Released CEM |
|---|---:|---:|
| assembly | 3/8 | 0/8 |
| button-press-topdown | 8/8 | 0/8 |
| coffee-button | 8/8 | 8/8 |
| dial-turn | 7/8 | 0/8 |
| door-close | 8/8 | 3/8 |
| door-open | 6/8 | 0/8 |
| drawer-close | 8/8 | 3/8 |
| drawer-open | 8/8 | 0/8 |
| faucet-open | 3/8 | 0/8 |
| handle-press | 8/8 | 8/8 |
| pick-place | 1/8 | 0/8 |
| plate-slide | 8/8 | 0/8 |
| reach | 4/8 | 2/8 |
| **Total** | **80/104** | **24/104** |

**All 80 CEM failures exhausted the 10-second controller allowance.** This is a
resource-limit result; it does not establish that those cases would fail with
more planning time. CEM uses its published 300-sample/30-iteration solver and
executes five two-action blocks per plan. A warm plan takes approximately one
second, limiting how many plans can finish within the shared allowance.
Validation consumed 0.514655 aggregate GPUh; no extra model training was needed.
The common previously trained world model is reused and is not charged again.

## Convergence and benchmark sufficiency

The 104 cases are useful development checks. Repeated method design and checkpoint
selection on them prevent treating them as the final publication benchmark.
The 3200 reserved final cases, including three held-out tasks, remain untouched.
The primary metric is task success under the joint 200-action/10-second allowance;
per-task results, controller timeouts, latency and steps-to-success explain it.

The current allowance measures bounded-compute performance. HWM's headline planner
takes approximately 15.68 seconds for one decision, so it cannot act under this
allowance. Its registered port therefore uses only the smaller settings already
listed in its paper's compute study. A broad claim about method quality would
also need a common, larger evaluation allowance. Saved checkpoints can support
that later comparison without retraining.

Do not claim that all methods have converged. Training and held-out losses are
logged separately from periodic task success. If a method is still improving at
its update/compute cap, call it budget-limited. Any later training extension should
apply a prospectively chosen common allowance/rule, rather than selectively
spending more on our model. This run does not add such an extension.
