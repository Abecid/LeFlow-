# LeFlow BTM-JEPA training: experiment status and results

**Latest check: October 7, 14:57 UTC — campaign and GPU access blocked.**
Both campaign-record reads and both GPU queries failed with SSH banner-exchange
timeouts. W&B succeeded on retry at 14:56:36–37 UTC and still reports `finished`
for both runs, with unchanged steps and runtimes. Current campaign files,
checkpoints, completion markers and GPU state are unverified. The last full
collection at 14:40 UTC found BTM 23,830 saved steps, flow 23,000 saved /
23,040 logged, and both completion markers absent. Recovery/completion remain
unverified and there is no active ETA. See the
[access-blocker report](reports/20261007T145701Z.md); the
[previous monitoring recovery](reports/20261007T135548Z.md) is preserved.

**Verified October 6, 2026, 1:09–1:11 AM EDT (05:09–05:11 UTC).**

This report covers the actual runs found in this repository's server deployment:
W&B project **`attentionx2023/btm-jepa`**, BTM run **`z78m0zk6`**, flow baseline
**`g0u3w723`**. A broader artifact inventory at 1:18 AM EDT also found the earlier
two-update verification run. These runs train the BTM/flow planners and inverse
heads over a frozen JEPA/LeWM model. No separate `train.py` encoder/predictor
training artifacts were found in the inspected deployment and configured/default
cache locations. The [run inventory](snapshots/20261006T050931Z/run-inventory.json)
records this scope; it is not a claim about every directory on the server.

**As of the last successful full check at 14:40 UTC on October 7, the full experiment suite had not been run and the paired pilot was incomplete.** BTM reached its planned training and validation endpoint. Flow stopped short. The available single-seed validation results **do not establish that BTM improves long-horizon success or preserves success with a controlled speedup**. Through the last successful October 7, 14:40 UTC full collection, no new training/evaluation results had been observed since the October 5 report. Current campaign records could not be verified at the 14:55 check; only online status was freshly read.

## What ran

Campaign status in this table was last verified at 14:40 UTC on October 7 and is cached during the current access failure.

| Experiment | Status |
|---|---|
| Earlier BTM verification run | 2 updates and three offset evaluations; a wiring check, excluded from research scores |
| PushT + frozen LeWM, BTM, seed 3072 | 23,830/23,830 saved updates; 33/33 validation cycles |
| Matched 16-step flow baseline, seed 3072 | 23,000 saved / 23,040 logged updates; 32/33 cycles |
| Deterministic generator, 2/4/8-step flow, flat-controller controls | No research results in this campaign |
| Same-GPU controlled timing / fixed planning deadline | Not run |
| Held-out test and additional training seeds | Not run |
| Exact published LeFlow reproduction; V-JEPA 2.1 / Meta-World | Not performed; the latter is not implemented |

Only **one trained baseline** is available. The broader controls have no results in the inspected campaign. At the last successful full collection, October 7 at 14:40 UTC, no known campaign trainer or flow controller was visible through the collector's name/path filters and accessible process view; the original supervisor was intentionally stopped. Both completion markers were absent. BTM's final checkpoint was valid at epoch 10, next batch 0, and its exited trainer had exit code zero. Flow needed **830 updates from its recoverable checkpoint**, including replay of 40 logged but unsaved updates, plus final validation. These campaign observations are cached. Fresh W&B reads at 14:56:36–37 UTC still report `finished` with unchanged steps and runtimes; that does not certify the planned budget. Current campaign and GPU reads are blocked after two SSH banner-exchange timeouts each. At the cached 14:40:42 UTC GPU query, four unidentified processes occupied GPUs 4–7, using 25,299 MiB on 4/5 and 30,453 MiB on 6/7; GPUs 0–3 were idle. Ownership and campaign association were unresolved. Current training recovery, completion and GPU health are unverified; the access and training-interruption causes remain unknown. **There is no active completion ETA.** No GPU jobs were changed.

## How good is BTM relative to flow?

Each offset uses 20 fixed validation cases and a 200-action budget. Both `best.pt` checkpoints were selected at update 1,000 by **mean success across offsets**, retaining the earliest tie.

| Comparison | Method | Update | Success @25 | @50 | @100 (primary) | Mean |
|---|---|---:|---:|---:|---:|---:|
| Validation-selected | BTM | 1,000 | 40% | 40% | 25% | 35.00% |
| Validation-selected | Flow | 1,000 | 50% | 30% | 20% | 33.33% |
| Latest matched | BTM | 23,000 | 5% | 0% | 10% | 5.00% |
| Latest matched | Flow | 23,000 | 45% | 15% | 5% | 21.67% |
| Unmatched final | BTM | 23,830 | 35% | 20% | 5% | 20.00% |

At selected checkpoints, the primary difference is **+5 percentage points**, paired bootstrap 95% interval **[−15, +25]**. At 23k it is **+5 points [−10, +20]**. Neither establishes superiority or equivalence. BTM's shorter-horizon results at 23k are worse: −40 points at offset 25 and −15 at offset 50. Its final checkpoint has no equally trained flow result.

Across all 32 matched checkpoints, descriptive average primary success is **8.125% BTM versus 10.469% flow**; BTM wins/ties/loses at 12/4/16 checkpoints. These repeated validation measurements are not independent tests. Both methods reach a maximum observed primary success of 25%, at different checkpoints. The 3,900 recorded rollout attempts reuse **60 cases from 57 episodes**, with only one training seed pair.

## What the diagnostics show

- **Improving latent errors do not produce dependable control.** From 1k to 23k, BTM path MSE falls 1.0003→0.1240 and flow 0.5236→0.1063. Both lose every offset-100 case they solved at 1k; later successes occur on different cases.
- **Physical failure mechanisms remain unmeasured.** Predicted and observed latent errors improve, but use different horizons and actions. We do not yet have aligned post-CEM execution traces, physical subgoal reach rates, or failure videos to distinguish bad proposals, inverse-control failure, or world-model error.
- **BTM uses fewer generator evaluations; the efficiency goal is unproven.** BTM uses 1 generator call versus 16. At matched 23k/offset 100, mean solver-batch time is 1.132 versus 1.561 seconds; p95 is 1.146 versus 1.589 seconds. This is descriptive timing on different GPUs/load, not controlled end-to-end latency with retained success. Both perform 665,600 world-model state predictions for that evaluation.
- **No logged numerical divergence explains the outcome.** All 4,848 metric rows and 195 evaluation records pass the finite-value/outcome audit. Candidate variance contracts in both methods; this alone cannot establish behavioral mode collapse. Raw generative losses optimize different objectives and are not cross-method scores.

## Targets and next decisions

The primary target remains repeatable higher offset-100 task success on matched data and budgets; the efficiency target is lower full-controller latency while retaining success. **Neither is demonstrated.** Assigned GPUs and optimizer updates match by design, but actual final exposure differs and equal FLOPs/GPU-hours were never enforced.

Priorities are: recover the frozen flow pilot if separately authorized; record and classify persistent failures; align prediction/execution measurements; test ground-truth waypoints and a shared inverse head; then evaluate horizon alignment, deterministic/few-step controls, and isolated controller ablations. Freeze the resulting method before held-out testing and multi-seed confirmation. Keep flow unchanged and the test split unused during development.

The [detailed October 6 report](reports/20261006T050931Z.md) contains the full metric assessment, checkpoint identities, failure hypotheses and discriminating tests. New [compact numerical evidence and fresh health checks](snapshots/20261006T050931Z/assessment.json) accompany [selected/matched/final rollout metrics](snapshots/20261006T050931Z/selected_matched_and_final_metrics.csv), [training diagnostics](snapshots/20261006T050931Z/selected_matched_and_final_training_metrics.csv), and [all paired success comparisons](snapshots/20261006T050931Z/paired_success.csv). The unchanged full histories and chart remain in the [October 5 evidence](snapshots/20261005T175219Z/summary.json) and [archived report](reports/20261005T175219Z.md). [Monitoring protocol](MONITORING.md).

W&B: [BTM](https://wandb.ai/attentionx2023/btm-jepa/runs/z78m0zk6), [flow](https://wandb.ai/attentionx2023/btm-jepa/runs/g0u3w723), [campaign group](https://wandb.ai/attentionx2023/btm-jepa/groups/pusht_pair_20261004).
