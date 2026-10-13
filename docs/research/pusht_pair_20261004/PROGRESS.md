# LeFlow BTM-JEPA training: experiment status and results

**Latest check: October 13, 14:00 UTC — all remote monitoring reads blocked again.**
Campaign, W&B/CPU-probe and GPU-process reads each failed twice during SSH
banner exchange. No current remote state was verified. The last campaign check
at 13:45 UTC found BTM saved at 23,830 updates and flow saved at 23,000 / logged
at 23,040, with both completion markers absent. Those records and the two idle
GPU samples are now cached; current recovery/completion and hardware are unknown.
There is no verified active ETA. See the [latest access-failure report](reports/20261013T140032Z.md)
and preserved [previous partial-recovery report](reports/20261013T134523Z.md).

**Verified October 6, 2026, 1:09–1:11 AM EDT (05:09–05:11 UTC).**

This report covers the actual runs found in this repository's server deployment:
W&B project **`attentionx2023/btm-jepa`**, BTM run **`z78m0zk6`**, flow baseline
**`g0u3w723`**. A broader artifact inventory at 1:18 AM EDT also found the earlier
two-update verification run. These runs train the BTM/flow planners and inverse
heads over a frozen JEPA/LeWM model. No separate `train.py` encoder/predictor
training artifacts were found in the inspected deployment and configured/default
cache locations. The [run inventory](snapshots/20261006T050931Z/run-inventory.json)
records this scope; it is not a claim about every directory on the server.

**The full experiment suite has not been run, and the paired pilot is incomplete.** BTM reached its planned training and validation endpoint. Flow stopped short. The available single-seed validation results **do not establish that BTM improves long-horizon success or preserves success with a controlled speedup**. Through the October 13, 13:45:23–13:45:26 UTC full collection, no new training/evaluation results had been observed since the October 5 report. All remote observations below are cached because campaign, W&B/CPU-probe and separate GPU-process reads failed twice in the current check. The last successful online observation remains 12:59 UTC. Current progress and recovery are unverified; earlier idle GPU samples do not establish continuing inactivity.

## What ran

Campaign status in this table was last verified at 13:45:26 UTC on October 13 and is now cached.

| Experiment | Status |
|---|---|
| Earlier BTM verification run | 2 updates and three offset evaluations; a wiring check, excluded from research scores |
| PushT + frozen LeWM, BTM, seed 3072 | 23,830/23,830 saved updates; 33/33 validation cycles |
| Matched 16-step flow baseline, seed 3072 | 23,000 saved / 23,040 logged updates; 32/33 cycles |
| Deterministic generator, 2/4/8-step flow, flat-controller controls | No research results in this campaign |
| Same-GPU controlled timing / fixed planning deadline | Not run |
| Held-out test and additional training seeds | Not run |
| Exact published LeFlow reproduction; V-JEPA 2.1 / Meta-World | Not performed; the latter is not implemented |

Only **one trained baseline** is available. The broader controls have no results in the inspected campaign. The October 13, 13:45:23–13:45:26 UTC full collection found no known campaign trainer or flow controller through the collector's name/path filters and accessible process view; the original supervisor remains intentionally stopped. At that cached check, both completion markers were absent. BTM's final checkpoint metadata was valid at epoch 10, next batch 0, with recorded trainer exit zero. At the last successful check, flow needed **830 updates from its recoverable checkpoint**, including replay of 40 logged but unsaved updates, plus final validation. Cached W&B checks at 12:59:58/59 UTC reported both runs `finished` at steps 23,830/23,040 and runtimes 8,304/9,240 seconds. The cached CPU probe at 12:59:40 UTC was unchanged. The cached campaign records preserve the same manifest; current campaign and online states remain unverified.

The cached 13:45:20 UTC GPU query showed all eight devices at 0 MiB reported memory and 0% utilization, with no driver compute-process entries. The collector reported the same memory/utilization at 13:45:26 UTC, with power 55.65/53.42/56.46/54.83/58.13/56.97/54.29/56.31 W. Those two samples do not establish current or sustained inactivity. Both current GPU-read attempts failed, as did the preceding detailed health follow-up. Current memory, utilization, process counts/identities, temperatures and hardware-error diagnostics are unverified. Prior device activity has unresolved ownership. **There is no verified active completion ETA.** Monitoring changed no GPU jobs.

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
