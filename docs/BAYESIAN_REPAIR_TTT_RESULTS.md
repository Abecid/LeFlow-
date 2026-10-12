# Bayesian repair: completed development result

The selected **15k checkpoint scores 77/104 (74.04%)**, below the retained
80/104 references. The complete curve is **71 / 76 / 77 / 77**; the registered
earliest-tie rule selects 15k. This ALPaCA-inspired posterior repair adaptation
did not establish a development improvement over retrieval-repair or execution
revision. There are **115 failures across 416 episodes**: 112 reach the 200-action
limit and three exhaust the cumulative 10-second controller allowance. No case
or optimizer update was replayed, and final tests remain sealed.

One fresh seed 3072 run completed 20,000 updates with the same 6,222 expert TRAIN
episodes, global batch 64 and eight GPUs. Frozen source is
`4d322c90e216702fae2cd29c91e49ad90da713d9`. The
[final CPU audit](reports/20261012-bayesian-repair-ttt/analysis/final-audit.json)
passes all 173 runtime files, source/config/data/world/bank identities, all milestone
checkpoints, 1,200 periodic records, four fixed evaluations and posterior/controller
accounting. At 2026-10-12T09:29:20.964544+00:00, the
[final readback](reports/20261012-bayesian-repair-ttt/final-health.json) verified all 44
recorded owned processes absent and [W&B u1hdmwqr](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/u1hdmwqr)
finished. The single-candidate authorization is complete; further launches stop
for review. No next candidate is authorized by the monitor.

## Fixed curve, timeouts and paired cases

| Updates | Success | Timeouts | Failures reaching 200 actions | Decision latency mean / p95 ms |
| --- | ---: | ---: | ---: | ---: |
| 5,000 | 71/104 | 0 | 33 | 88.59 / 92.73 |
| 10,000 | 76/104 | 2 | 26 | 89.26 / 92.90 |
| 15,000 | 77/104 | 0 | 27 | 88.63 / 92.52 |
| 20,000 | 77/104 | 1 | 26 | 88.51 / 92.77 |

The three timeouts are assembly 00002 at 10k (194 actions), assembly 00006 at 10k
(192), and assembly 00003 at 20k (192). Their traces contain isolated calls of
1.611, 1.561 and 1.430 seconds; recorded controller overruns are respectively
0.000389, 0.034504 and 0.000480 seconds. Ordinary calls in those traces are much
shorter. This does not identify the cause of the spikes or show that those
cases would have succeeded without them. The unchanged 95 ms **mean** startup
gate passed at 81.12 ms; it does not bound every later call. Every timeout and its
full latency sequence are preserved, with a [compact timeout index](reports/20261012-bayesian-repair-ttt/analysis/timeouts.json).

| Task (eight cases each) | 5k | 10k | 15k | 20k |
| --- | ---: | ---: | ---: | ---: |
| assembly | 1 | 1 | 2 | 4 |
| button-press-topdown | 8 | 8 | 8 | 8 |
| coffee-button | 8 | 8 | 8 | 8 |
| dial-turn | 7 | 8 | 8 | 8 |
| door-close | 7 | 8 | 8 | 8 |
| door-open | 6 | 7 | 8 | 6 |
| drawer-close | 8 | 8 | 8 | 8 |
| drawer-open | 8 | 8 | 8 | 8 |
| faucet-open | 1 | 1 | 2 | 1 |
| handle-press | 8 | 8 | 8 | 8 |
| pick-place | 0 | 0 | 0 | 1 |
| plate-slide | 8 | 8 | 8 | 8 |
| reach | 1 | 3 | 1 | 1 |

The following comparisons use selected 15k 77 and identical episode hashes,
case IDs, reset seeds and model seeds. Full paired IDs are in the
[saved-record analysis](reports/20261012-bayesian-repair-ttt/analysis/final-details.json).

| Preserved reference | Reference successes | Wins | Regressions |
| --- | ---: | ---: | ---: |
| guided_ttt | 75 | 6 | 4 |
| progress_ttt | 78 | 4 | 5 |
| flow_reasoning | 79 | 5 | 7 |
| execution_revision | 80 | 6 | 9 |
| controller_grounded | 79 | 7 | 9 |
| leflow_release | 21 | 62 | 6 |
| hwm_paper_port | 5 | 72 | 0 |
| cem_release | 24 | 54 | 1 |
| repair_flow_ttt | 80 | 5 | 8 |
| retrieval_repair_ttt | 80 | 2 | 5 |
| shifted_repair_ttt | 79 | 5 | 7 |
| progress_objective_ttt | 78 | 3 | 4 |

Against retrieval-repair 80, selected 15k has **2 wins / 5 regressions**: gains assembly 3
and dial-turn 7; losses assembly 1/5/6, faucet 0 and pick-place 4. Against execution-revision 80,
there are **6 wins / 9 regressions**: gains assembly 0/3, dial-turn 6, door-open 0/1
and faucet 2; losses assembly 5/6/7, faucet 0/4, pick-place 4 and reach 0/1/2.
These shorthand numbers denote the corresponding zero-padded validation IDs.
Twenty-k ties the total score but changes 5 successes and
5 failures relative to 15k; those IDs remain in the analysis.
The retained 80 reference is unchanged. Custom baseline ports and matched budget
ceilings are not native-author benchmark, matched-convergence or SOTA claims.

## Posterior support and physical success

The [method document](BAYESIAN_REPAIR_TTT.md) distinguishes the adaptation of
[ALPaCA](https://arxiv.org/abs/1807.08912) and its pinned author code from a faithful
reproduction or novelty claim. Each executed transition supplies one likelihood
observation; spatial patches are pooled before regression. Candidate/incumbent
comparisons share posterior weight uncertainty and pass the deployed bounded
positive prefix penalty. A repair must exceed 1.644854 standard deviations of
the paired cost difference; the incumbent remains admissible. Uncertainty covers
prefix-error conditional mean, not terminal bias or physical task success.
Generated actions never inherit recorded expert outcomes. Fresh training uses
depths 1/2/3, fixed evaluation depth 2; no inference-depth sweep was performed.

| Updates | Raw observed-prefix error MAE | Applied-correction MAE | Nominal90% predictive coverage | Executed plan is repair |
| --- | ---: | ---: | ---: | ---: |
| 5,000 | 0.002648 | 0.001775 | 85.65% | 42.52% |
| 10,000 | 0.002748 | 0.001846 | 84.28% | 40.05% |
| 15,000 | 0.002729 | 0.001850 | 83.45% | 35.92% |
| 20,000 | 0.002651 | 0.001827 | 84.29% | 35.12% |

At selected 15k, the deployed correction reduces the encoded prediction-error
proxy from 0.002729 to0.001850, while predictive coverage is 83.45%, below nominal 90%.
The unbounded posterior mean MAE is 0.001852. This is evidence of better prediction
than the uncorrected proxy on visited states, not a physical-success improvement.
The TRAIN-only diagnostic's 90.79% coverage did not carry over unchanged to these
policy trajectories. Across checkpoints the states visited differ; neither lower
NLL nor lower MAE isolates a causal controller benefit.

| Updates | Accepted repairs / all candidate-route decisions | Positive-mean-gain repairs rejected by margin | Anchors: generated / retrieved / shifted |
| --- | ---: | ---: | ---: |
| 5,000 | 21087/46040 | 7882/41029 | 1563 / 1763 / 2429 |
| 10,000 | 17645/42544 | 6137/36646 | 1426 / 1578 / 2314 |
| 15,000 | 17067/43368 | 7303/38095 | 1578 / 1639 / 2204 |
| 20,000 | 16682/42184 | 7551/38724 | 1517 / 1511 / 2245 |

The route-level gate and final executed-plan choice are different counts. All
logged costs, shared margins, eligible sets, selected incumbents, action bounds,
causal histories and episode resets pass the method-specific audit. Every full
decision uses 280 predicted world transitions and 40 cached incumbent transitions;
the eight-target/four-candidate pool is unchanged. This saves duplicate rollout
work relative to 320-call predecessors, rather than enlarging the controller budget.
Rejected alternatives have no measured outcomes; acceptance frequencies cannot
show which repairs physically helped or harmed.

Pick-place scores 0/0/0/1. At5k/10k all failed pick-place last 20-decision tails use
only reach/direct-goal provenance. At selected 15k seven do, but failed case3
retains pick-place routes throughout its last 20 decisions. That case succeeds
at 20k. Matching provenance alone is therefore not a sufficient failure diagnosis,
and these records do not authorize oracle task-ID filtering. Final20k improves
assembly 2→4 and pick-place 0→1 versus15k, but door-open falls 8→6 and
faucet 2→1. Reach remains 1/8 at the selected and final checkpoints.
Selected 15k has 20 failures with a positive raw predicted tail advance and no
positive observed proxy advance, or11 after correction; these are model-space
stall signs, not direct measurements of physical contact or task progress.

## Training diagnostics and saved visual evidence

| Last500-update window, mixed depths 1/2/3 | Action-flow loss | Residual-flow loss | Teacher coordinates clipped | Predictive NLL |
| --- | ---: | ---: | ---: | ---: |
| 5,000 | 0.162581 | 0.261004 | 4.50% | -1.855844 |
| 10,000 | 0.116172 | 0.181118 | 2.71% | -1.835614 |
| 15,000 | 0.086502 | 0.164680 | 2.39% | -1.916486 |
| 20,000 | 0.077501 | 0.140781 | 2.24% | -1.841985 |

Depth-specific records are retained in the analysis. The mixed-depth clipping
fraction falls 4.50%→2.24%, and both action losses decrease, while success stalls
at 77. Negative NLL is valid for a continuous density. Teacher targets contain
recorded expert actions; their lower action error is not a physical outcome label
for generated candidates. Proposal-diversity comparisons use matched initial
cases, empty histories and verified matching frozen route costs; their numerical
results are preserved in `initial_proposal_diversity` in the final analysis.
The first-pool mean spread falls 0.061789→0.023997 (61.16%) from 5k to 20k;
the residual-pool spread falls 0.026461→0.013416 (49.30%). All 104 initial cases
contract in both pools by 15k and 20k. Contraction also accompanies the improvement
from 71 to 77, so it does not by itself explain the remaining performance gap.
Neither contraction nor clipping alone establishes a failure cause.

All **40 exact saved contact sheets**, ten per checkpoint, were rendered from
recorded frames and visually inspected at readable resolution. Pick-place00000
leaves the red object on the table throughout; faucet 00003 approaches the handle
without success; reach 00003 remains near its target without registered success.
Assembly00000 succeeds at 5k, fails at 10k, then succeeds at 15k/20k (192/90 actions).
Reach00005 fails at 5k then succeeds at 10k/15k/20k (46/66/54 actions). The saved
subset does not cover every failure. Sparse frames cannot establish contact forces,
slippage or the outcome of rejected actions. Every sheet, original trajectory
reference, reset-frame identity check and per-image review is retained under
`reports/20261012-bayesian-repair-ttt/contact-sheets-STEP/` and `visual-review-STEP.json`.

## Compute and preservation

Optimization consumed **4.147635455 aggregate GPUh**
(14931.487637 GPU-seconds); fixed evaluations consumed
**2.109850832 GPUh**. The 20k update limit bound first;
optimizer overrun is 0 seconds, below the 28,800 aggregate GPU-second ceiling.
The separate TRAIN diagnostic/preflight/no-update DDP/retrieval setup charges are
0.001555405 / 0.005896763 / 0.010083863 / 0.005356530 GPUh. All new learned modules
and per-update retrieval are inside the optimization timer.

Prior measured components total 105.28452734094218 GPUh. Adding this run and its
separately recorded startup work gives **111.56490618890 GPUh**.
This is a cumulative ledger of known measured components, not a full cluster bill.
CPU reporting, transfers, idle reservations and unmeasured infrastructure are not
assigned invented GPU charges. The bank-remapping wall time is recorded separately.
See the [final compute ledger](reports/20261012-bayesian-repair-ttt/compute-accounting-final.json).

All 1,200 periodic metrics and all four complete evaluations remain lossless.
Each roughly0.9–1.0 GB JSON report is gzip-compressed and split into32 MiB parts;
the manifest carries per-part, combined-stream and original JSON SHA256 hashes.
`scripts.analysis.result_io.read_record` reconstructs and verifies it before parsing.
No candidate trace or negative result is removed. Checkpoints remain at
`/home/mtxu/adam/LeFlow-experiments/20261012-bayesian-repair-ttt/campaign/runs/bayesian_repair_ttt_3072/`,
with milestone, selected and final hashes in the final audit. Raw datasets,
model binaries and credentials are excluded from Git.

The first-two-round publication was remotely verified at c5421c2 after an HTTP400
push failure was recovered with a non-force HTTP/1.1 buffered retry. The saved
publication receipt and reporting-operation notes preserve transport failures;
none caused training/evaluation replay or frozen-source edits. Complete publication
requires a successful final push and live verification of the ledger artifact blobs;
a local commit alone is not proof. The final review retains80/104, holds further
launches and leaves monitoring active for later explicit authorization.

One seed, repeatedly reused development selection, changed training depth and
eight-rank random-stream assignment do not establish generalization, a component
causal benefit, inference scaling, novelty or publication readiness.
