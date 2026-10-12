# Nonlinear episodic repair: completed development result

October 12, 2026. The fixed curve is **73/66/67/72**, selecting **5k 73/104**, below the preserved retrieval-repair and execution-revision **80/104** references. This authorized single run is complete. No next candidate is authorized; further launches are held for review.

All 800 periodic training records, four full 104-case evaluations, 4 controller timeouts and 40 individually inspected saved contact sheets are preserved. The final CPU audit verifies all 200 runtime hashes, source/config/data/bank/checkpoint identities and causal nonlinear-memory/controller accounting. All 44 recorded owned processes are absent and W&B `mxujto8j` is finished at 2026-10-12T10:48:21.611824+00:00.

## Fixed results and paired failures

| Updates | Success /104 | Timeouts | Other failures at 200 actions |
| --- | ---: | ---: | ---: |
| 5,000 | 73 | 0 | 31 |
| 10,000 | 66 | 4 | 34 |
| 15,000 | 67 | 0 | 37 |
| 20,000 | 72 | 0 | 32 |

| Task (eight cases) |5k|10k|15k|20k|
| --- | ---: | ---: | ---: | ---: |
| assembly | 0 | 0 | 0 | 0 |
| button-press-topdown | 8 | 7 | 8 | 8 |
| coffee-button | 8 | 8 | 8 | 8 |
| dial-turn | 3 | 2 | 3 | 6 |
| door-close | 8 | 7 | 6 | 7 |
| door-open | 7 | 5 | 6 | 7 |
| drawer-close | 8 | 8 | 8 | 8 |
| drawer-open | 7 | 8 | 8 | 8 |
| faucet-open | 0 | 0 | 1 | 0 |
| handle-press | 8 | 8 | 8 | 8 |
| pick-place | 0 | 0 | 0 | 0 |
| plate-slide | 8 | 7 | 7 | 8 |
| reach | 8 | 6 | 4 | 4 |

| Selected 5k versus preserved reference | Reference successes | Wins | Regressions |
| --- | ---: | ---: | ---: |
| guided_ttt | 75 | 8 | 10 |
| progress_ttt | 78 | 7 | 12 |
| flow_reasoning | 79 | 2 | 8 |
| execution_revision | 80 | 5 | 12 |
| controller_grounded | 79 | 8 | 14 |
| leflow_release | 21 | 56 | 4 |
| hwm_paper_port | 5 | 68 | 0 |
| cem_release | 24 | 49 | 0 |
| repair_flow_ttt | 80 | 4 | 11 |
| retrieval_repair_ttt | 80 | 8 | 15 |
| shifted_repair_ttt | 79 | 5 | 11 |
| progress_objective_ttt | 78 | 6 | 11 |
| bayesian_repair_ttt | 77 | 7 | 11 |

[All selected paired case IDs and complete per-round diagnostics](reports/20261012-nonlinear-repair-ttt/analysis/final-details.json) retain case/reset/model-seed/hash matching. [15k paired cases](reports/20261012-nonlinear-repair-ttt/paired-and-timeouts-15000.json) and [20k paired cases](reports/20261012-nonlinear-repair-ttt/paired-and-timeouts-20000.json) preserve later regressions separately. Eight-rank training differs from the four-rank retrieval reference in random streams and within-rank negative goals despite the same global batch and data. These comparisons do not isolate the memory component.

The final 20k checkpoint gains dial-turn 1/3/6 and drawer-open 0 versus selected 5k, but loses door-close 3 and reach 1/4/6/7. All five regressions reach 200 actions. Assembly and pick-place remain 0/8 throughout; faucet scores 0/0/1/0 and reach 8/6/4/4. At 20k, failed pick-place last-20 route tails contain only reach/direct-goal choices. Those routes are provenance, not physical failure labels.

## Memory fit, prediction and physical success

| Updates | Supported fits | Support MSE before→after | Raw / prior / adapted next-prefix MAE | Repair choices / logged plans |
| --- | ---: | --- | --- | --- |
| 5000 | 5495 | 0.255950 → 0.120733 | 0.003162 / 0.005203 / 0.002080 | 2221/5599 |
| 10000 | 5693 | 0.251404 → 0.127880 | 0.003104 / 0.004852 / 0.001935 | 2515/5797 |
| 15000 | 5778 | 0.232229 → 0.121413 | 0.003000 / 0.004529 / 0.001845 | 2110/5882 |
| 20000 | 5539 | 0.224672 → 0.119800 | 0.002935 / 0.004338 / 0.001833 | 1906/5643 |

Every supported fit lowers its factual support MSE. That is an in-support property; normalized support MSE and model-distance prefix MAE use different units. Applied adaptation lowers aggregate next-prefix error relative to raw prediction, but physical success does not improve over 80. Empty-history prior error is worse than raw at every checkpoint. Observed support-count, repair and incumbent groups visit different states and are not an ablation or randomized comparison.

| Updates | Initial pool spread | Residual pool spread | Generated / retrieved / shifted anchors |
| --- | ---: | ---: | --- |
| 5000 | 0.059628 | 0.029319 | 1987 / 1552 / 2060 |
| 10000 | 0.034830 | 0.020647 | 2148 / 1500 / 2149 |
| 15000 | 0.025751 | 0.015577 | 2055 / 1591 / 2236 |
| 20000 | 0.023137 | 0.014233 | 1913 / 1565 / 2165 |

Initial states, route costs, goals and case seeds are matched, with empty histories. Proposal spread contracts; it is not likelihood entropy or a demonstrated cause of failure. Generated and retrieved anchors appear in training; shifted deployment plans remain an approximation inherited from retrieval-repair. This run does not test that approximation separately. The learned memory is fitted once per decision from at most four executed two-action prefixes, then fixed throughout search. Generated alternatives receive no invented physical outcomes.

| Last 500-update logged window | Action-flow loss | Repair-flow loss (mixed depths) | Teacher clipping (mixed depths) |
| --- | ---: | ---: | ---: |
| 5000 | 0.174954 | 0.216217 | 4.180% |
| 10000 | 0.122822 | 0.146968 | 2.354% |
| 15000 | 0.091324 | 0.129612 | 1.963% |
| 20000 | 0.082920 | 0.117075 | 2.160% |

Depth 1 has no repair loss; saved diagnostics include depth-specific values. Lower action/repair loss and clipping do not establish improved execution. Teacher interpolation contains the target action and is not a physical-success label.

## Timeouts, stalls and saved visual evidence

Across 416 episodes, 138 failures comprise 134 episodes reaching 200 actions and 4 controller timeouts. Full timing sequences and budget overruns remain in the original reports and [10k timeout record](reports/20261012-nonlinear-repair-ttt/paired-and-timeouts-10000.json). No work was replayed or given a larger budget.

At 10k the four timeouts are assembly 00000 (198 actions), faucet-open 00005 (178), and pick-place 00004/00005 (112/96). The pick-place traces contain 44/57 and 41/49 calls above 150 ms. The source of the slow calls is not established. Their observed overruns are 0.014535/0.043772/0.001743/0.062748 seconds; time is charged and actions returned after the shared deadline are withheld. Pick-place 00004 has a fully logged final plan that was never executed. Plan-choice counts therefore do not all denote executed actions. All four timeout cases already failed at 5k; this does not reveal their counterfactual outcome with more time.

Failure tails preserve model-predicted, corrected and next-observed encoded progress. These are model proxies, not contact forces or physical counterfactual truth. Late reach/direct-goal route provenance persists in failed pick-place episodes. Task provenance may explain what was selected, but is not an oracle task-ID filter or a causal failure label.

All ten saved cases at each checkpoint were individually inspected at readable resolution. The saved assembly case withdraws late at the first three checkpoints; at 20k its hand remains near the object through action 200 without success. In the pick-place examples the object remains visible near the table in late samples; several dial/faucet examples move near their apparatus and later away. At 15k the saved door-close case leaves the door open, while sampled reach, coffee, drawer and handle examples succeed. Exact per-image observations, original trajectory identities and PNG hashes are in the review records below. Sparse samples cover ten cases per checkpoint, not every failure, and cannot resolve forces, slip or missed contact.

- [5k exact saved-image review](reports/20261012-nonlinear-repair-ttt/visual-review-5000.json) and [image index](reports/20261012-nonlinear-repair-ttt/contact-sheets-5000/index.json).
- [10k exact saved-image review](reports/20261012-nonlinear-repair-ttt/visual-review-10000.json) and [image index](reports/20261012-nonlinear-repair-ttt/contact-sheets-10000/index.json).
- [15k exact saved-image review](reports/20261012-nonlinear-repair-ttt/visual-review-15000.json) and [image index](reports/20261012-nonlinear-repair-ttt/contact-sheets-15000/index.json).
- [20k exact saved-image review](reports/20261012-nonlinear-repair-ttt/visual-review-20000.json) and [image index](reports/20261012-nonlinear-repair-ttt/contact-sheets-20000/index.json).

## Identity, budget and completion

Frozen source: `d0cae3007c7d678cdc5ddb68ce731e8a2db3005d`. Server record: `/home/mtxu/adam/LeFlow-experiments/20261012-nonlinear-repair-ttt`. The frozen execution tree remained clean. [Final audit](reports/20261012-nonlinear-repair-ttt/analysis/final-audit.json) verifies the same 6,222 expert TRAIN episodes, seed 3072, global batch 64, frozen world/route-bank tensors, all milestone weights and best/last checkpoints; model binaries remain on the server under `campaign/runs/nonlinear_repair_ttt_3072/`. [Final health](reports/20261012-nonlinear-repair-ttt/final-health.json) records online completion and all 44 owned processes absent.

The method is a small-context LaCT-inspired adaptation, not a full LaCT/Muon reproduction: dimension 32, hidden 64, three fast matrices, four differentiable bounded MSE updates, prior protection and gradient clipping. Empty history returns the prior; episode reset clears adaptation. Evaluation uses two rounds, radius 0.25, incumbent retention, 320 predicted world transitions per full decision, 200 primitive actions and 10 cumulative controller seconds. The runtime remained unchanged after freeze; the reporting checkout supplied only read-only CPU analysis.

| Measured component | GPU hours |
| --- | ---: |
| Prior cumulative components | 111.56490618890255 |
| Staging and frozen preflight/DDP checks | 0.032640774132 |
| Retrieval setup | 0.005036057739 |
| Optimization | 4.812525830430 |
| Four evaluations | 2.325841965845 |
| Cumulative known measured components | **118.74095081704824** |

Optimization stops at 20,000 updates and 17325.092989549 aggregate GPU-seconds, below 28,800; last-update overrun is 0.0 seconds. Learned memory and per-update retrieval are inside that timer. Preparation is separately recorded; failed configuration-mismatch preflight GPU time remains explicitly unmeasured. CPU deployment, reporting, transfer and idle time are not assigned invented GPU charges. This is not a full cluster bill. [Full compute accounting](reports/20261012-nonlinear-repair-ttt/compute-accounting-final.json).

Full metrics and every evaluation are lossless gzip, with original byte hashes in readback records and stored-file hashes in the [ledger](results-ledger.json). Earlier first/second/third metric and compute snapshots remain under `run/snapshots/`; [preservation checks](reports/20261012-nonlinear-repair-ttt/analysis/preservation-checks.json) verify their original bytes and exact metric prefixes. No raw datasets, credentials or model binaries are published.

## Interpretation and review hold

The selected 73/104 has 8 wins/15 regressions versus retrieval-repair 80 and 5/12 versus execution-revision 80. It remains below both references. Lower fitting/prediction losses, a more expressive memory and shrinking proposals do not demonstrate physical-success gains. One seed, repeatedly reused development selection, custom baseline ports, changed rank-dependent random streams and no component ablations limit inference. This does not establish generalization, native-benchmark SOTA, novelty or a causal benefit/harm of the memory. Final tests remain sealed.

This single authorized run is complete; stop further launches for review. The remaining supervision, history coverage, diversity or latency questions do not authorize a speculative variant. Read the [post-run review](NONLINEAR_REPAIR_TTT_REVIEW.md). Publication is complete only after live origin/main and all ledger artifact bytes have been verified; the receipt is added after that check.
