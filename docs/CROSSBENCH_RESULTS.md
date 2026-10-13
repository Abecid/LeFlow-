# Native PushT/Cube campaign — measured development results

Sixteen authorized fresh runs; completed MetaWorld results are reused. Training source is frozen at `747f635ad36af2cbe3e2bbb0b7997efe9caccaa1`. All runs use seed3072/global64, 20k updates OR8 optimizer-GPUh, four fixed50-case evaluations,50 primitive actions AND10 cumulative controller seconds. Native success and backbone differ from MetaWorld; do not pool them as SOTA.

| Dataset | Method | State | Success curve /50 | Best measured |
| --- | --- | --- | --- | ---: |
| cube | action_flow_search | completed, audited | 28 / 26 / 28 / 25 | 28 |
| cube | diffusion_policy_port | completed, audited | 28 / 32 / 36 / 38 | 38 |
| cube | execution_revision | completed, audited | 28 / 30 / 32 / 31 | 32 |
| cube | feedback_wm_port | completed, audited | 35 / 38 / 42 / 43 | 43 |
| cube | leflow | completed, audited | 33 / 40 / 37 / 40 | 40 |
| cube | retrieval_repair_ttt | completed, audited | 30 / 32 / 32 / 31 | 32 |
| cube | sage_port | completed, audited | 44 / 48 / 48 / 49 | 49 |
| cube | vanilla_action_flow | completed, audited | 29 / 37 / 38 / 38 | 38 |
| pusht | action_flow_search | completed, audited | 37 / 29 / 24 / 31 | 37 |
| pusht | diffusion_policy_port | completed, audited | 4 / 11 / 10 / 11 | 11 |
| pusht | execution_revision | completed, audited | 33 / 30 / 38 / 33 | 38 |
| pusht | feedback_wm_port | completed, audited | 15 / 13 / 24 / 27 | 27 |
| pusht | leflow | completed, audited | 12 / 8 / 9 / 12 | 12 |
| pusht | retrieval_repair_ttt | completed, audited | 34 / 39 / 37 / 38 | 39 |
| pusht | sage_port | completed, audited | 39 / 36 / 40 / 38 | 40 |
| pusht | vanilla_action_flow | completed, audited | 7 / 13 / 15 / 17 | 17 |

Paired outcomes at each method’s selected development checkpoint:

| Dataset | Method versus plain flow | Additional successes | Lost successes | Both fail |
| --- | --- | ---: | ---: | ---: |
| cube | action_flow_search | 2 | 12 | 10 |
| cube | diffusion_policy_port | 4 | 4 | 8 |
| cube | execution_revision | 4 | 10 | 8 |
| cube | feedback_wm_port | 8 | 3 | 4 |
| cube | leflow | 5 | 3 | 7 |
| cube | retrieval_repair_ttt | 5 | 11 | 7 |
| cube | sage_port | 12 | 1 | 0 |
| pusht | action_flow_search | 23 | 3 | 10 |
| pusht | diffusion_policy_port | 3 | 9 | 30 |
| pusht | execution_revision | 24 | 3 | 9 |
| pusht | feedback_wm_port | 15 | 5 | 18 |
| pusht | leflow | 7 | 12 | 26 |
| pusht | retrieval_repair_ttt | 25 | 3 | 8 |
| pusht | sage_port | 24 | 1 | 9 |

These pairs compare different trained systems on identical development cases. They do not isolate the effect of TTT or search, or reveal rejected-action counterfactuals.

Snapshot: 2026-10-13T16:05:18.765579+00:00. Queue failure hold: False. Lane states: {'A': 'completed', 'B': 'completed'}.

The registered methods use two disjoint four-GPU lanes. No missing score is estimated. All archived case outcomes are actual native simulator success flags. CPU/synthetic checks alone are not benchmark results.

Source/data/protocol details: [launch record](CROSSBENCH_RUNS_20261013.md). Complete/interim training logs and all physical milestone reports: [archive](reports/20261013-crossbench-baselines/).

## Final preservation and interpretation

All 16 runs reached 20,000 updates and have four complete 50-case reports, passing CPU audits, finished W&B records and zero-return queue entries. All 64 reports have zero controller timeouts. The queue finished at 15:44:43 UTC; the publication watcher verified all 454 native artifact/report paths at commit `c54f7966b9f07d9f078641b7df39c02e0dc2f553` and exited at 16:06:32 UTC. An independent final check verified the newly completed run and unchanged prior declarations. The subsequent [owned-process check](reports/20261013-crossbench-baselines/batch-final-health.json) found no remaining owned training/preflight/queue processes.

SAGE has the highest selected development score on both datasets: 40/50 PushT and 49/50 Cube. Retrieval repair selects 39/50 PushT but 32/50 Cube; LeFlow selects 12/50 PushT versus plain flow 17/50, and 40/50 Cube versus plain flow 38/50. These negative findings and task differences are retained. Custom ports, different trained systems and reused one-seed development selection prevent SOTA, generalization or causal-component conclusions.

All [16 contact sheets](reports/20261013-crossbench-baselines/visual-review/README.md) were inspected in full, covering every saved frame and goal from 160 archived trajectories. The subset is the first ten registered cases per run; no new evaluation or rendering was performed in the simulator. The full 50-case outcome records remain the quantitative evidence.

The completed runs consumed **14.0533655446664329 optimizer GPUh** and **3.229509477849221886 registered-evaluation GPUh**. The [component ledger](reports/20261013-crossbench-baselines/measured-compute-summary.json) additionally retains measured cache preparation, current preflights and the failed W&B-attempt preflights. Unmeasured failed precision-preflight and startup waiting time are explicitly excluded rather than recorded as zero; this is not a full cluster bill. Earlier MetaWorld evidence remains separate and unchanged. No further GPU work is authorized by batch completion.
