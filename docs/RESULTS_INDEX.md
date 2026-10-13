<!-- native-crossbench-current:start -->
# October 13 native campaign checkpoint

Native PushT/Cube status: **16/16 completed, 64/64 full physical reports; complete**. Frozen training source remains `747f635ad36af2cbe3e2bbb0b7997efe9caccaa1`; seed3072/global64,20k updates OR8 optimizer GPUh, four native50 evaluations,50 actions/10 seconds. Native records use [their separate ledger](crossbench-results-ledger.json). The MetaWorld21-comparison/81-report ledger is unchanged.

Cube search's first attempt stopped at W&B initialization before any optimizer update or registered evaluation. Its native preflight0.004619815980291201GPUh and no-update DDP0.00035402999363011786GPUh remain preserved. Startup wait time is unmeasured and is not an optimizer charge. The second operational recovery retained ten completed runs and finished the six remaining registered jobs. All sixteen jobs are now consumed; neither completed queue nor any completed job may be restarted. No changed training source, completed work replay, extra method, seed, data or budget is authorized. A failed recovery holds again without automatic restart.

See [native results](CROSSBENCH_RESULTS.md), [failed-attempt evidence](reports/20261013-crossbench-baselines/failures/cube-action-flow-search-wandb-init/failure.json) and [recovery registration](reports/20261013-crossbench-baselines/recovery2-registration.json). All 16 native contact sheets, covering all 160 archived selected-checkpoint trajectories, were visually inspected in full. Review scope, case mapping, hashes and findings are in docs/reports/20261013-crossbench-baselines/visual-review/README.md; this is a ten-case subset per run, not visual inspection of all 800 selected cases. Source-backbone overlap, custom ports and single-seed development selection preclude causal-component or SOTA claims. Final tests remain sealed.

<!-- native-crossbench-current:end -->

# Completed results and publication index

**All 21 registered MetaWorld comparisons are complete; no run remains active.** The latest five-method batch adds 20 physical development evaluations, giving 81 complete evaluation reports. Every trained policy has four fixed checkpoints. These are single-seed development selections; final tests remain sealed.

The [five-method report](ADDITIONAL_BASELINE_RUNS_20261012.md) contains the common setup, full new curves, per-task and paired comparisons, actual compute and failure findings. The [machine-readable ledger](results-ledger.json) records immutable source/run identities and artifact hashes.

**Vanilla flow 98 is the strongest measured reference. Our proposed retrieval-repair 80 does not outperform it.** Do not identify the standard BC controller as a new reasoning/TTT method or claim scaling benefits from development scores alone.

## All preserved comparisons

| Method and full report | Success curve /104 | Selected | Optimization GPUh |
| --- | --- | ---: | ---: |
| [vanilla_action_flow](VANILLA_ACTION_FLOW_RESULTS.md) | 77 / 84 / 89 / 98 | 98 at 20,000 | 0.613 |
| [action_flow_search](ACTION_FLOW_SEARCH_RESULTS.md) | 74 / 79 / 84 / 78 | 84 at 15,000 | 0.601 |
| [execution_revision](EXECUTION_REVISION_RESULTS.md) | 80 / 78 / 79 / 76 | 80 at 5,000 | 3.233 |
| [repair_flow_ttt](REPAIR_FLOW_TTT_RESULTS.md) | 80 / 73 / 75 / 75 | 80 at 5,000 | 1.620 |
| [retrieval_repair_ttt](RETRIEVAL_REPAIR_TTT_RESULTS.md) | 75 / 77 / 79 / 80 | 80 at 20,000 | 1.749 |
| [controller_grounded](CONTROLLER_GROUNDED_RESULTS.md) | 79 / 78 / 75 / 78 | 79 at 5,000 | 2.422 |
| [flow_reasoning](FLOW_REASONING_RESULTS.md) | 79 / 75 / 76 / 76 | 79 at 5,000 | 3.166 |
| [shifted_repair_ttt](SHIFTED_REPAIR_TTT_RESULTS.md) | 79 / 78 / 74 / 78 | 79 at 5,000 | 2.233 |
| [progress_objective_ttt](PROGRESS_OBJECTIVE_TTT_RESULTS.md) | 72 / 72 / 78 / 77 | 78 at 15,000 | 4.585 |
| [progress_ttt](PROGRESS_TTT_RESULTS.md) | 77 / 76 / 78 / 74 | 78 at 15,000 | 3.542 |
| [bayesian_repair_ttt](BAYESIAN_REPAIR_TTT_RESULTS.md) | 71 / 76 / 77 / 77 | 77 at 15,000 | 4.148 |
| [latent_revision](LATENT_REVISION_RESULTS.md) | 73 / 75 / 74 / 77 | 77 at 20,000 | 3.244 |
| [sage_port](SAGE_PORT_RESULTS.md) | 62 / 77 / 77 / 75 | 77 at 10,000 | 0.714 |
| [feedback_wm_port](FEEDBACK_WM_PORT_RESULTS.md) | 42 / 65 / 75 / 76 | 76 at 20,000 | 0.582 |
| [guided_ttt](GUIDED_TTT_RESULTS.md) | 72 / 70 / 72 / 75 | 75 at 20,000 | 4.595 |
| [diffusion_policy_port](DIFFUSION_POLICY_PORT_RESULTS.md) | 43 / 69 / 70 / 74 | 74 at 20,000 | 0.593 |
| [nonlinear_repair_ttt](NONLINEAR_REPAIR_TTT_RESULTS.md) | 73 / 66 / 67 / 72 | 73 at 5,000 | 4.813 |
| [leflow_ttt](LEFLOW_TTT_RESULTS.md) | 27 / 21 / 27 / 29 | 29 at 20,000 | 2.437 |
| [cem_release](RELEASE_BASELINE_RESULTS.md) | 24 | 24 at 0 | 0.000 |
| [leflow_release](RELEASE_BASELINE_RESULTS.md) | 14 / 21 / 19 / 14 | 21 at 10,000 | 0.852 |
| [hwm_paper](RELEASE_BASELINE_RESULTS.md) | 2 / 3 / 5 / 5 | 5 at 15,000 | 2.077 |

Each learned run uses seed 3072, global batch 64, the same 6,222 successful expert TRAIN episodes and frozen shared representation/world, a 20k-update OR eight optimizer-GPUh ceiling, and the same 104-case development/controller limits. Equal caps are not equal consumed compute or convergence. CEM has no learned head. SAGE/Feedback-WM are inspired ports; Diffusion Policy and LeFlow have documented shared-backbone adaptation differences. Final success within 200 actions and ten cumulative controller seconds is the primary metric.

The publication verifier checks all hashed artifacts and all 81 evaluation reports against identical episode/reset/model-seed/task identities. It performs no model/simulator calls and reads no final-test outcomes. Run `python3 scripts/analysis/verify_result_publications.py`. [Final live-server health](reports/20261012-additional-baselines/batch-final-health.json) and [remote publication receipt](reports/20261012-additional-baselines/publication-final-verification.json) preserve completion and exact delivery verification.

## Earlier campaigns

[Historical repairs and baseline defects](REPAIR_RESULTS.md),
[repaired-method results and changed-world limits](REPAIRED_VALIDATION_RESULTS.md),
[baseline release fidelity](RELEASE_BASELINES.md), and
[chronological progress](PROGRESS.md) preserve earlier results and implementation
history. Historical LeFlow/HWM adaptations and different-world measurements
must not be pooled with the fixed comparison table above. A poor or superseded
result remains evidence and must not be deleted or silently relabeled.

## Paper thesis and technical formulation

The [paper blueprint](PAPER_BLUEPRINT.md) defines execution-calibrated bounded
flow repair as the intended central mechanism. It records the implemented
objective, factual adaptation, candidate-conditioned workspace, residual flow,
incumbent retention, contribution status, related-work boundaries, current
abstract, figure designs and prospective evidence requirements. It distinguishes
the retained retrieval80 method from the unsuccessful nonlinear73 extension.

## Implementation and behavior map

The source revision in each ledger entry fixes the implementation used for that
run; current source files retain the named variants. Configuration, runtime,
checkpoint and provenance records are linked from each result report.

| Experiment family | Implementation | Observed result / finding |
| --- | --- | --- |
| Grounded / latent / execution revision | [model](../flow_jepa/execution/model.py), [revision](../flow_jepa/execution/revision.py), [controller](../flow_jepa/execution/aligned.py) | Selected 79 / 77 / 80; GMM reference is not a flow generator |
| Reasoning-conditioned flow | [flow model](../flow_jepa/execution/flow_reasoning.py), [controller](../flow_jepa/execution/flow_controller.py) | 79; later demonstration fitting does not increase success |
| Episodic correction / guided flow | [progress TTT](../flow_jepa/execution/progress_ttt.py), [guidance](../flow_jepa/execution/guided_ttt.py) | 78 / 75; calibration and controller time matter separately |
| Bounded repair / retrieved anchors | [repair](../flow_jepa/execution/repair_flow_ttt.py), [retrieval](../flow_jepa/execution/retrieval_repair.py) | Both select 80; retained flow references |
| Shifted-plan training | [implementation](../flow_jepa/execution/shifted_repair.py) | 79; no improvement over retained reference |
| Goal-progress objective | [implementation](../flow_jepa/execution/progress_objective.py) | 78; task-relevant progress remains unresolved |
| Bayesian repair acceptance | [model](../flow_jepa/execution/bayesian_repair.py), [controller](../flow_jepa/execution/bayesian_controller.py) | 77; evidential support does not establish better executed choices |
| Nonlinear memory on repair | [implementation](../flow_jepa/execution/nonlinear_repair.py) | 73; adapted next-prefix error improves while physical success does not |
| Direct LeFlow TTT | [model](../flow_jepa/leflow_ttt/model.py), [controller](../flow_jepa/leflow_ttt/controller.py) | 29; separate from TTT on the stronger repair controller |
| Released / paper baseline ports | [implementations and upstream provenance](../flow_jepa/baselines/) | LeFlow 21, HWM 5, CEM 24; native-method and runtime caveats above |
| Original and repaired latent-flow campaigns | [historical models](../flow_jepa/models.py), [training](../flow_jepa/train.py) | Different-world results and confirmed sampler defect remain separately documented |

The strongest recurring observation is a mismatch between model-distance error
and physical success. Lower calibration/action losses do not establish useful
counterfactual action rankings. Pick-place can follow reach/direct-goal routes,
but some earlier failures retained pick-place routes; route filtering is not a
confirmed remedy. Proposal contraction occurs alongside some regressions, yet
also appeared in improving runs, so it is not a demonstrated cause. Timeouts
must be distinguished from failures reaching 200 actions. Detailed paired cases,
full curves and saved visual reviews remain in the linked per-run reports.

The older PushT/BTM work belongs to a different campaign and remains preserved
on [its original remote branch](https://github.com/Abecid/LeFlow-/tree/codex/server-launch-20261004),
including [its report archive](https://github.com/Abecid/LeFlow-/tree/codex/server-launch-20261004/docs/research/pusht_pair_20261004).
It is not pooled with the MetaWorld comparison or silently merged into current
method implementations.

The October 12 publication audit additionally compares all six archived full
training logs from the October 7/9 MetaWorld campaigns with the live server's
original JSONL records: [verification receipt](reports/20261012-publication-audit/historical-training-log-verification.json).
Their original JSON-array export preserves every parsed metric record.

## Standing completion requirement

The user requires **every training run, including baselines, to have its results,
findings and evaluation metrics logged and pushed to `origin/main` when it
completes**. Reporting is part of completion, not a later optional task.

1. Before an authorized launch, register the run in `active_runs` in the ledger:
   method, seed, immutable source, server record/run paths, expected validation
   milestones, data/controller/compute contract and W&B identity when available.
   Registration is not permission to launch another experiment.
2. At completion, confirm the final registered evaluations and completion marker,
   process exit/online status, case identities, source/data/checkpoint hashes and
   charged compute. A training-exit marker alone is not evaluation completion.
3. Preserve full periodic metrics and all evaluation records, including negative
   results, timeouts, interruptions/recoveries and failure evidence. Lossless gzip
   is appropriate. Preserve checkpoints/raw data on the server and record their
   paths and hashes; never add credentials or large model binaries to Git.
4. Write a findings report with the full learning curve, selected/final results,
   appropriate paired baseline comparisons, actual compute and material claim
   limits. Distinguish observations from causal hypotheses. Do not invent missing
   outcomes or rerun training/evaluation merely to fill a report.
5. Update this index, `PROGRESS.md`, current status documents and the ledger;
   remove only the completed active entry. Recheck the actual saved evidence and
   run the publication verifier. Preserve the frozen execution checkout and all
   completed baseline references.
6. Commit the relevant report/evidence changes, push to `origin/main` without
   force, and verify the published commit and report contents. Reconcile concurrent
   remote changes without discarding anyone's work. A local commit or ledger entry
   alone is not proof of remote publication. If publication fails, preserve the
   local commit, record the exact failure and retry; report the unresolved failure.

The project completion monitor is configured to check every15 minutes as a
fallback to immediate publication by the active agent. It remains quiet when
there is no actionable change. The October11 user request now authorizes one
failure-supported next correction only after the active run is completed, analyzed
and published, under the unchanged data/seed/budget/evaluation contract. Stop new
launches for review after an improvement over80/104 or when no concrete supported
correction remains. The monitor itself supplies no additional authorization; no
baseline retraining, extra seeds, blind sweeps or sealed tests are allowed.
Scheduled execution depends on the Codex host
being available and able to reach the server and GitHub; an outage must never be
reported as a successful push.
