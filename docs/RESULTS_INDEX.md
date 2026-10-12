# Results and completion publication

Updated October 12, 2026. Nonlinear episodic memory completes **73/66/67/72**,
selecting **5k 73/104**, below the retained **80/104** references. All scores use
the same 104 development cases and seed 3072; final tests remain sealed. Custom
method ports, one seed, reused cases and matched ceilings do not establish
native-benchmark SOTA, component causality or generalization.

## Current iteration status

[Nonlinear memory is complete](NONLINEAR_REPAIR_TTT_RESULTS.md). Its 800 periodic
records, four complete evaluations, four controller timeouts and 40 inspected
contact sheets are preserved with identity and compute records. The final CPU
audit verifies source/runtime/data/bank/checkpoints and nonlinear memory accounting;
all 44 owned processes exited and W&B mxujto8j finished. No run remains active.
[Further launches are held for review](NONLINEAR_REPAIR_TTT_REVIEW.md); the user
authorized this one candidate only. The monitor stays available for future explicitly
authorized and registered work. Cumulative measured components: 118.74095081704824 GPUh,
excluding the disclosed unmeasured preparation; this is not a full cluster bill.
Live origin/main and artifact verification is required before publication is confirmed.

## Preserved comparisons

The four learned-model scores correspond to5k/10k/15k/20k total updates. CEM has
no head training. All completed reports, full evaluation records, periodic
training metrics and compute records remain in Git; checkpoints stay on the
server with recorded paths/hashes. Highest development success selects the
checkpoint, with earliest ties.

| Method and findings | Success curve /104 | Selected | Optimization GPUh |
| --- | --- | ---: | ---: |
| [Nonlinear episodic repair](NONLINEAR_REPAIR_TTT_RESULTS.md) | 73 / 66 / 67 / 72 | 73 at5k | 4.813 |
| [Bayesian posterior repair](BAYESIAN_REPAIR_TTT_RESULTS.md) | 71 / 76 / 77 / 77 | 77 at15k | 4.148 |
| [Execution-calibrated goal progress](PROGRESS_OBJECTIVE_TTT_RESULTS.md) | 72 / 72 / 78 / 77 | 78 at15k | 4.585 |
| [Shifted-plan repair](SHIFTED_REPAIR_TTT_RESULTS.md) | 79 / 78 / 74 / 78 | 79 at5k | 2.233 |
| [Retrieval-conditioned repair](RETRIEVAL_REPAIR_TTT_RESULTS.md) | 75 / 77 / 79 / 80 | 80 at20k | 1.749 |
| [Repair-flow TTT](REPAIR_FLOW_TTT_RESULTS.md) | 80 / 73 / 75 / 75 | 80 at5k | 1.620 |
| [Guided action-flow TTT](GUIDED_TTT_RESULTS.md) | 72 / 70 / 72 / 75 | 75 at20k | 4.595 |
| [Execution revision, GMM](EXECUTION_REVISION_RESULTS.md) | 80 / 78 / 79 / 76 | 80 at5k | 3.233 |
| [Reasoning-conditioned action flow](FLOW_REASONING_RESULTS.md) | 79 / 75 / 76 / 76 | 79 at5k | 3.166 |
| [Flow + episodic adaptation](PROGRESS_TTT_RESULTS.md) | 77 / 76 / 78 / 74 | 78 at15k | 3.542 |
| [Controller-grounded planner](CONTROLLER_GROUNDED_RESULTS.md) | 79 / 78 / 75 / 78 | 79 at5k | 2.422 |
| [Latent revision](LATENT_REVISION_RESULTS.md) | 73 / 75 / 74 / 77 | 77 at20k | 3.244 |
| [Direct LeFlow TTT](LEFLOW_TTT_RESULTS.md) | 27 / 21 / 27 / 29 | 29 at20k | 2.437 |
| [LeFlow release-module/spatial port](RELEASE_BASELINE_RESULTS.md) | 14 / 21 / 19 / 14 | 21 at10k | .852 |
| [HWM paper port](RELEASE_BASELINE_RESULTS.md) | 2 / 3 / 5 / 5 | 5 at15k | 2.077 |
| [Released CEM](RELEASE_BASELINE_RESULTS.md) | 24 | 24 | 0 |

Optimization excludes validation and shared-world training; those costs are
preserved in the reports. Caps are matched, consumed compute is not. Each learned
method stopped at20k updates within the8 optimization-GPUh ceiling. The identical
6,222 successful training episodes supply policy training, with replacement
window sampling; no conventional epoch count is implied. The controller
allowance is10 cumulative seconds and200 primitive actions. The 80/104 GMM
controller does not use a diffusion/flow generator.

The machine-readable [results ledger](results-ledger.json) points to each run's
identity, complete marker, metrics, all evaluations and SHA256 hashes. Its
read-only verifier checks the hashed evidence artifacts,61 evaluation reports and all
104 case identities across the sixteen current comparisons. Run
`python3 scripts/analysis/verify_result_publications.py` from the repository.
It invokes no models or simulators and does not read final-test outcomes.

## Earlier campaigns

[Historical repairs and baseline defects](REPAIR_RESULTS.md),
[repaired-method results and changed-world limits](REPAIRED_VALIDATION_RESULTS.md),
[baseline release fidelity](RELEASE_BASELINES.md), and
[chronological progress](PROGRESS.md) preserve earlier results and implementation
history. Historical LeFlow/HWM adaptations and different-world measurements
must not be pooled with the fixed comparison table above. A poor or superseded
result remains evidence and must not be deleted or silently relabeled.

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
