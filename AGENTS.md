# Experiment continuity

- Completion recorded October 9, 07:22 UTC: the authorized repaired world/head
  and four validation rounds are finished, W&B is finished, and all owned
  processes exited. Read docs/REPAIRED_VALIDATION_RESULTS.md. Preserve all
  results. The existing monitor is paused after verified publication. Do not
  restart any coordinator or launch training, ablations or final tests without
  new user authorization. Later bullets retain historical execution context.

- Latest throughput authorization: use all available GPUs for the SAME repaired
  run and registered evaluations. The execution-only runtime preserves global
  batch64, original per-rank random streams and per-microbatch consistency
  subsets while fusing physical microbatches to16. Train on GPUs0–3 and overlap
  validation on4–7; use all8 for final validation after optimization stops.
  Preserve the original 7,200-second four-GPU optimization allowance, charged
  ledgers, seed3072, cases and four validation rounds. No baseline reruns, extra
  seeds, new variants or final tests are authorized. Read docs/THROUGHPUT_RUNTIME.md
  and the migration/runtime manifests before recovery; never resume superseded
  coordinators807392/858560 after the checkpoint handoff is recorded.


- Latest user correction (October 8, Pacific evening): freeze previously trained
  baselines and their evaluation records. Retraining all baselines was an overly
  broad interpretation. Continue the already-running compatible world and train
  ONLY joint_flow_consistent for this repaired iteration, under unchanged budgets.
  Read docs/OURS_ONLY_ITERATION.md. The old two-pool/all-method schedules are
  cancelled. NEVER resume supervisor 778158 or coordinator 795290; they would
  dispatch cancelled baselines. Use the deployed ours-only continuation and its
  scope/status records. No LeFlow/HWM retraining or new baseline final tests.
  Compare fixed validation case outcomes honestly as historical full-pipeline
  references: state/world representations differ and old LeFlow has a known
  sampler defect. Do not claim a common-world comparison or corrected-SOTA win.
  Keep test data reserved for the selected final method. No automatic subsequent
  variants, extra seeds, ablations or budget extensions; analyze this run first.
  Preserve old campaigns and publish evidence/plans to origin/main.

- The active project is the non-BTM V-JEPA 2.1 / MetaWorld campaign on branch
  `research/joint-flow-metaworld`. Follow `docs/FLOW_EXPERIMENT.md`.
- The user authorizes SSH to `target_server_2` (or `target_server_2_cf`),
  continuation of the existing compatible world and then one repaired
  joint_flow_consistent head with online W&B and fixed validation. All available GPUs may accelerate
  this same run and its registered evaluations; the earlier baseline dual-pool
  queue remains cancelled.
- Use exactly one training seed, 3072, per the user's October 7 correction.
  Never launch additional training seeds. Train only the already-running world
  and the one authorized joint_flow_consistent head. LeFlow/HWM/CEM references
  are frozen; existing registered world CEM diagnostics continue. No final tests,
  additional variants or ablations are queued. Preserve per-run ceilings and
  original cases, and report cumulative compute and historical-comparison limits.
- Keep `docs/PROGRESS.md` current. Record code changes, commands/configuration,
  tests, experiment findings (including negative results), run/checkpoint paths,
  W&B links, failures, and the concrete next step.
- Commit and push relevant additions, edits, progress reports, and compact
  experiment results to Git remote `origin` at each meaningful checkpoint and
  before ending a turn. Verify the remote commit. Never rely on chat history as
  the only record. If push fails, preserve a local commit and report the failure.
- Keep credentials, raw datasets, and large model binaries out of Git. Preserve
  them on the authorized server; record their paths and hashes in reports.
- Keep a running campaign's checkout/revision frozen. Publish later reports from
  a separate checkout; do not change the code beneath an active run.
- Use validation for development and selection; keep test results sealed until
  the registered models are trained. Report actual comparisons without assuming
  the proposed method wins. Method-family adaptations are not exact SOTA replicas.
