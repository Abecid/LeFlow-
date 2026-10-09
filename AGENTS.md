# Experiment continuity

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
- The user authorizes SSH to `target_server_2`, up to eight GPUs total for this
  campaign in two four-GPU jobs, training, online W&B logging, and matched
  baseline evaluation. The world continues unchanged on GPUs 0–3.
- Use exactly one training seed, 3072, per the user's October 7 correction.
  Never launch additional training seeds. Run only joint_flow_consistent,
  leflow_adapted, hwm_adapted, and cem_long. No ablations until first results
  and failure analysis justify them and the user authorizes subsequent work.
  Enforce matched compute limits, shared train/evaluation data and periodic W&B evaluation.
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
