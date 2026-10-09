# Experiment continuity

- Latest user correction, October 9 UTC: after repairs, run training FROM SCRATCH
  for a fresh matched comparison, including periodic evaluation and a final
  failure analysis. This explicitly supersedes the earlier no-training hold for
  the NEW campaign only. Follow docs/REPAIRED_COMPARISON.md. Retrain the shared
  world and all three learned heads, same seed 3072, original per-model time/update
  ceilings, exact original train/evaluation episodes, and GPUs 0–3 only.
  Keep the original campaign held and its checkpoints/ledgers immutable. No
  extra seeds, ablations, sweeps, budget extensions, or selective test reruns.
  Publish latest evidence and plans to origin/main, preserving remote history.

- The active project is the non-BTM V-JEPA 2.1 / MetaWorld campaign on branch
  `research/joint-flow-metaworld`. Follow `docs/FLOW_EXPERIMENT.md`.
- The user authorizes SSH to `target_server_2`, at most four GPUs total for this
  campaign, training, online W&B logging, and matched baseline evaluation.
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
