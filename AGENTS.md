# Experiment continuity

- Baseline recovery, October 10: both sources remain frozen, but use
  `PYTHONPATH=/tmp/mtxu-baseline-runtime-guard-v2:<execution-checkout>` when
  resuming these same workers. The guard retries transient shared-filesystem
  journal-lock EAGAIN and exposes errors before NCCL teardown. It changes no
  algorithm or training state. Both resumed from 10k with 52 journaled cases;
  source/hash/attempt records are in `recovery-*` under each run. LeFlow launcher
  is2265942, HWM2266131. Never start a fresh run or discard their saved journals.
  Read the runtime-recovery section in docs/RELEASE_BASELINES.md. Count interrupted
  process occupancy separately from optimizer and successful-validation compute.

- Active corrected baselines: server root
  `/home/mtxu/adam/LeFlow-experiments/20261010-baseline-release`.
  LeFlow `leflow_3072`, GPUs0–3, frozen source564b52f (`repo`), W&Bown982ph.
  HWM `hwm_3072`, GPUs4–7, frozen sourcecde9c39 (`repo-hwm-bundle`), W&B8gpcqqc2.
  Both are running, not completed. Do not edit these execution trees or relaunch
  them fresh. Resume only the same run with its original source/configuration.
  CEM completed24/104, registry verified, W&Btupg6zpf finished; never rerun it
  automatically. Native HWM default exceeded10s before acting; the registered
  paper-port uses Appendix C smaller planning settings, not headline defaults.
  Read docs/RELEASE_BASELINE_RESULTS.md and docs/RELEASE_BASELINES.md. Preserve
  the same caps/cases. Final tests and our-method retraining remain disabled.


- Latest user authorization, October 9 (Pacific): repair baseline fidelity and
  run the relevant corrected baselines once. This supersedes older no-baseline-
  rerun holds. Read docs/RELEASE_BASELINES.md. LeFlow uses pinned release
  modules and its Appendix-E adapter; CEM uses the pinned released solver.
  HWM robotics code is not released: label any implementation a paper-based
  shared-backbone port, never an exact author-code reproduction. No techniques
  from our policy may enter baseline heads/controllers. Preserve seed3072,
  identical episodes/fixed104, global64,20k/28,800 optimizer-GPU-second caps
  (including any adapter learning), four validations,10s/200-action limits.
  All8 GPUs may be used non-preemptively. Do not retrain our candidate or open
  final tests. Freeze source/config/checkpoints/reports in a hashed registry
  and reuse completed baselines. No automatic seeds, sweeps or budget extension.
  Older completed-run bullets below describe historical authorization only.

- Execution-revision iteration completed October 9: one fresh seed3072 run,
  20,000 updates and all four fixed104 validations (80/78/79/76). Selected5k
  scores80/104 versus controller-grounded79 and latent-revision77. This is a
  one-case gain over the strongest reference, not reliable SOTA evidence.
  Read docs/EXECUTION_REVISION_RESULTS.md and its final audit/compute ledger.
  Later training regresses; matched-initial proposal diversity falls about51%.
  All14 tests passed, W&B synced, owned workers exited and all8 GPUs were idle
  at2026-10-09T23:50:16.141415+00:00. The one-run authorization is fulfilled. Preserve
  frozen runs and sealed tests; no automatic training, variants, baseline reruns,
  additional seeds, ablations or final tests. Later authorization bullets are history.

- Latest user authorization: implement the diagnosed execution-alignment,
  repeated-stall, plan-reuse and sampling-coverage fixes, then run ONE fresh
  `execution_revision` candidate. This supersedes the completed-run hold for
  this iteration. Read docs/EXECUTION_REVISION_RUN.md. Preserve seed3072,
  global64,20k-update/28,800 optimization GPU-second ceilings, same episodes,
  four fixed104 validations and10s/200-action limits. Use all8 idle GPUs;
  no baseline reruns, additional seeds, variants, ablations or final tests.
  Preserve prior frozen execution trees and report the changed sampling
  distribution explicitly. Publish code, results and negative findings.

- Latent-revision iteration completed October 9, 22:39 UTC: one fresh seed3072
  run, 20,000 updates and four fixed104 validations (73/75/74/77). Selected20k
  scores77/104 versus the preserved controller-grounded79/104 reference. Read
  docs/LATENT_REVISION_RESULTS.md and its final audit/compute ledger. W&B synced,
  owned workers exited and all8 GPUs were idle at22:40 UTC. Preserve this run,
  all saved references and sealed tests. No automatic additional training,
  variants, seeds, ablations, baseline reruns or final tests are authorized.
  Later bullets are historical; the one-run authorization is fulfilled.

- Latest authorization, October 9: implement and run ONE fresh `latent_revision`
  candidate using the accepted execution-error-conditioned latent reasoning
  direction. This supersedes the completed-run hold for this iteration only.
  Preserve seed3072, same train/validation sets, global64,20k-update/28,800
  aggregate optimization GPU-second caps, four fixed104 validations and
  10s/200-primitive-action allowance. Use all8 available GPUs. Frozen world,
  encoder/cache and saved baselines remain references; no baseline reruns,
  additional seeds/variants/ablations or final tests. Read docs/LATENT_REVISION_RUN.md.

- Controller-grounded iteration completed October 9, 20:06 UTC: 20,000 updates,
  all four fixed104 validations (79/78/75/78), selected5k79/104. All owned workers
  exited; W&B synced and all8 GPUs were idle at the final inspection. Read
  docs/CONTROLLER_GROUNDED_RESULTS.md and its postrun audit/compute ledger.
  The latest LARC request is an applicability/design question, recorded in
  docs/LARC_APPLICATION_20261009.md; no second candidate was authorized/launched.
  Preserve completed results and sealed tests. No automatic further training,
  ablations, seeds, baseline reruns or final tests. Earlier bullets are history.

- Latest authorization, October 9: implement and run ONE first
  `controller_grounded` candidate, seed3072, from scratch, using the frozen
  selected world/cache and unchanged train/validation cases. The user's latest
  message explicitly supersedes the completed-run hold for this new isolated
  iteration only. Preserve 20k/global64/28,800 optimization GPU-second ceilings,
  four periodic validations and 10s/200-action control limits. All eight GPUs
  may accelerate this run. Compare saved baselines and analyze failure modes;
  no baseline reruns, extra seeds, automatic next variants, ablations or final
  tests. Keep previous execution trees frozen and record the new formulation.

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
