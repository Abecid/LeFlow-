# Monitoring and reporting protocol

The user authorized recurring progress reports and pushes to origin. A 15-minute
Codex heartbeat, `monitor-leflow-btm-flow-comparison`, is attached to the working
chat. Publish on branch `codex/server-launch-20261004`; preserve prior reports.
The training checkout remains frozen at
`69babde283f504120dae3dfc6e9188a0e2242be4`. Reporting changes belong only in the
local checkout, not the active server training checkout.

## Collection

Use the existing SSH alias and the existing training environment. No GPU work is
needed to collect a snapshot: model checkpoints are read on CPU. The working
chat retains the exact server paths and credentials remain in their existing
authentication stores. Do not copy credentials into reports or command output.

1. Inspect the last report, Git status, branch, and origin. Fetch origin and
   preserve unrelated changes. The report branch must advance without force.
2. Run `scripts/monitoring/collect_campaign_report.py` on the server with its
   existing Conda Python. It accepts `--record-root`, `--stablewm-home`, and
   `--campaign`; the deployment's `runtime.env` defines those locations. Send
   the script over SSH stdin or keep a copy outside the frozen repository.
   Save raw output in local `work/`, validate JSON and required sections, and
   preserve the previous good snapshot on SSH/parse failures. Retry one transient
   SSH timeout. A second failure is an access problem, not a training failure.
3. Inspect section-level errors and partial trailing JSONL rows. Snapshot files
   can advance during collection; never turn partial data into zeros. Inspect
   checkpoints, process state, evaluation progress and metric-file timestamps
   together before calling a job stalled. NCCL waiters during evaluation and the
   stopped sequential supervisor are expected.
4. Read both W&B runs back through the server's existing authenticated API:
   BTM `attentionx2023/btm-jepa/z78m0zk6`, flow
   `attentionx2023/btm-jepa/g0u3w723`. Preserve the check time and online state.
   Latest W&B summaries can lag local files and contain metrics from different
   optimizer steps; they are not selected-checkpoint comparisons.

The shared run group is
[pusht_pair_20261004](https://wandb.ai/attentionx2023/btm-jepa/groups/pusht_pair_20261004).
BTM owns GPUs 0–3 and flow owns 4–7. Both receive 23,830 optimizer updates,
10 epochs, batch 128 and four A800s. This matches training exposure and allocation;
FLOPs and elapsed GPU-hours differ. Keep the existing protocol fixed.

## Analysis and evidence

Install plotting dependencies in a separate reporting environment using
`scripts/monitoring/requirements.txt`, then run from the repository root:

```bash
python scripts/monitoring/analyze_campaign_report.py \
  --snapshot /path/to/validated-raw-snapshot.json \
  --out docs/research/pusht_pair_20261004/snapshots/UTC_TIMESTAMP
python scripts/monitoring/eta_estimator.py \
  /path/to/validated-raw-snapshot.json \
  --output docs/research/pusht_pair_20261004/snapshots/UTC_TIMESTAMP/eta.json
```

The analyzer checks paired protocols with the existing comparison implementation,
excludes incomplete three-offset cycles from complete-cycle comparisons, and
exports raw numeric histories, paired results, episode transitions and plots.
Compare the same optimizer step across methods. Report validation-selected
`best.pt` separately from the latest checkpoint, along with its selection step,
mean-success score, offset 100 result and checkpoint hash from its evaluation.

The ETA estimator models complete evaluation-end intervals as optimizer-step
time plus a three-offset evaluation cost. It includes the union of 1,000-step
and epoch-end evaluations. Its ±15% timing envelope is a planning sensitivity
range, not a confidence interval. Completion time includes remaining training
and scheduled validation; it excludes held-out testing and multi-seed studies.
Recheck after a process failure, large host-load change, or overdue projection.

For every update:

- Record exact collection time, frozen source/manifest/encoder hashes, active
  steps, checkpoint health, online metric check and ETA.
- Give matched per-offset success, paired uncertainty and selected-checkpoint
  results. One seed and 20 repeated validation cases per offset cannot establish
  a repeatable research gain. Repeated checkpoints are not independent trials.
- Log observed behaviors and failure-case episode/start IDs. Separate numerical
  evidence, causal hypotheses and unmeasured quantities. Lower latent losses,
  clamped boundary error or smaller mid-segment residuals are not task success.
- Interpret candidate variance as a small latent-space probe; it is not mode
  coverage. Interpret concurrent solver-batch latency descriptively. Do not call
  pre-CEM and post-execution errors a calibrated model-error comparison.
- List the next discriminating diagnostic and only then a possible improvement.
  Preserve the flow control. Do not tune on the reserved test split.
- Inspect the rendered chart and check the tables against numeric exports.

## Publication

Update `PROGRESS.md`, append a timestamped report under `reports/`, and add the
new compact numerical snapshot. Preserve older evidence. Only publish a new
snapshot when completed evaluations, a meaningful status/ETA change or a failure
adds information; timestamp-only rewrites are unnecessary. Exclude raw checkpoint
weights, full raw snapshots, credentials, internal host identifiers and unrelated
files. Stage explicit report/monitoring paths, inspect the diff, commit, push
`codex/server-launch-20261004` to origin without force, and verify the remote SHA.
Do not merge to main as part of this monitor.

The current source fixes were checked with 26 server regression tests, plus 13
allocation tests and 2 real-process concurrency tests. The reporting analyzer and
ETA estimator are checked against a real snapshot and independent calculations.
These checks do not imply that training meets the research target.

## Completion and notifications

The original sequential supervisor is deliberately stopped while its BTM child
continues. The concurrent controller owns flow, writes a genuine completion
marker after a successful trainer exit and final checkpoint check, then resumes
the original supervisor. That supervisor records BTM completion and skips flow.
`concurrency-status=complete` means flow finished and control was handed back;
whole-campaign completion needs both method completion markers and valid final
checkpoints at 23,830 updates. A finished BTM trainer can wait for that handoff.

Do not start, stop, restart, retune, or alter GPU jobs during monitoring. Report
failures with their last successful checkpoint, current logs and a recovery
recommendation. Stay quiet on unchanged state. Give the user concise progress
updates for new completed evaluations, changed conclusions, significant ETA
changes, failures or completion, with the GitHub report and W&B links.

After both runs finish, publish the final validation-based findings, selected
checkpoints, unresolved failure modes and prioritized future experiments. State
that held-out testing and multi-seed confirmation remain outstanding. Notify the
user, then pause this heartbeat using its automation ID. Desktop scheduling needs
the computer on and the app running; the server training itself is detached.
