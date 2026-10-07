# Active campaign progress

## 2026-10-07 — recovered interrupted work

User requested resumption after “Resume stream unavailable” and persistent GitHub
backups of code, progress, and experiment results. The new non-BTM work is intact
on `origin/research/joint-flow-metaworld`, recovered at `342f1eb` (four campaign
commits beginning at `3a40659`). `main` still contains the older flow/BTM setup.
The active branch includes model/data preparation, distributed training, paired
evaluation, comparison reporting, preflight, and bootstrap logic.

Direct SSH to `target_server_2` succeeds from this desktop chat. Initial inspection
showed eight NVIDIA A800 80 GiB GPUs idle. This campaign remains limited to four.
The older PushT/BTM campaign at `/home/mtxu/adam/LeFlow-experiments/20261004`
is historical, not the requested new experiment. No new campaign job has been
launched at this checkpoint; previous CPU verification claims are recorded in
`FLOW_EXPERIMENT.md` and will be checked on the actual server.

Next: deploy the recovered branch into a dedicated persistent server checkout,
verify dependencies and online W&B, run four-GPU preflight, then launch the frozen
data collection and matched training/evaluation campaign. Preserve source and
reports on GitHub while keeping the execution checkout fixed.

### Server setup underway

Dedicated execution checkout:
`/home/mtxu/adam/LeFlow-experiments/20261007-joint-flow/repo`.
Dedicated Conda environment: `/tmp/mtxu-flow-jepa-20261007/env`, cloned from the
working historical environment then pinned to `requirements-flow.txt` without
modifying the older campaign environment. Setup log is `setup.log` beside repo.

Server inspection recovered a previously reproduced CUDA/NCCL startup issue:
querying CUDA availability before selecting the rank's device broke this server's
runtime. The new launcher now explicitly requests CUDA and lazy module loading;
its distributed setup selects the rank device first. Added a regression test
that rejects early availability/count queries. Real NCCL verification is pending.

Storage: persistent `/home` has ~102 GiB free; scratch `/tmp` has ~2.2 TiB.
Use scratch for regenerable encoded episodes, and persistent directories for
model checkpoints, stage logs, and per-episode evaluation reports.

### Deployment fixes and reporting

The server's outbound GitHub clone failed with GnuTLS receive error (-110).
Transferred a verified Git bundle over the working SSH connection instead.
Pinned MetaWorld and V-JEPA source are likewise available as local transfers;
this preserves their configured revisions, without changing the methods.

The campaign now checks free space on the actual `episodes` storage target,
allowing a scratch symlink while retaining the campaign root on persistent disk.
The persistent campaign root will be
`/home/mtxu/adam/LeFlow-experiments/20261007-joint-flow/campaign`.
Only regenerable encoded episodes and downloaded encoder weights use scratch.

Added `scripts/snapshot_flow_campaign.py`: copies registered configuration,
preflight, preparation counts, all training metrics, validation reports, final
test reports and comparison results into a separate reporting checkout. It
copies no credentials or large model binaries. Unchanged reports are not
rewritten, so monitoring can commit/push only meaningful new evidence.

### Server verification — CPU gates passed

- All 29 `tests_flow` tests passed on the dedicated server environment (11.80 s),
  including reset/transition pairing, expert goal integrity, generated-plan
  gradients, exact resume, evaluation resume, and comparison guards.
- `pip check` reports no broken requirements. Removed inherited `ogbench` and
  `dm-control` from the new clone because they conflict with the pinned MuJoCo;
  neither is used by this campaign. The historical environment is untouched.
- Official V-JEPA 2.1 checkpoint (~5.15 GB) passed the configured SHA-256 check.
- Pinned V-JEPA source was transferred as a Git bundle and checked out on Linux.
- All eight GPUs were idle at the prelaunch recheck. The launcher explicitly
  restricts visible devices to 0,1,2,3 and the campaign limit to four.

The detached supervisor is launching from frozen source `448a30f`. It first
verifies online W&B and idle GPUs, then runs real CUDA/NCCL/encoder/renderer
preflight. Full training has not yet begun. Subsequent reports are published
from the separate desktop checkout, leaving this execution revision untouched.

### First GPU preflight failed safely: missing renderer

Online W&B launch-check succeeded:
https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/jqlr2md0.
Four CUDA/NCCL workers loaded the official encoder, but MetaWorld rendering failed:
EGL reported zero devices (`MUJOCO_EGL_DEVICE_ID` valid range 0..-1). The container
has compute libraries but no NVIDIA EGL graphics libraries. This is a renderer
provisioning failure, not an experimental result. The supervisor exited before
full data collection/training; the failure log is preserved on the server as
`egl-preflight-failure.log`, and the attempt metadata is copied under
`docs/reports/20261007-joint-flow/attempt-1`.

Provisioning Mesa software EGL in the dedicated environment. The launcher now
allows explicit renderer-device mapping independently of CUDA devices and records
renderer selection in the frozen campaign plan. This supports four GPU workers
sharing software EGL device 0 without requesting unavailable graphics devices.
The failed registered attempt will remain archived; the corrected attempt will
register its new code and renderer settings before collecting any data.

A 15-minute chat heartbeat is active (`continue-flow-jepa-campaign-and-preserve-results`)
to continue this campaign, preserve new evidence on origin, and notify only
meaningful progress, failure, completion, or a required action.

Reporting note: compact JSON under `docs/reports/**/runs/` is explicitly exempt
from the global `runs/` ignore rule, so training metrics and validation evidence
are included in routine commits. Model binaries remain ignored.
