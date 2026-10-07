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
