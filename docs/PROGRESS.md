# Active campaign progress

## Current verified state

- **Running:** real MetaWorld data collection and V-JEPA feature caching on
  target_server_2, GPUs 0–3. The shared dataset must finish before full training.
- **Passed:** 29 server tests; actual four-rank CUDA/NCCL preflight, official encoder,
  repeatable simulator rendering, and two optimizer updates per architecture.
- **Active execution revision:** `61a5d73257f47c728fab6333bb0aab30b98acdc4`.
  The desktop reporting branch advances independently; do not pull it into the
  running execution checkout.
- **Continuity:** detached server supervisor, persistent logs/checkpoints/evaluation
  records, GitHub progress/evidence backups, and a 15-minute chat heartbeat.
- **No benchmark results yet:** no full model has completed training, and the
  frozen test comparison has not run. No improvement is claimed.

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

### Renderer fixed; corrected GPU preflight active

Mesa 26.2.4 / llvmpipe (LLVM 23.1.2) is installed inside the dedicated environment.
A ten-action MetaWorld reach rollout repeats pixel-exactly, with frame SHA-256
`ef2302cc4443026d0c2b88783ee0054f69efd74e70357ea468df8b98426234c9`.
`pip check` remains clean and all 29 tests passed again (11.76 s).

The failed campaign is archived at `attempt-1-egl-failed` under the server record.
The corrected campaign is registered from frozen source `61a5d73`, same scientific
configuration, with explicit software-renderer settings. Supervisor PID 2851417
started the four-GPU preflight. Its online launch check is
https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/reckez4q.
The saved `launch.sh`, `runtime-verified.json`, and `renderer-probe.json` record
actual deployment settings and packages. This is still wiring validation, not a
trained-model success result.

### Four-GPU gate passed; real preparation confirmed

`gpu_preflight.json` reports success on all four A800 ranks with PyTorch
2.8.0+cu126 / CUDA 12.6. Each rank encoded real observations to `[6,32,1024]`,
ran two optimizer updates for the world model and each distinct planner
architecture (joint flow with consistency, deterministic with consistency,
LeFlow adaptation, and HWM adaptation), and checked finite planner gradients.
This verification used wiring fixtures and is not a benchmark score.

The supervisor automatically advanced to `prepare_data`. At the direct cache
verification, 24 real episodes were complete. The inspected episode has
`z` and single-image hindsight goals `[101,32,1024]`, actions `[100,8]`, and finite
features. The early collection rows are the protocol's random-action portion;
their zero task success is expected and is not a learned-method result.

Online W&B readback independently confirms the corrected launch-check run is
saved under the correct execution revision. It is a finished setup check;
actual training runs will be created once data preparation completes.

Next autonomous step: finish all shared data/goal-screening rows and freeze the
manifest, then train world models and all registered planners for three seeds;
run validation during training and the 3,200-reset/model paired test campaign
after all models finish. All method results, including losses, will be reported.

### 2026-10-07 19:01 UTC heartbeat — preparation continues

The supervisor (PID 2851417), distributed launcher, and all four data workers
remain active. The execution checkout is clean and unchanged at `61a5d73`.
The captured snapshot contains 56 completed episodes, up from the first verified
24; all four ranks are making progress. These are still the scheduled random
assembly episodes, so their unsuccessful task outcomes are not benchmark scores.
Only GPUs 0–3 have campaign allocations (about 2.3 GiB each at this check).

No training checkpoint/run has been created yet; the shared dataset remains the
prerequisite. Fresh W&B API readback confirms the saved launch-check run remains
accessible under the correct code revision. No intervention or protocol change
was needed. Next: continue preparation, then let the supervisor start the
registered training stages. This is routine ongoing progress, not a new result.

### 2026-10-07 19:16 UTC heartbeat — expert collection underway

The supervisor, launcher and four workers remain active, with a clean execution
checkout at `61a5d73`. The captured snapshot has 193 completed assembly episodes
(up from 56). Collection has progressed from the initial random-action episodes
to the prescribed expert episodes; recent expert episodes report success. These
are collection-expert outcomes, not learned-planner evaluation scores.

Only GPUs 0–3 are allocated to the campaign; feature encoding was observed active
on the GPUs. Fresh online W&B readback succeeded. No training run or checkpoint
exists yet, and no test evaluation has started. No recovery or code/configuration
change was needed. Next: continue shared dataset preparation and automatic
transition to training.

### 2026-10-07 19:31 UTC heartbeat — preparation remains healthy

The snapshot contains 332 completed episodes, up from 193. The supervisor and
all four data workers remain active; recent assembly expert collection records
report successful task completion. These remain data-collection outcomes only.
The execution checkout is clean at `61a5d73`; the registered code and protocol
have not changed. GPUs 0–3 remain the only campaign allocation.

Fresh online W&B readback succeeded. No training runs/checkpoints or benchmark
evaluations exist yet. No intervention was needed. Continue the shared cache
and frozen goal manifest preparation before automatic training begins.

### 2026-10-07 19:46 UTC heartbeat

Preparation advanced from 332 to 470 completed episodes. The supervisor and all
four workers are alive; recent assembly expert episodes succeed. Source remains
clean at `61a5d73`, and only GPUs 0–3 are allocated. Fresh online W&B readback
passed; training runs/checkpoints remain at zero pending the shared dataset. A
follow-up file copy timed out during SSH handshake; the verified API JSON already
returned by the successful server check was saved directly to the reporting
checkout. No campaign intervention was needed. Continue preparation and the
automatic training queue.

Reporting recovery: the desktop snapshot helper now includes
`online-readback.json` in its single SSH response. Future checks should refresh
server/W&B health first and then run the snapshot helper; no separate file-copy
connection is needed. This reporting-only change does not modify the frozen
execution checkout. Python compilation passed.

### 2026-10-07 20:01 UTC heartbeat — first task cache complete

The snapshot contains 616 completed episodes: all 600 training assembly episodes
and 16 button-press-topdown episodes. The second task has begun its configured
random-action portion. The supervisor and four workers are active, execution
source remains clean at `61a5d73`, and allocation remains limited to GPUs 0–3.

Fresh W&B readback and the consolidated snapshot transfer both succeeded. There
are still no training runs/checkpoints or learned-method evaluation results.
No intervention was needed; continue the remaining shared data preparation and
the automatic training queue.

### 2026-10-07 20:16 UTC heartbeat

Preparation advanced from 616 to 756 completed episodes: 600 assembly and
156 button-press-topdown episodes. The latter has reached expert collection;
recent expert episodes report success. All four workers and the supervisor are
active, the execution checkout is clean at `61a5d73`, and GPU use remains on 0–3.

Fresh W&B readback passed; no training runs/checkpoints exist yet. Storage still
has approximately 101 GiB free on persistent storage and 2.2 TiB on scratch. No
intervention was needed. Continue preparation before automatic training.
