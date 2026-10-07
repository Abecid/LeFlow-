# Active campaign progress

## Current verified state

- **User correction applied:** exactly one training seed, 3072. Seeds 3073/3074
  are removed from execution. Use `repo-single-seed` and the updated `launch.sh`.

- **Running:** corrected single-seed supervisor on target_server_2, limited to
  GPUs 0–3. The new GPU preflight passed and shared cache preparation resumed;
  2,128 completed episodes were retained at the switch.
- **Passed:** 31 server tests; actual four-rank CUDA/NCCL preflight, official encoder,
  repeatable simulator rendering, and two optimizer updates per architecture.
- **Active execution revision:** `06c5d02e542bfedf3772cc1575fd0e473b618dc7`
  in server checkout `repo-single-seed`. The old `repo` is preserved only for
  its original cache-compatible data configuration; its three-seed supervisor
  is stopped and must not be restarted.
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

### 2026-10-07 20:31 UTC heartbeat

Preparation advanced from 756 to 894 completed episodes (600 assembly,
294 button-press-topdown). The supervisor and all four workers remain active;
recent expert collection episodes succeed. Source is clean at `61a5d73`, and
only GPUs 0–3 are allocated. Fresh W&B readback passed; training runs and
checkpoints remain at zero. No intervention or protocol change was needed.
Continue shared preparation before the automatic training stages.

### 2026-10-07 20:46 UTC heartbeat

Preparation advanced from 894 to 1037 completed episodes (600 assembly,
437 button-press-topdown). All four workers and the supervisor remain active;
recent collection-expert episodes succeed. Execution source is clean at
`61a5d73`, allocation remains on GPUs 0–3, and fresh online W&B readback passed.
No training run/checkpoint exists yet. No intervention was needed; continue
shared dataset preparation and the automatic training queue.

### 2026-10-07 21:01 UTC heartbeat

Preparation advanced from 1037 to 1179 completed episodes (600 assembly,
579 button-press-topdown). All four workers and the supervisor remain active;
recent expert collection succeeds. The execution checkout is clean at `61a5d73`,
and allocation remains limited to GPUs 0–3. Fresh W&B readback succeeded; there
are no training runs/checkpoints yet. No intervention was required. Continue
shared preparation before the automatic training and evaluation stages.

### 2026-10-07 21:16 UTC heartbeat — second task cache complete

The snapshot has 1320 completed episodes: 600 assembly, 600
button-press-topdown, and 120 coffee-button. Collection has moved to the third
training task. The inspected coffee-button log rows were from its prescribed
random-action portion; their failures are not learned-planner evaluation results.

The supervisor and all four workers remain active, source is clean at `61a5d73`,
and allocation is limited to GPUs 0–3. Fresh W&B readback passed. Training
runs/checkpoints remain at zero until the shared preparation finishes. No
intervention was needed; continue the registered preparation/training queue.

### 2026-10-07 21:31 UTC heartbeat

Preparation advanced from 1320 to 1463 completed episodes: 600 each for
assembly and button-press-topdown, plus 263 coffee-button. Recent coffee-button
expert collection reports success. The supervisor and all four workers remain
active, source is clean at `61a5d73`, and allocation stays on GPUs 0–3.

Fresh W&B readback passed; no training runs/checkpoints exist yet. No intervention
was required. Continue the shared preparation and automatic training queue.

### 2026-10-07 21:46 UTC heartbeat

Preparation advanced from 1463 to 1603 completed episodes: 600 each for
assembly and button-press-topdown, plus 403 coffee-button. Recent expert collection
reports success. The supervisor and all four workers are active, execution
source is clean at `61a5d73`, and GPU allocation remains 0–3.

Fresh W&B readback succeeded; training runs/checkpoints remain at zero pending
shared data preparation. No intervention was required. Continue the registered
preparation and automatic training queue.

### 2026-10-07 22:01 UTC heartbeat

Preparation advanced from 1603 to 1754 completed episodes: 600 each for
assembly and button-press-topdown, plus 554 coffee-button. Recent expert collection
reports success; this is collection evidence, not learned-planner performance.
The supervisor and all four workers remain active, execution source is clean at
`61a5d73`, and allocation remains on GPUs 0–3.

Fresh online W&B readback passed. The finished run is the launch check; training
runs and checkpoints remain at zero while shared preparation continues. No
intervention was required. Continue the registered preparation/training queue.

### 2026-10-07 22:16 UTC heartbeat

Preparation advanced from 1754 to 1885 completed episodes: assembly,
button-press-topdown and coffee-button now have 600 each; dial-turn has 85.
Recent dial-turn collection is in the registered random-action portion, so
unsuccessful episodes at these indices are expected. All four workers and the
supervisor remain alive, with clean frozen source `61a5d73` and GPUs 0–3 only.

Online W&B readback passed; training runs/checkpoints remain zero until shared
preparation finishes. Free space remains 101 GiB on the persistent volume and
2.2 TiB on scratch. No intervention was required; continue the registered queue.

### 2026-10-07 22:31 UTC heartbeat

Preparation advanced from 1885 to 2028 completed episodes: 600 each for assembly,
button-press-topdown and coffee-button, plus 228 dial-turn. Recent dial-turn
expert collection reports success. The supervisor and all four workers are
active, source remains clean at frozen revision `61a5d73`, and only GPUs 0–3
are allocated.

Fresh W&B online readback passed. Training runs/checkpoints remain zero while
shared preparation continues; the finished W&B run is only the launch check.
No intervention was required. Continue the registered preparation/training queue.

### 2026-10-07 22:36 UTC — continuation verified from desktop

The continuation chat reconnected directly to `target_server_2` and recovered
this active non-BTM campaign instead of launching a duplicate. The first two SSH
handshakes timed out; retry succeeded through the existing configured route.
No credential, SSH configuration, campaign code, or protocol changes were needed.

Supervisor PID 2851417 and all four preparation workers are alive. The execution
checkout is clean at `61a5d73`; campaign GPU allocation remains 0–3. The new snapshot
contains 2,069 completed training episodes: 600 each for assembly,
button-press-topdown and coffee-button, plus 269 dial-turn. Persistent storage has
101 GiB free and scratch has 2.2 TiB free.

Online W&B readback passed at 22:35:54 UTC. The W&B run `reckez4q` is a finished
launch check; actual training runs and checkpoints are still zero. The campaign
will finish shared feature preparation and goal screening before training the
registered methods. Training and the paired 3,200-reset/model test evaluation
remain queued, with three training seeds and validation-only selection.

The existing 15-minute monitor in the “Resume GPU baseline comparison” chat is
active and remains the sole campaign monitor. It preserves subsequent progress,
recovers failures within the authorized four-GPU limit, and reports meaningful
changes. Next step: complete shared data preparation; allow the detached
supervisor to advance automatically into world-model and planner training.

### 2026-10-07 22:42 UTC — user correction: one training seed

The user explicitly rejected multiple training seeds. The campaign now registers
`execution_seeds: [3072]` and uses exactly that seed for all training, evaluation,
and comparison stages. Seeds 3073 and 3074 are no longer queued. The eight method
entries include our method, three ablations, LeFlow/HWM adaptations, and short/
long CEM; no methods were removed without user instruction.

Code `06c5d02` passed all 31 server tests. Added checks that a one-seed execution
can reuse an unchanged historical data protocol, that comparison rejects extra
seed reports, and that uncertainty is labeled as paired-reset uncertainty only.
Fresh default configurations now also contain just seed 3072.

The actual supervisor was replaced before any full training run existed. Its
old process tree was stopped by verified PID identities; 2,128 completed cache
files were retained. The original registration and launch metadata are archived
at `/home/mtxu/adam/LeFlow-experiments/20261007-joint-flow/seed-scope-change-20261007`.
No completed data was re-encoded or reselected. A first administrative command
failed on an unavailable optional process library before changing anything; the
successful switch used only the standard library and checked process identities.

New frozen execution checkout: `repo-single-seed` at `06c5d02` under the same
server record. New supervisor PID: 3094037. Launcher: the same `launch.sh`, now
with `--seed 3072 --config "$record/repo/config/flow_metaworld.json"`. The original
configuration is retained only to match existing cache fingerprints; the explicit
single execution seed supersedes its historical list. `campaign/seed-scope-change.json`
records this distinction and preservation evidence. New supervisor log:
`launcher-single-seed.log`. Do not change either frozen checkout during execution.

The existing heartbeat now explicitly prohibits extra training seeds and points
to the new checkout. Next: verify resumed preparation after GPU checks, then allow
one world model and six learned planners to train; CEM uses the shared world
without separate planner training. Final paired evaluation remains 3,200 resets
per method, 25,600 executions across eight methods, with validation-only selection.
No confidence interval will claim to measure variation across training runs.

Live verification: the new supervisor acquired GPUs 0–3 and started all four
GPU-preflight workers from the new frozen checkout. Fresh W&B API readback of
launch-check run `l2sux44s` confirms execution seed `[3072]` and code `06c5d02`:
https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/l2sux44s.
The monitor can continue normally; the seed migration is complete. Full training
remains pending shared preparation, and there are still no benchmark results.

### 2026-10-07 22:46 UTC heartbeat — single-seed preparation resumed

The new four-rank GPU preflight passed. Supervisor PID 3094037 and preparation
workers 3096481–3096484 are active in `repo-single-seed`, clean at `06c5d02`.
Both the process arguments and campaign registration confirm only execution seed
3072; GPUs 0–3 remain the sole allocation. The old supervisor was not restarted.
The supervisor log is at the record root, `20261007-joint-flow/launcher-single-seed.log`.

The retained cache advanced from 2,128 to 2,148 episodes: 600 each for assembly,
button-press-topdown and coffee-button, plus 348 dial-turn. Fresh online readback
of W&B launch check `l2sux44s` confirms `[3072]` and the new revision. Training runs
and checkpoints remain zero pending shared preparation. No recovery intervention
was needed. Continue the one-seed training queue after preparation; final paired
reset intervals must not claim variation across independent training runs.
