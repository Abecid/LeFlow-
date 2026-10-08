# Active campaign progress

## Current verified state

- **Four-method migration complete:** only joint_flow_consistent, leflow_adapted,
  hwm_adapted and cem_long, exactly one seed (3072). No ablations, extra seeds,
  or automatic expansion. See `FIRST_PASS.md` for the evidence and scope.
- **Budget:** each learned model gets 7,200 optimization seconds or 20,000 updates
  on the same four GPUs, whichever ends first. One shared world plus three heads:
  up to 32 optimization GPU-hours, plus separately recorded preparation/evaluation
  and reported last-update overruns. Every controller gets 10 seconds per episode.
- **Matched data:** identical learned-head windows, shared encoder/fine world,
  four periodic 104-episode validation rounds, and 3,200 fixed test resets per
  method after training. Validation metrics are logged online to W&B.
- **Running:** supervisor PID 3126314 on target_server_2, limited to GPUs 0–3.
  The revised online launch check and actual four-rank GPU preflight passed;
  all 7,800 training episodes are cached. Validation candidate preparation is
  underway (114/1,300 at the 09:01 UTC heartbeat); test-goal preparation follows.
  All 2,381 episodes present at the scope migration were retained.
- **Passed:** the bulk server suite (36 tests) plus all 8 final targeted
  budget/training tests after the last changes; actual CUDA/NCCL, official encoder,
  repeatable renderer and two optimizer updates for each selected architecture.
- **Frozen execution revision:** `16747bb451915b2488c67f5d550a97c1b0411290`
  in server checkout `repo-first-pass`. `launch.sh` uses its execution config and
  the original `repo/config/flow_metaworld.json` only as `--data-config` for cache
  compatibility. Older supervisors are stopped and must not be restarted.
  The desktop reporting branch advances independently; never pull into the
  running execution checkout.
- **Continuity:** detached supervisor, persistent records, GitHub evidence backups,
  and the existing 15-minute heartbeat. W&B launch check:
  https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/jhc5g1dk.
- **No benchmark results yet:** shared preparation is still required before full
  model training and the frozen test comparison. No improvement is claimed.

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

### 2026-10-07 — four-method, bounded first comparison prepared

The user restricted the first pass to three major baselines plus our best
motivated proposal, one seed, matched compute/data and periodic evaluation.
Current choice: joint_flow_consistent, leflow_adapted, hwm_adapted, cem_long.
All deterministic/no-consistency ablations and short CEM are removed from the
execution configuration; no follow-up sweep is scheduled. The literature and
code review, including why the proposed method remains a hypothesis, is saved
in `docs/FIRST_PASS.md`. Recent primary sources reviewed: Planning Limits,
LeFlow, HWM, FF-JEPA, Qantara, Flow-JEPA and LeWAM.

Default first-pass budget: 7,200 optimization seconds or 20,000 updates per
learned model on the same four-GPU allocation. One world plus three heads gives
up to 32 optimization GPU-hours; shared data preparation and validation are
separate and logged. A user preference question offered 1/2/4-hour caps; no reply
had arrived before proceeding with the stated 2-hour default. The runtime limits
are checked at optimizer boundaries, preserving charged time across resume and
reporting overruns. Learning-rate and consistency warm-ups use budget progress.

Each method receives the same 10-second cumulative controller allowance per
episode and 200 primitive actions. Late actions are discarded and failures stay
in the denominator. Periodic evaluation uses the same 104 validation episodes
at four budget milestones, with online W&B and local logs. The final shared test
remains 3,200 resets per method (12,800 executions across four methods).

Found and corrected an input fairness issue before training: HWM previously
used short random/expert windows, while the generative heads used successful
expert long windows. All three learned heads now receive identical successful
expert trajectories/start positions/windows; HWM learns all macro transitions
from them. The shared fine world retains the common expert/random dataset.

Cache reuse is explicit: original data configuration drives ongoing collection;
collection/encoder/split fields must match the execution configuration. On
completion, preserve the original manifest and its digest, then register the new
execution protocol while retaining every episode hash, reset, goal and statistic.
No expensive feature data needs to be regenerated.

The first server CPU test pass passed 36 tests. The final budget/resume/deadline
checks are being verified before deployment. The live preparation checkout is
still frozen and collecting data. Migration remains in progress until the new
supervisor is registered and verified.

Final targeted verification passed: 8 budget/training tests, including retained
compute accounting on resume, late-action discard with failures retained, and
identical learned-head input windows. One SSH route timed out; the configured
Cloudflare fallback reached the same server. The corrected learning-rate warm-up
was included in these targeted checks. Ready to deploy the four-method revision.

### 2026-10-07 23:11 UTC — four-method migration deployed

Published execution revision `16747bb` and transferred a verified Git bundle to
new clean server checkout `repo-first-pass`. Its dry run registered only the four
selected methods, one seed, 7,200 optimization seconds/model, the shared 10-second
controller allowance, and the unchanged original data protocol. No training run
existed before switching. The old supervisor PID 3094037 and its preparation-only
process tree were stopped after checking process identities; unrelated jobs were
untouched. All 2,381 completed episodes were preserved.

Archive: `first-pass-scope-change-20261007` under the persistent server record.
The updated `launch.sh` starts supervisor PID 3126314 with the current execution
configuration and a separate original `--data-config`. `campaign/compute-scope-change.json`
records revisions, budgets, process switch and data preservation. Supervisor log:
`launcher-first-pass.log`. Fresh W&B API readback of run `jhc5g1dk` confirms the
new revision, four methods, seed 3072 and the two-hour optimization allowance.
GPU preflight is being checked before declaring the resumed collection healthy.

### 2026-10-07 23:13 UTC — GPU checks passed; migration complete

The new revision passed actual four-rank CUDA/NCCL preflight on GPUs 0–3. Every
rank loaded the official encoder, reproduced a real simulator reset, and ran two
optimizer updates with finite losses/gradients for the world model and all three
selected learned heads. The revised HWM full-window inputs and generated-plan
consistency objective passed. Peak allocated preflight memory was about 1.87 GiB
per rank. These are wiring checks, not trained-method benchmark results.

Supervisor PID 3126314 automatically advanced to `prepare_data`; four preparation
workers (3138302–3138305 at this check) are active. The existing 15-minute heartbeat
now points to `repo-first-pass`, enforces four methods/seed 3072/unchanged budgets,
and explicitly prohibits automatic follow-up ablations or expansion. No duplicate
monitor was created. Full training and final test evaluation remain pending shared
preparation. Next: finish the existing cache and goal screening, freeze the shared
manifest, train the shared world and three heads within their caps with periodic
online evaluation, then run the paired four-method test and analyze failures.

Live follow-up confirmed all four collection ranks writing new cache rows; the
completed cache advanced from 2,381 to 2,390 episodes after the switch. This
confirms actual preparation progress under the new supervisor. Full training
runs/checkpoints remain absent, as expected until shared preparation completes.
The compact snapshot now contains the current successful GPU-preflight report.
Intermittent SSH handshake failures were recovered using the configured fallback;
they did not interrupt the detached campaign.

### 2026-10-07 23:16 UTC heartbeat — first-pass scope verified

The current supervisor PID 3126314 and four preparation workers 3128857–3128860
are alive in `repo-first-pass`; its source is clean at frozen revision `16747bb`.
Campaign registration and fresh online W&B readback both confirm exactly
joint_flow_consistent, leflow_adapted, hwm_adapted and cem_long, seed 3072,
7,200 optimization seconds/model and 10 controller seconds/episode. Only GPUs
0–3 are allocated. No supervisor or scientific configuration was changed.

Preparation advanced from the post-migration snapshot of 2,390 to 2,414 completed
episodes: 600 each for assembly, button-press-topdown, coffee-button and dial-turn,
plus 14 door-close. The new door-close rows are in the registered random-action
portion; their unsuccessful collection outcomes are expected. Training runs and
checkpoints remain zero pending shared preparation. No recovery was needed.
Next: finish the shared cache/goal screening, then let the registered bounded
world/three-head training and four-method evaluation queue proceed.

The 23:01 heartbeat obeyed the then-active migration hold: read-only checks found
2,288 cached episodes and healthy preparation/W&B, without writing reports,
committing pending edits or changing the supervisor during that migration.

### 2026-10-07 23:31 UTC heartbeat

Preparation advanced from 2,414 to 2,549 completed episodes: 600 each for assembly,
button-press-topdown, coffee-button and dial-turn, plus 149 door-close. Recent
expert door-close collection reports success. Supervisor 3126314 and all four
workers remain active, with clean frozen execution revision `16747bb` and GPUs
0–3 only.

Campaign registration and fresh W&B online readback still match the four methods,
seed 3072, and unchanged optimization/controller limits. Training runs and
checkpoints remain zero while shared preparation continues. No intervention was
required; continue the registered bounded preparation/training/evaluation queue.

### 2026-10-07 23:46 UTC heartbeat

Preparation advanced from 2,549 to 2,686 completed episodes: 600 each for assembly,
button-press-topdown, coffee-button and dial-turn, plus 286 door-close. Recent
expert collection reports success. Supervisor 3126314 and all four preparation
workers remain active, execution source is clean at `16747bb`, and allocation
remains GPUs 0–3.

Fresh W&B online readback passed and agrees with the registered four methods,
seed 3072 and unchanged compute limits. Full training runs/checkpoints remain
zero pending shared preparation. No intervention was required. Continue the
registered bounded preparation, training and evaluation queue.

### 2026-10-08 00:01 UTC heartbeat

Preparation advanced from 2,686 to 2,822 completed episodes: 600 each for assembly,
button-press-topdown, coffee-button and dial-turn, plus 422 door-close. Recent
expert collection reports success. Supervisor 3126314 and all four workers are
active, execution source remains clean at `16747bb`, and allocation stays on
GPUs 0–3. Persistent storage has 101 GiB free; scratch has 2.2 TiB free.

Fresh W&B online readback passed; four methods, seed 3072, optimization allowance
and controller allowance remain unchanged. Training runs/checkpoints are still
zero pending shared preparation. No intervention was required; continue the
registered bounded preparation/training/evaluation queue.

### 2026-10-08 00:16 UTC heartbeat

Preparation advanced from 2,822 to 2,958 completed episodes: 600 each for assembly,
button-press-topdown, coffee-button and dial-turn, plus 558 door-close. Recent
expert collection reports success. Supervisor 3126314 and all four workers remain
active, execution source is clean at frozen revision `16747bb`, and only GPUs
0–3 are allocated.

Fresh online W&B readback confirms the registered four methods, seed 3072 and
unchanged optimization/controller limits. Training runs/checkpoints remain zero
pending shared preparation. No intervention was required; continue the bounded
preparation, training and evaluation queue without expanding scope.

### 2026-10-08 00:31 UTC heartbeat

Preparation advanced from 2,958 to 3,109 completed episodes: assembly,
button-press-topdown, coffee-button, dial-turn and door-close now have 600 each;
door-open has 109. Recent door-open rows are in the registered random-action
portion, where unsuccessful collection outcomes are expected. Supervisor 3126314
and all four workers remain active, execution source is clean at `16747bb`, and
only GPUs 0–3 are allocated.

Fresh W&B online readback confirms the four methods, seed 3072 and unchanged
compute limits. Full training runs/checkpoints remain zero pending shared
preparation. The primary SSH health check succeeded, but the subsequent snapshot
SSH connection exited 255; retry through configured `target_server_2_cf` succeeded.
No campaign process was restarted or changed. Continue the registered bounded
preparation/training/evaluation queue.

### 2026-10-08 00:46 UTC heartbeat

Preparation advanced from 3,109 to 3,233 completed episodes: 600 each for assembly,
button-press-topdown, coffee-button, dial-turn and door-close, plus 233 door-open.
Recent expert door-open collection reports success. Supervisor 3126314 and all
four workers remain active, execution source is clean at `16747bb`, and allocation
remains GPUs 0–3.

Fresh online W&B readback passed and confirms the registered four methods, seed
3072 and unchanged compute limits. The compact snapshot was retrieved through
the configured fallback route. Training runs/checkpoints remain zero while shared
preparation continues. No recovery intervention was required; continue the
registered bounded preparation/training/evaluation queue.

### 2026-10-08 01:01 UTC heartbeat

Preparation advanced from 3,233 to 3,379 completed episodes: 600 each for assembly,
button-press-topdown, coffee-button, dial-turn and door-close, plus 379 door-open.
Recent door-open expert rows include successes and an unsuccessful episode
(`train/door-open/00377`); collection failures remain recorded, and these are not
learned-planner evaluation results. Supervisor 3126314 and all four workers are
active with clean frozen source `16747bb` and GPUs 0–3 only.

Initial SSH handshakes timed out on both routes; a retry on `target_server_2_cf`
succeeded, followed by a successful snapshot. The recovered health check and
fresh online W&B readback confirm unchanged methods, seed and compute limits.
Training runs/checkpoints remain zero pending shared preparation. No campaign
process or configuration was changed. Continue the registered bounded queue.

### 2026-10-08 01:16 UTC heartbeat

Preparation advanced from 3,379 to 3,507 completed episodes: 600 each for assembly,
button-press-topdown, coffee-button, dial-turn and door-close, plus 507 door-open.
Recent expert collection reports success. Supervisor 3126314 and all four workers
remain active, execution source is clean at `16747bb`, and allocation stays on
GPUs 0–3.

Health checks and snapshot retrieval succeeded through `target_server_2_cf`.
Fresh W&B online readback confirms four methods, seed 3072 and unchanged compute
limits. Full training runs/checkpoints remain zero pending shared preparation.
No recovery intervention was required; continue the registered bounded queue.

### 2026-10-08 01:31 UTC heartbeat

Preparation advanced from 3,507 to 3,645 completed episodes: the first six tasks
through door-open have 600 each; drawer-close has 45. Recent drawer-close rows
are in the registered random-action portion and include both successes and
failures. Supervisor 3126314 and all four workers remain active, execution source
is clean at `16747bb`, and only GPUs 0–3 are allocated.

Health checks, online W&B readback and snapshot retrieval succeeded through the
configured fallback route. The four methods, seed 3072 and compute limits remain
unchanged. Training runs/checkpoints remain zero pending shared preparation.
No intervention was required; continue the registered bounded queue.

### 2026-10-08 01:46 UTC heartbeat

Preparation advanced from 3,645 to 3,792 completed episodes: the first six tasks
through door-open have 600 each; drawer-close has 192. Recent drawer-close expert
collection reports success. Supervisor 3126314 and all four workers remain
active, source is clean at frozen revision `16747bb`, and allocation stays on
GPUs 0–3.

Health checks and snapshot retrieval succeeded through the configured fallback.
Fresh online W&B readback confirms the four methods, seed 3072 and unchanged
compute limits. Training runs/checkpoints remain zero while shared preparation
continues. No intervention was required; continue the registered bounded queue.

### 2026-10-08 02:01 UTC heartbeat

Preparation advanced from 3,792 to 3,940 completed episodes: the first six tasks
through door-open have 600 each; drawer-close has 340. Recent expert collection
reports success. Supervisor 3126314 and all four workers remain active, frozen
source is clean at `16747bb`, and only GPUs 0–3 are allocated. Free space remains
101 GiB on persistent storage and 2.2 TiB on scratch.

Health checks, fresh online W&B readback and snapshot retrieval succeeded through
`target_server_2_cf`. Four methods, seed 3072 and compute limits remain unchanged.
Full training runs/checkpoints remain zero pending shared preparation. No
intervention was required; continue the registered bounded queue.

### 2026-10-08 02:16 UTC heartbeat

Preparation advanced from 3,940 to 4,084 completed episodes: the first six tasks
through door-open have 600 each; drawer-close has 484. Recent expert collection
reports success. Supervisor 3126314 and all four workers remain active, source
is clean at frozen revision `16747bb`, and only GPUs 0–3 are allocated.

Health checks and compact snapshot retrieval succeeded through the configured
fallback route. Fresh online W&B readback confirms the four methods, seed 3072
and unchanged compute limits. Full training runs/checkpoints remain zero pending
shared preparation. No intervention was required; continue the registered queue.

### 2026-10-08 02:31 UTC heartbeat

Preparation advanced from 4,084 to 4,231 completed episodes: the first seven tasks
through drawer-close have 600 each; drawer-open has 31. Recent drawer-open rows
are in the registered random-action portion, where unsuccessful outcomes are
expected. Supervisor 3126314 and all four workers remain active, frozen source
is clean at `16747bb`, and only GPUs 0–3 are allocated.

Health checks, fresh online W&B readback and compact snapshot retrieval succeeded
through the configured fallback. Four methods, seed 3072 and compute limits
remain unchanged. Full training runs/checkpoints remain zero pending shared
preparation. No intervention was required; continue the registered bounded queue.

### 2026-10-08 02:46 UTC heartbeat

Preparation advanced from 4,231 to 4,377 completed episodes: the first seven tasks
through drawer-close have 600 each; drawer-open has 177. Recent expert drawer-open
collection reports success. Supervisor 3126314 and all four workers remain
active, execution source is clean at frozen revision `16747bb`, and allocation
remains GPUs 0–3.

Health checks, fresh online W&B readback and snapshot retrieval succeeded through
the configured fallback. Four methods, seed 3072 and compute limits remain
unchanged. Full training runs/checkpoints remain zero pending shared preparation.
No intervention was required; continue the registered bounded queue.

### 2026-10-08 03:01 UTC heartbeat

Preparation advanced from 4,377 to 4,533 completed episodes: the first seven tasks
through drawer-close have 600 each; drawer-open has 333. Recent expert collection
reports success. Supervisor 3126314 and all four workers remain active, execution
source is clean at `16747bb`, and only GPUs 0–3 are allocated. Free space remains
101 GiB on persistent storage and 2.2 TiB on scratch.

The fallback-route health check and fresh W&B readback succeeded, confirming the
four methods, seed 3072 and unchanged compute limits. The subsequent snapshot
connection via `target_server_2_cf` exited 255; retry via `target_server_2`
succeeded. Full training runs/checkpoints remain zero pending shared preparation.
No campaign processes or configuration were changed. Continue the bounded queue.

### 2026-10-08 03:16 UTC heartbeat

Preparation advanced from 4,533 to 4,675 completed episodes: the first seven tasks
through drawer-close have 600 each; drawer-open has 475. Recent expert collection
reports success. Supervisor 3126314 and all four workers remain active, frozen
execution source is clean at `16747bb`, and allocation remains GPUs 0–3.

Fresh W&B online readback confirms four methods, seed 3072 and unchanged compute
limits. The primary snapshot SSH connection exited 255; retry through
`target_server_2_cf` succeeded. Full training runs/checkpoints remain zero pending
shared preparation. No campaign process or configuration was changed. Continue
the registered bounded preparation/training/evaluation queue.

### 2026-10-08 03:31 UTC heartbeat

Preparation advanced from 4,675 to 4,812 completed episodes: the first eight tasks
through drawer-open have 600 each; faucet-open has 12. Recent faucet-open rows
are in the registered random-action portion, where unsuccessful outcomes are
expected. Supervisor 3126314 and all four workers remain active, execution source
is clean at `16747bb`, and only GPUs 0–3 are allocated.

Health checks, fresh W&B online readback and compact snapshot retrieval succeeded
through the configured fallback. Four methods, seed 3072 and compute limits
remain unchanged. Full training runs/checkpoints remain zero pending shared
preparation. No intervention was required; continue the registered bounded queue.

### 2026-10-08 03:46 UTC heartbeat

Preparation advanced from 4,812 to 4,959 completed episodes: the first eight tasks
through drawer-open have 600 each; faucet-open has 159. Recent expert collection
reports success. Supervisor 3126314 and all four workers remain active, frozen
source is clean at `16747bb`, and only GPUs 0–3 are allocated.

The first health-check handshake via `target_server_2_cf` timed out; retry through
`target_server_2` succeeded, as did snapshot retrieval. Fresh W&B online readback
confirms four methods, seed 3072 and unchanged compute limits. Full training
runs/checkpoints remain zero pending shared preparation. No campaign process or
configuration was changed. Continue the registered bounded queue.

### 2026-10-08 04:01 UTC heartbeat

Preparation advanced from 4,959 to 5,094 completed episodes: the first eight tasks
through drawer-open have 600 each; faucet-open has 294. Recent expert collection
reports success. Supervisor 3126314 and all four workers remain active, execution
source is clean at `16747bb`, and only GPUs 0–3 are allocated. Free space remains
101 GiB on persistent storage and 2.2 TiB on scratch.

Health checks, fresh W&B online readback and compact snapshot retrieval succeeded
via `target_server_2`. Four methods, seed 3072 and compute limits remain unchanged.
Full training runs/checkpoints remain zero pending shared preparation. No
intervention was required; continue the registered bounded queue.

### 2026-10-08 04:16 UTC heartbeat

Preparation advanced from 5,094 to 5,254 completed episodes: the first eight tasks
through drawer-open have 600 each; faucet-open has 454. Recent expert collection
reports success. Supervisor 3126314 and all four workers remain active, execution
source is clean at `16747bb`, and only GPUs 0–3 are allocated.

Health checks, fresh W&B online readback (04:18 UTC) and compact snapshot retrieval
succeeded via `target_server_2`. Four methods, seed 3072 and compute limits remain
unchanged. Full training runs/checkpoints remain zero pending shared preparation.
No intervention was required; continue the registered bounded queue.

### 2026-10-08 04:31 UTC heartbeat

Preparation advanced from 5,254 to 5,379 completed episodes: the first eight tasks
through drawer-open have 600 each; faucet-open has 579. Recent expert collection
reports success. Supervisor 3126314 and all four workers remain active, frozen
execution source is clean at `16747bb`, and only GPUs 0–3 are allocated.

Health checks, fresh W&B online readback and compact snapshot retrieval succeeded
via `target_server_2`. Four methods, seed 3072 and compute limits remain unchanged.
Full training runs/checkpoints remain zero pending shared preparation. No
intervention was required; continue the registered bounded queue.

### 2026-10-08 04:46 UTC heartbeat

Preparation advanced from 5,379 to 5,518 completed episodes: the first nine tasks
through faucet-open have 600 each; handle-press has 118. Recent handle-press rows
include the registered random-action portion (mixed success/failure) and the
first successful expert episodes. These are collection outcomes, not learned
planner results. Supervisor 3126314 and all four workers remain active, frozen
source is clean at `16747bb`, and only GPUs 0–3 are allocated.

Health checks, fresh W&B online readback and compact snapshot retrieval succeeded
via `target_server_2`. Four methods, seed 3072 and compute limits remain unchanged.
Full training runs/checkpoints remain zero pending shared preparation. No
intervention was required; continue the registered bounded queue.

### 2026-10-08 05:01 UTC heartbeat

Preparation advanced from 5,518 to 5,659 completed episodes: the first nine tasks
through faucet-open have 600 each; handle-press has 259. Recent expert collection
reports success. Supervisor 3126314 and all four workers remain active, frozen
execution source is clean at `16747bb`, and only GPUs 0–3 are allocated.

Fresh W&B online readback confirms four methods, seed 3072 and unchanged compute
limits. The primary snapshot SSH connection exited 255; retry through
`target_server_2_cf` succeeded. Full training runs/checkpoints remain zero pending
shared preparation. No campaign process or configuration was changed. Continue
the registered bounded preparation/training/evaluation queue.

### 2026-10-08 05:16 UTC heartbeat

Preparation advanced from 5,659 to 5,796 completed episodes: the first nine tasks
through faucet-open have 600 each; handle-press has 396. Recent expert collection
reports success. Supervisor 3126314 and all four workers remain active, execution
source is clean at `16747bb`, and only GPUs 0–3 are allocated.

The first health-check handshake via `target_server_2_cf` timed out; retry through
`target_server_2` succeeded, as did snapshot retrieval. Fresh W&B online readback
confirms four methods, seed 3072 and unchanged compute limits. Full training
runs/checkpoints remain zero pending shared preparation. No campaign process or
configuration was changed. Continue the registered bounded queue.

### 2026-10-08 05:31 UTC heartbeat

Preparation advanced from 5,796 to 5,928 completed episodes: the first nine tasks
through faucet-open have 600 each; handle-press has 528. Recent expert collection
reports success. Supervisor 3126314 and all four workers remain active, frozen
execution source is clean at `16747bb`, and only GPUs 0–3 are allocated.

Health checks, fresh W&B online readback and compact snapshot retrieval succeeded
via `target_server_2`. Four methods, seed 3072 and compute limits remain unchanged.
Full training runs/checkpoints remain zero pending shared preparation. No
intervention was required; continue the registered bounded queue.

### 2026-10-08 05:46 UTC heartbeat

Preparation advanced from 5,928 to 6,067 completed episodes: the first ten tasks
through handle-press have 600 each; pick-place has 67. Recent pick-place rows are
in the registered random-action portion, where unsuccessful outcomes are
expected. Supervisor 3126314 and all four workers remain active, frozen execution
source is clean at `16747bb`, and only GPUs 0–3 are allocated.

Health checks, fresh W&B online readback and compact snapshot retrieval succeeded
via `target_server_2`. Four methods, seed 3072 and compute limits remain unchanged.
Full training runs/checkpoints remain zero pending shared preparation. No
intervention was required; continue the registered bounded queue.

### 2026-10-08 05:49–05:55 UTC — storage and throughput audit

User requested the dataset definition, cache size, remaining storage, ETA and
whether processing maximizes throughput without sacrificing outputs. Live SSH
inspection found 6,088 dense training episodes occupying 76.76 GiB, a recent rate
of 552/hour (1,691 over three hours), 2.108 TiB free on scratch and 100.59 GiB free
on persistent storage. The whole campaign scratch tree occupied about 90.03 GiB
including its environment/encoder assets. Source and running protocol were unchanged.

This is locally generated MetaWorld v3 data: 7,800 train episodes across 13 tasks,
1,300 validation candidates selecting 650, and 6,400 test candidates selecting
3,200 across 16 tasks. All candidates are cached, so 15,500 files are prepared;
9,100 carry dense features. A train/validation episode averages 13.54 MB with two
101x32x1024 float16 feature arrays, actions, and initial/goal RGB. Test candidates
store only initial/goal RGB, goal features, success flags and metadata. Measured
component sizes project 116–120 GiB for all cache files and about 130–135 GiB for
campaign scratch including existing assets. Checkpoints/logs use persistent storage.

The current pipeline is NOT maximally optimized. Four workers use encoder batches
of two; each worker runs software-rendered collection, encoding, and writing
serially. No producer/consumer overlap exists. Thirty utilization samples averaged
14–29% per GPU, with 20–25 idle samples out of 30 and about 2.3 GiB used out of
80 GiB. CPU-only full-episode probes took 21.13 seconds for reach and 20.75 seconds
for assembly (201 frames each). The initial three-task probe hit its 55-second
limit without a timing result; only the successful separate probes inform the ETA.
Rendering is the primary measured bottleneck; increasing batch size alone will
not remove it. Goal-only test screening also renders every frame it later discards.

At the unchanged implementation/rate, roughly 3.1 hours remain for the training
cache, 5.5 hours for all dense train/validation candidates, and approximately
15–17 hours for all required preparation including test-goal candidates, excluding
model training/evaluation. This is a rough extrapolation; task-dependent times,
goal-screening success and interruptions can change it. Lossless improvement would
require pipelining more CPU render producers with GPU encoding/writes, batching
benchmarks and output-equivalence checks, and checking whether discarded test
rendering can be eliminated without changing retained images. Larger batches can
change numerical results; do not declare bitwise equivalence without testing.
No live code, model precision, dataset size, method scope or budget was changed by
this audit. Evidence: `docs/reports/20261007-joint-flow/storage-throughput-audit.json`.

### 2026-10-08 06:01 UTC heartbeat

Preparation advanced from the previous heartbeat's 6,067 to 6,210 completed
episodes: the first ten tasks through handle-press have 600 each; pick-place has
210. Recent expert collection reports success. Supervisor 3126314 and all four
workers remain active, frozen source is clean at `16747bb`, and only GPUs 0–3
are allocated. The intervening storage/throughput audit is preserved separately;
no throughput implementation change or scope expansion was made.

Health checks, fresh W&B online readback and compact snapshot retrieval succeeded
via `target_server_2`. Four methods, seed 3072 and compute limits remain unchanged.
Full training runs/checkpoints remain zero pending shared preparation. No
intervention was required; continue the registered bounded queue.

### 2026-10-08 06:16 UTC heartbeat

Preparation advanced from 6,210 to 6,354 completed episodes: the first ten tasks
through handle-press have 600 each; pick-place has 354. Recent expert collection
reports success. Supervisor 3126314 and all four workers remain active, frozen
execution source is clean at `16747bb`, and only GPUs 0–3 are allocated.

Health checks, fresh W&B online readback and compact snapshot retrieval succeeded
via `target_server_2`. Four methods, seed 3072 and compute limits remain unchanged.
Full training runs/checkpoints remain zero pending shared preparation. No
intervention was required; continue the registered bounded queue.

### 2026-10-08 06:31 UTC heartbeat

Preparation advanced from 6,354 to 6,496 completed episodes: the first ten tasks
through handle-press have 600 each; pick-place has 496. Recent expert collection
reports success. Supervisor 3126314 and all four workers remain active, frozen
execution source is clean at `16747bb`, and only GPUs 0–3 are allocated.

Health checks, fresh W&B online readback and compact snapshot retrieval succeeded
via `target_server_2`. Four methods, seed 3072 and compute limits remain unchanged.
Full training runs/checkpoints remain zero pending shared preparation. No
intervention was required; continue the registered bounded queue.

### 2026-10-08 06:46 UTC heartbeat

Preparation advanced from 6,496 to 6,650 completed episodes: the first eleven
tasks through pick-place have 600 each; plate-slide has 50. Recent plate-slide
rows are in the registered random-action portion, where unsuccessful outcomes
are expected. Supervisor 3126314 and all four workers remain active, frozen
execution source is clean at `16747bb`, and only GPUs 0–3 are allocated.

The first W&B API check could not connect to verify the token; a fresh retry at
06:48 UTC succeeded without changing credentials or campaign processes. Online
readback confirms four methods, seed 3072 and unchanged compute limits. Snapshot
retrieval via `target_server_2_cf` succeeded. Full training runs/checkpoints remain
zero pending shared preparation. Continue the registered bounded queue.

### 2026-10-08 07:01 UTC heartbeat

Preparation advanced from 6,650 to 6,780 completed episodes: the first eleven
tasks through pick-place have 600 each; plate-slide has 180. Recent expert
collection reports success. Supervisor 3126314 and all four workers remain
active, frozen execution source is clean at `16747bb`, and only GPUs 0–3 are
allocated.

Health checks, fresh W&B online readback and compact snapshot retrieval succeeded
via `target_server_2_cf`. Four methods, seed 3072 and compute limits remain
unchanged. Full training runs/checkpoints remain zero pending shared preparation.
No intervention was required; continue the registered bounded queue.

### 2026-10-08 07:16 UTC heartbeat

Preparation advanced from 6,780 to 6,926 completed episodes: the first eleven
tasks through pick-place have 600 each; plate-slide has 326. Recent expert
collection reports success. Supervisor 3126314 and all four workers remain
active, frozen source is clean at `16747bb`, and only GPUs 0–3 are allocated.
Free space remains 101 GiB on persistent storage and 2.1 TiB on scratch.

Health checks, fresh W&B online readback and compact snapshot retrieval succeeded
via `target_server_2_cf`. Four methods, seed 3072 and compute limits remain
unchanged. Full training runs/checkpoints remain zero pending shared preparation.
No intervention was required; continue the registered bounded queue.

### 2026-10-08 07:31 UTC heartbeat

Preparation advanced from 6,926 to 7,069 completed episodes: the first eleven
tasks through pick-place have 600 each; plate-slide has 469. Recent expert
collection reports success. Supervisor 3126314 and all four workers remain
active, frozen execution source is clean at `16747bb`, and only GPUs 0–3 are
allocated.

The first health-check handshake via `target_server_2_cf` timed out; retry through
`target_server_2` succeeded, as did snapshot retrieval. Fresh W&B online readback
confirms four methods, seed 3072 and unchanged compute limits. Full training
runs/checkpoints remain zero pending shared preparation. No campaign process or
configuration was changed. Continue the registered bounded queue.

### 2026-10-08 07:46 UTC heartbeat

Preparation advanced from 7,069 to 7,209 completed episodes: the first eleven
tasks through pick-place have 600 each; plate-slide has 599 and reach has 10.
Workers are crossing into the final training-data task; reach's initial
random-action rows are unsuccessful as expected. Validation and test-goal
preparation still follow. Supervisor 3126314 and all four workers remain active,
frozen source is clean at `16747bb`, and only GPUs 0–3 are allocated.

Health checks, fresh W&B online readback and compact snapshot retrieval succeeded
via `target_server_2`. Four methods, seed 3072 and compute limits remain unchanged.
Full training runs/checkpoints remain zero pending shared preparation. No
intervention was required; continue the registered bounded queue.

### 2026-10-08 08:01 UTC heartbeat

Preparation advanced from 7,209 to 7,345 completed episodes: the first twelve
tasks through plate-slide have 600 each; reach has 145. Recent reach expert
collection reports success. Supervisor 3126314 and all four workers remain
active, frozen execution source is clean at `16747bb`, and only GPUs 0–3 are
allocated. Validation and test-goal preparation still follow the training cache.

Health checks, fresh W&B online readback and compact snapshot retrieval succeeded
via `target_server_2`. Four methods, seed 3072 and compute limits remain unchanged.
Full training runs/checkpoints remain zero pending shared preparation. No
intervention was required; continue the registered bounded queue.

### 2026-10-08 08:16 UTC heartbeat

Preparation advanced from 7,345 to 7,489 completed episodes: the first twelve
tasks through plate-slide have 600 each; reach has 289. Recent expert collection
reports success. Supervisor 3126314 and all four workers remain active, frozen
execution source is clean at `16747bb`, and only GPUs 0–3 are allocated.
Validation and test-goal preparation still follow the training cache.

Health checks, fresh W&B online readback and compact snapshot retrieval succeeded
via `target_server_2`. Four methods, seed 3072 and compute limits remain unchanged.
Full training runs/checkpoints remain zero pending shared preparation. No
intervention was required; continue the registered bounded queue.

### 2026-10-08 08:31 UTC heartbeat

Preparation advanced from 7,489 to 7,631 completed episodes: the first twelve
tasks through plate-slide have 600 each; reach has 431. Recent expert collection
reports success. Supervisor 3126314 and all four workers remain active, frozen
execution source is clean at `16747bb`, and only GPUs 0–3 are allocated.
Validation and test-goal preparation still follow the training cache.

Health checks, fresh W&B online readback and compact snapshot retrieval succeeded
via `target_server_2`. Four methods, seed 3072 and compute limits remain unchanged.
Full training runs/checkpoints remain zero pending shared preparation. No
intervention was required; continue the registered bounded queue.

### 2026-10-08 08:46 UTC heartbeat

Preparation advanced from 7,631 to 7,774 completed episodes: the first twelve
tasks through plate-slide have 600 each; reach has 574. Recent expert collection
reports success. There are 26 training-cache episodes remaining at this snapshot;
validation and test-goal preparation still follow. Supervisor 3126314 and all
four workers remain active, frozen source is clean at `16747bb`, and only GPUs
0–3 are allocated.

Health checks, fresh W&B online readback and compact snapshot retrieval succeeded
via `target_server_2`. Four methods, seed 3072 and compute limits remain unchanged.
Full training runs/checkpoints remain zero pending shared preparation. No
intervention was required; continue the registered bounded queue.

### 2026-10-08 09:01 UTC heartbeat — training cache complete

All 7,800 registered training episodes are now cached: 600 for each of the 13
training tasks. Total preparation advanced from 7,774 to 7,914 files, including
114 validation candidates (assembly 100, button-press-topdown 14). Validation
candidate preparation and the 6,400 test-goal candidates remain before full model
training. These collection outcomes are not learned-planner validation results.

Supervisor 3126314 and all four workers remain active, frozen execution source
is clean at `16747bb`, and only GPUs 0–3 are allocated. Health checks, fresh W&B
online readback and snapshot retrieval succeeded via `target_server_2`. Four
methods, seed 3072 and compute limits remain unchanged. Full training runs and
checkpoints remain zero. No recovery was required; continue the registered
preparation queue, then bounded model training and evaluation.

### 2026-10-08 09:16 UTC heartbeat

Preparation advanced from 7,914 to 8,058 completed files. All 7,800 training
episodes remain cached; validation candidates increased from 114 to 258:
assembly 100, button-press-topdown 100 and coffee-button 58. Recent expert
collection reports success. These are data-preparation outcomes, not trained
planner validation scores. Test-goal preparation follows validation candidates.

Supervisor 3126314 and all four workers remain active, frozen source is clean
at `16747bb`, and only GPUs 0–3 are allocated. Health checks, fresh W&B online
readback and snapshot retrieval succeeded via `target_server_2`. Four methods,
seed 3072 and compute limits remain unchanged. Full training runs/checkpoints
remain zero. No intervention was required; continue the registered bounded queue.

### 2026-10-08 09:31 UTC heartbeat

Preparation advanced from 8,058 to 8,202 completed files. All 7,800 training
episodes remain cached; validation candidates increased from 258 to 402:
assembly, button-press-topdown and coffee-button have 100 each; dial-turn has
95 and door-close has 7. Recent expert collection reports success. Test-goal
preparation still follows validation candidates; no planner scores exist yet.

Supervisor 3126314 and all four workers remain active, frozen source is clean
at `16747bb`, and only GPUs 0–3 are allocated. Health checks, fresh W&B online
readback and snapshot retrieval succeeded via `target_server_2`. Four methods,
seed 3072 and compute limits remain unchanged. Full training runs/checkpoints
remain zero. No intervention was required; continue the registered bounded queue.
