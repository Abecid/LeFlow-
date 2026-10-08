# Active campaign progress

## Current verified state

- **Throughput migration complete; training is running.** All 15,500 candidate
  cache files are prepared. The frozen manifest selects 7,800 training, 650
  validation and 3,200 test episodes. No full comparison result exists yet.
- **Shared world complete:** `world_3072` stopped at its 20,000-update cap with
  all four validations complete. Optimization used 2,653.535 seconds (2.94837
  GPU-hours); validation used 1,780.427 seconds (1.97825 GPU-hours), separately
  accounted. W&B verified the run finished and both checkpoints are preserved:
  https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/fbcl549w.
- **Joint-flow complete:** `joint_flow_consistent_3072` stopped at its
  20,000-update cap with all four validations complete. Optimization used
  5,498.695 seconds (6.10966 GPU-hours); validation used 2,102.580 seconds
  (2.33620 GPU-hours), separately accounted. W&B verified the run finished:
  https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/32710958.
- **LeFlow fourth validation active:** `leflow_adapted_3072` reached its
  20,000-update cap, charging 5,371.992 optimization seconds (5.96888 GPU-hours),
  plus 1,408.647 seconds (1.56516 GPU-hours) for three completed validations.
  The active round is charged at completion. It uses the same world hash,
  manifest, protocol, code, seed and four GPUs as joint-flow. W&B readback at
  23:47 UTC confirmed step 19,950 and three completed validations online:
  https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/9jhufn8j.
- **LeFlow validation:** the first three rounds achieved 26/104 (25.00%),
  20/104 (19.23%) and 25/104 (24.04%) successes. All 79 third-round timeouts
  remain failures. Against round two, nine previous failures succeeded and four
  successes became timeouts. The step-5,000 checkpoint remains selected at
  26/104; against joint-flow's selected 23/104, 17 cases succeeded under both,
  nine only under LeFlow and six only under joint-flow. LeFlow's last validation
  and HWM remain pending; this is not a final ranking or test result.
- **Joint-flow validation:** the four rounds achieved 23/104 (22.12%), 17/104
  (16.35%), 16/104 (15.38%) and 19/104 (18.27%) successes. All 85 fourth-round
  timeouts remain failures. The registered selection rule retains the step-5,000
  best checkpoint; its actual step and SHA-256 are verified. CEM using the same
  selected world achieved 7/104 (6.73%); all joint-flow rounds retained those
  successes. These are validation results; LeFlow/HWM and final tests are pending.
- **CEM validation results:** CEM succeeded on 10/104 cases (9.62%) at step 5,000,
  9/104 (8.65%) at step 10,000, and 7/104 (6.73%) at both 15,000 and 20,000.
  Round four had drawer-close 5/8 and handle-press 2/8; the other 11 tasks had
  zero successes. All 97 failures exhausted the 10-second controller allowance
  and remain in the results. World prediction loss improved without improving
  task success. These are validation results; final testing is still sealed.
- **Lossless throughput changes:** 48 CPU producers, encoder/goal batches of 64,
  replay-based two-image test collection, overlapped writes and parallel ordered
  cache summarization. Final 1,270 files took about 42.2 seconds of worker elapsed
  time (approximately 30 files/second). See `THROUGHPUT.md` for measured limits.
- **Preservation verified:** all 14,230 pre-existing cache file SHA-256 hashes
  matched again after optimized preparation completed. No cases, image resolution,
  numerical precision, model weights, split definitions or screening rules changed.
- **Fixed experiment:** only joint_flow_consistent, leflow_adapted, hwm_adapted
  and cem_long; one seed, 3072; no ablations or automatic expansion. Each learned
  model gets 7,200 optimization seconds or 20,000 updates on the same four GPUs,
  whichever ends first. Shared preparation/evaluation are accounted separately.
  Periodic evaluation uses 104 fixed cases; final testing uses 3,200 per method.
- **Passed:** 43 server tests; 20 real trajectory/image equivalence cases covering
  all 16 tasks; two end-to-end cache checks with exact arrays/statistics; actual
  four-rank CUDA/NCCL/encoder/renderer/planner-gradient preflight.
- **Frozen execution:** `56419ed14cf127d4b7a8ab09a9d68e6681d2edc0` in server
  `repo-throughput`; supervisor PID 142087; launcher `launch.sh`; supervisor log
  `launcher-throughput.log`. `repo/config/flow_metaworld.json` is only the original
  `--data-config`. Older supervisors are stopped and must not be restarted.
- **Continuity:** the existing 15-minute monitor now follows the optimized run.
  Persistent results stay on the server; compact evidence is published from this
  separate desktop checkout. Never pull reports into the running source checkout.

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

### 2026-10-08 09:46 UTC heartbeat

Preparation advanced from 8,202 to 8,339 completed files. All 7,800 training
episodes remain cached; validation candidates increased from 402 to 539:
the first five tasks through door-close have 100 each; door-open has 39. Recent
expert collection reports success. Test-goal preparation still follows validation
candidates; no trained-planner validation results exist yet.

Supervisor 3126314 and all four workers remain active, frozen source is clean
at `16747bb`, and only GPUs 0–3 are allocated. Health checks, fresh W&B online
readback and snapshot retrieval succeeded via `target_server_2`. Four methods,
seed 3072 and compute limits remain unchanged. Full training runs/checkpoints
remain zero. No intervention was required; continue the registered bounded queue.

### 2026-10-08 10:01 UTC heartbeat

Preparation advanced from 8,339 to 8,499 completed files. All 7,800 training
episodes remain cached; validation candidates increased from 539 to 699:
the first six tasks through door-open have 100 each; drawer-close has 92 and
drawer-open has 7. Recent expert collection reports success. These are data
collection outcomes; no trained-planner validation results exist yet.

Supervisor 3126314 and all four workers remain active, frozen source is clean
at `16747bb`, and only GPUs 0–3 are allocated. Health checks, fresh W&B online
readback and snapshot retrieval succeeded via `target_server_2`. Four methods,
seed 3072 and compute limits remain unchanged. Full training runs/checkpoints
remain zero. No intervention was required; continue validation-candidate and
then test-goal preparation before the registered bounded training queue.


### 2026-10-08 10:16 UTC heartbeat

Preparation advanced from 8,499 to 8,624 completed files. All 7,800 training
episodes remain cached; validation candidates increased from 699 to 824:
the first eight tasks through drawer-open have 100 each; faucet-open has 24.
Recent expert collection reports success. These are data-collection outcomes,
not trained-planner scores; test-goal preparation follows validation candidates.

Supervisor 3126314 and all four workers remain active, frozen source is clean
at `16747bb`, and only GPUs 0–3 are allocated. Health checks, fresh W&B online
readback and snapshot retrieval succeeded via `target_server_2`. Four methods,
seed 3072 and compute limits remain unchanged. Full training runs/checkpoints
remain zero. No intervention was required; continue the registered preparation
and bounded training queue.

### 2026-10-08 10:31 UTC heartbeat

Preparation advanced from 8,624 to 8,763 completed files. All 7,800 training
episodes remain cached; validation candidates increased from 824 to 963:
the first nine tasks through faucet-open have 100 each; handle-press has 63.
Recent expert collection reports success. These are data-collection outcomes,
not trained-planner scores; test-goal preparation follows validation candidates.

Supervisor 3126314 and all four workers remain active, frozen source is clean
at `16747bb`, and only GPUs 0–3 are allocated. Health checks, fresh W&B online
readback and snapshot retrieval succeeded via `target_server_2`. Four methods,
seed 3072 and compute limits remain unchanged. Full training runs/checkpoints
remain zero. No intervention was required; continue the registered preparation
and bounded training queue.

### 2026-10-08 10:46 UTC heartbeat

Preparation advanced from 8,763 to 8,906 completed files. All 7,800 training
episodes remain cached; validation candidates increased from 963 to 1,106:
the first ten tasks through handle-press have 100 each; pick-place has 96 and
plate-slide has 10. Recent expert collection reports success. These are data
collection outcomes, not trained-planner scores. Test-goal preparation follows
completion of the 1,300 validation candidates.

Supervisor 3126314 and all four workers remain active, frozen source is clean
at `16747bb`, and only GPUs 0–3 are allocated. Health checks, fresh W&B online
readback and snapshot retrieval succeeded via `target_server_2`. Four methods,
seed 3072 and compute limits remain unchanged. Full training runs/checkpoints
remain zero. No intervention was required; continue the registered preparation
and bounded training queue.


### 2026-10-08 11:01 UTC heartbeat

Preparation advanced from 8,906 to 9,071 completed files in the snapshot retrieved
after connection retries. All 7,800 training episodes remain cached; validation
candidates increased from 1,106 to 1,270: the first twelve tasks have 100 each,
and reach has 70. The first test/assembly goal candidate is also cached as a
worker advances to its next assigned split. This is fixed goal-data preparation,
not planner evaluation or test-based model selection. No planner scores exist.

Supervisor 3126314 and all four workers were active in the fresh health check,
frozen source was clean at `16747bb`, and only GPUs 0–3 were allocated. Online
W&B readback succeeded at 11:01:53 UTC. The initial snapshot request timed out
after 120 seconds; the fallback alias then exited with SSH status 255. Retrying
the primary alias retrieved the snapshot successfully without restarting or
modifying the campaign. Four methods, seed 3072 and compute limits are unchanged;
full training runs/checkpoints remain zero. Continue the remaining validation
candidates and registered test-goal preparation before bounded training.


### 2026-10-08 11:16 UTC heartbeat — validation candidates complete

All 1,300 validation candidates are now cached, 100 for each of the thirteen
training tasks. Together with the unchanged 7,800 training episodes, all 9,100
dense-feature episodes have finished preparation. All four workers have moved
to the registered test-goal candidate pool: test/assembly has 109 files, bringing
the cache to 9,209 completed files. The fixed pool is 6,400 candidates across
sixteen tasks, from which the protocol selects 3,200 goal-constructible resets.
This prepares fixed inputs before training; no planner test evaluation or
test-based model selection has occurred. No benchmark scores exist yet.

Supervisor 3126314 and all four workers remain active, frozen source is clean
at `16747bb`, and only GPUs 0–3 are allocated. Primary SSH health inspection,
W&B online readback at 11:17:01 UTC and snapshot retrieval succeeded this time.
Four methods, seed 3072 and compute limits remain unchanged; full training
runs/checkpoints remain zero. No intervention was required. Continue fixed
test-goal preparation, then the registered bounded training/validation queue.

### 2026-10-08 11:31 UTC heartbeat

Preparation advanced from 9,209 to 9,381 completed files. The 7,800 training
episodes and all 1,300 validation candidates remain cached. Fixed test-goal
candidates increased from 109 to 281, currently all in assembly, out of the
registered 6,400-candidate pool. Recent expert collection reports success;
these are goal-preparation outcomes, not planner test scores.

Supervisor 3126314 and all four workers remain active, frozen source is clean
at `16747bb`, and only GPUs 0–3 are allocated. Primary SSH health inspection,
fresh W&B online readback and snapshot retrieval succeeded. Four methods,
seed 3072 and compute limits remain unchanged; full training runs/checkpoints
remain zero. No intervention was required. Continue fixed test-goal preparation
before the registered bounded training/validation queue.

### 2026-10-08 11:46 UTC heartbeat

Preparation advanced from 9,381 to 9,556 completed files. All 7,800 training
episodes and 1,300 validation candidates remain cached. Fixed test-goal
candidates increased from 281 to 456: assembly is complete at 400, and
button-press-topdown has 56. The registered pool remains 6,400 candidates.
Recent expert collection reports success; these are goal-preparation outcomes,
not planner test scores.

Supervisor 3126314 and all four workers remain active, frozen source is clean
at `16747bb`, and only GPUs 0–3 are allocated. Primary SSH health inspection,
fresh W&B online readback and snapshot retrieval succeeded. Four methods,
seed 3072 and compute limits remain unchanged; full training runs/checkpoints
remain zero. No intervention was required. Continue fixed test-goal preparation
before the registered bounded training/validation queue.

### 2026-10-08 12:01 UTC heartbeat

Preparation advanced from 9,556 to 9,759 completed files in the snapshot
retrieved after connection retries. All 7,800 training episodes and 1,300
validation candidates remain cached. Fixed test-goal candidates increased from
456 to 659: assembly has 400 and button-press-topdown has 259, out of the
registered 6,400-candidate pool. Recent expert collection reports success;
these are goal-preparation outcomes, not planner test scores.

The primary SSH health request timed out during banner exchange. The fallback
alias succeeded: supervisor 3126314 and all four workers were active, source
was clean at `16747bb`, only GPUs 0–3 were allocated, and fresh W&B online
readback succeeded at 12:02:59 UTC. A subsequent snapshot request through the
fallback exited with SSH status 255; retrying the primary alias succeeded.
No campaign restart or code change was needed. Four methods, seed 3072 and
compute limits remain unchanged; full training runs/checkpoints remain zero.
Continue fixed test-goal preparation before bounded training/validation.

### 2026-10-08 12:16 UTC heartbeat — healthy live check, snapshot unavailable

The primary SSH health check succeeded. Supervisor 3126314, launcher 3128848
and workers 3128857–3128860 were active in `prepare_data`; execution source was
clean at `16747bb`. GPUs 0–3 held approximately 2.3 GiB each, and GPUs 4–7 were
unused. Recent logs show continued expert goal preparation: worker 0 reached
`test/coffee-button/00060`, worker 1 `test/coffee-button/00001`, and workers 2/3
were finishing their button-press-topdown assignments. These are goal-data
collection outcomes, not planner test scores. Full training runs/checkpoints
remain zero, with four methods, seed 3072 and compute limits unchanged.

Online W&B readback succeeded at 12:16:59 UTC. Its returned JSON is preserved
in `online-readback.json` from the successful health-command output. Subsequent
snapshot retrieval failed three times: primary alias, fallback alias, then
primary retry all exited with SSH status 255; the final attempt explicitly
reported a banner-exchange timeout. No snapshot files were updated by those
failed requests. `preparation-progress.json` therefore remains the last complete
snapshot (9,759 total files, including 659 test-goal candidates), not a current
count. No restart or code change was made. Retry the snapshot at the next
heartbeat and continue monitoring fixed test-goal preparation before training.


### 2026-10-08 12:31 UTC heartbeat — snapshot retrieval restored

Primary SSH health inspection, W&B online readback and the full compact snapshot
all succeeded. The snapshot is current again after the previous heartbeat's
connection failures: 10,092 completed files versus the last verified 9,759.
All 7,800 training episodes and 1,300 validation candidates remain cached.
Fixed test-goal candidates now total 992 of 6,400: assembly 400,
button-press-topdown 400 and coffee-button 192. Recent expert collection reports
success; these are fixed goal-data outcomes, not planner test scores.

Supervisor 3126314 and all four workers remain active, frozen execution source
is clean at `16747bb`, and only GPUs 0–3 are allocated. Online W&B readback
succeeded at 12:32:01 UTC. Four methods, seed 3072 and compute limits remain
unchanged; full training runs/checkpoints remain zero. No restart or code change
was required. Continue fixed test-goal preparation before bounded training and
validation, keeping planner test evaluation sealed until all models finish.

### 2026-10-08 12:46 UTC heartbeat

Preparation advanced from 10,092 to 10,269 completed files. All 7,800 training
episodes and 1,300 validation candidates remain cached. Fixed test-goal
candidates increased from 992 to 1,169: assembly and button-press-topdown have
400 each, coffee-button has 362 and dial-turn has 7. The registered pool remains
6,400 candidates. Recent expert collection reports success; these are goal-data
outcomes, not planner test scores.

Supervisor 3126314 and all four workers remain active, frozen execution source
is clean at `16747bb`, and only GPUs 0–3 are allocated. Primary SSH health
inspection, fresh W&B online readback and snapshot retrieval succeeded. Four
methods, seed 3072 and compute limits remain unchanged; full training
runs/checkpoints remain zero. No intervention was required. Continue fixed
test-goal preparation before the registered bounded training/validation queue.

### 2026-10-08 13:01 UTC heartbeat

Preparation advanced from 10,269 to 10,458 completed files. All 7,800 training
episodes and 1,300 validation candidates remain cached. Fixed test-goal
candidates increased from 1,169 to 1,358: assembly, button-press-topdown and
coffee-button have 400 each; dial-turn has 158. The registered pool remains
6,400 candidates. Recent expert collection reports success; these are goal-data
outcomes, not planner test scores.

The primary SSH health request timed out during banner exchange. The fallback
alias succeeded for health inspection, fresh W&B online readback at 13:02:44 UTC
and snapshot retrieval. Supervisor 3126314 and all four workers remain active,
frozen execution source is clean at `16747bb`, and only GPUs 0–3 are allocated.
Four methods, seed 3072 and compute limits remain unchanged; full training
runs/checkpoints remain zero. No campaign intervention was required. Continue
fixed test-goal preparation before the bounded training/validation queue.

### 2026-10-08 13:16 UTC heartbeat

Preparation advanced from 10,458 to 10,642 completed files. All 7,800 training
episodes and 1,300 validation candidates remain cached. Fixed test-goal
candidates increased from 1,358 to 1,542: assembly, button-press-topdown and
coffee-button have 400 each; dial-turn has 341 and door-close has 1. The
registered pool remains 6,400 candidates. Recent expert collection reports
success; these are goal-data outcomes, not planner test scores.

The primary SSH health request timed out during banner exchange. The fallback
alias succeeded for health inspection, fresh W&B online readback at 13:17:39 UTC
and snapshot retrieval. Supervisor 3126314 and all four workers remain active,
frozen execution source is clean at `16747bb`, and only GPUs 0–3 are allocated.
Four methods, seed 3072 and compute limits remain unchanged; full training
runs/checkpoints remain zero. No campaign intervention was required. Continue
fixed test-goal preparation before the bounded training/validation queue.


### 2026-10-08 13:31 UTC heartbeat

Preparation advanced from 10,642 to 10,807 completed files. All 7,800 training
episodes and 1,300 validation candidates remain cached. Fixed test-goal
candidates increased from 1,542 to 1,707: the first four tasks through dial-turn
have 400 each; door-close has 107. The registered pool remains 6,400 candidates.
Recent expert collection reports success; these are goal-data outcomes, not
planner test scores.

Used the configured fallback alias directly following recent primary connection
timeouts. Health inspection, fresh W&B online readback at 13:31:56 UTC and
snapshot retrieval all succeeded. Supervisor 3126314 and all four workers
remain active, frozen execution source is clean at `16747bb`, and only GPUs 0–3
are allocated. Four methods, seed 3072 and compute limits remain unchanged;
full training runs/checkpoints remain zero. No campaign intervention was
required. Continue fixed test-goal preparation before bounded training/validation.

### 2026-10-08 13:46 UTC heartbeat

Preparation advanced from 10,807 to 10,977 completed files. All 7,800 training
episodes and 1,300 validation candidates remain cached. Fixed test-goal
candidates increased from 1,707 to 1,877: the first four tasks through dial-turn
have 400 each; door-close has 277. The registered pool remains 6,400 candidates.
Recent expert collection reports success; these are goal-data outcomes, not
planner test scores.

Health inspection, fresh W&B online readback at 13:46:56 UTC and snapshot
retrieval succeeded through the configured fallback alias. Supervisor 3126314
and all four workers remain active, frozen execution source is clean at
`16747bb`, and only GPUs 0–3 are allocated. Four methods, seed 3072 and compute
limits remain unchanged; full training runs/checkpoints remain zero. No campaign
intervention was required. Continue fixed test-goal preparation before the
registered bounded training/validation queue.

### 2026-10-08 14:01 UTC heartbeat

Preparation advanced from 10,977 to 11,148 completed files. All 7,800 training
episodes and 1,300 validation candidates remain cached. Fixed test-goal
candidates increased from 1,877 to 2,048: the first five tasks through door-close
have 400 each; door-open has 48. The registered pool remains 6,400 candidates.
Recent expert collection reports success; these are goal-data outcomes, not
planner test scores.

Health inspection, fresh W&B online readback at 14:02:01 UTC and snapshot
retrieval succeeded through the configured fallback alias. Supervisor 3126314
and all four workers remain active, frozen execution source is clean at
`16747bb`, and only GPUs 0–3 are allocated. Four methods, seed 3072 and compute
limits remain unchanged; full training runs/checkpoints remain zero. No campaign
intervention was required. Continue fixed test-goal preparation before the
registered bounded training/validation queue.

### 2026-10-08 14:16 UTC heartbeat

Preparation advanced from 11,148 to 11,316 completed files. All 7,800 training
episodes and 1,300 validation candidates remain cached. Fixed test-goal
candidates increased from 2,048 to 2,216: the first five tasks through door-close
have 400 each; door-open has 216. The registered pool remains 6,400 candidates.
The recent log includes a failed expert goal-collection attempt at
`test/door-open/00186`, retained with `success: false`, alongside successful
attempts. This is expected input to the preregistered goal-constructibility
screening, not a planner test result or a reason to change the candidate pool.

Health inspection, fresh W&B online readback at 14:17:01 UTC and snapshot
retrieval succeeded through the configured fallback alias. Supervisor 3126314
and all four workers remain active, frozen execution source is clean at
`16747bb`, and only GPUs 0–3 are allocated. Four methods, seed 3072 and compute
limits remain unchanged; full training runs/checkpoints remain zero. No campaign
intervention was required. Continue fixed test-goal preparation before the
registered bounded training/validation queue.


### 2026-10-08 14:31 UTC heartbeat

Preparation advanced from 11,316 to 11,486 completed files. All 7,800 training
episodes and 1,300 validation candidates remain cached. Fixed test-goal
candidates increased from 2,216 to 2,386: the first five tasks through door-close
have 400 each; door-open has 372 and drawer-close has 14. The registered pool
remains 6,400 candidates. Recent expert collection reports success; these are
goal-data outcomes, not planner test scores.

Health inspection, fresh W&B online readback at 14:32:05 UTC and snapshot
retrieval succeeded through the configured fallback alias. Supervisor 3126314
and all four workers remain active, frozen execution source is clean at
`16747bb`, and only GPUs 0–3 are allocated. Four methods, seed 3072 and compute
limits remain unchanged; full training runs/checkpoints remain zero. No campaign
intervention was required. Continue fixed test-goal preparation before the
registered bounded training/validation queue.

### 2026-10-08 14:46 UTC heartbeat

Preparation advanced from 11,486 to 11,684 completed files in the snapshot
retrieved after a connection retry. All 7,800 training episodes and 1,300
validation candidates remain cached. Fixed test-goal candidates increased from
2,386 to 2,584: the first six tasks through door-open have 400 each;
drawer-close has 184. The registered pool remains 6,400 candidates. Recent
expert collection reports success; these are goal-data outcomes, not planner
test scores.

Health inspection and fresh W&B online readback at 14:47:00 UTC succeeded
through the fallback alias. Its subsequent snapshot request exited with SSH
status 255; retrying the primary alias retrieved the snapshot successfully.
Supervisor 3126314 and all four workers remain active, frozen execution source
is clean at `16747bb`, and only GPUs 0–3 are allocated. Four methods, seed 3072
and compute limits remain unchanged; full training runs/checkpoints remain
zero. No campaign intervention was required. Continue fixed test-goal
preparation before the registered bounded training/validation queue.


### 2026-10-08 15:01 UTC heartbeat

Preparation advanced from 11,684 to 11,853 completed files. All 7,800 training
episodes and 1,300 validation candidates remain cached. Fixed test-goal
candidates increased from 2,584 to 2,753: the first six tasks through door-open
have 400 each; drawer-close has 345 and drawer-open has 8. The registered pool
remains 6,400 candidates. Recent expert collection reports success; these are
goal-data outcomes, not planner test scores.

Health inspection, fresh W&B online readback at 15:02:09 UTC and snapshot
retrieval succeeded through the configured fallback alias. Supervisor 3126314
and all four workers remain active, frozen execution source is clean at
`16747bb`, and only GPUs 0–3 are allocated. Four methods, seed 3072 and compute
limits remain unchanged; full training runs/checkpoints remain zero. No campaign
intervention was required. Continue fixed test-goal preparation before the
registered bounded training/validation queue.


### 2026-10-08 15:16 UTC heartbeat

Preparation advanced from 11,853 to 12,036 completed files. All 7,800 training
episodes and 1,300 validation candidates remain cached. Fixed test-goal
candidates increased from 2,753 to 2,936: the first seven tasks through
drawer-close have 400 each; drawer-open has 136. The registered pool remains
6,400 candidates. Recent expert collection reports success; these are goal-data
outcomes, not planner test scores.

Health inspection, fresh W&B online readback at 15:17:04 UTC and snapshot
retrieval succeeded through the configured fallback alias. Supervisor 3126314
and all four workers remain active, frozen execution source is clean at
`16747bb`, and only GPUs 0–3 are allocated. Four methods, seed 3072 and compute
limits remain unchanged; full training runs/checkpoints remain zero. No campaign
intervention was required. Continue fixed test-goal preparation before the
registered bounded training/validation queue.


### 2026-10-08 15:31 UTC heartbeat — live check passed, snapshot unavailable

The fallback SSH health request timed out during banner exchange (status 255).
Retrying the primary alias succeeded: supervisor 3126314 and data workers
3128857–3128860 were active under launcher 3128848, stage `prepare_data`.
The frozen execution checkout remained clean at `16747bb`; GPUs 0–3 held
2,363/2,363/2,363/2,379 MiB and GPUs 4–7 held zero. Scope assertions confirmed
only the four registered methods, seed 3072, 7,200-second training allowances
and 10-second controller allowances. Full training runs/checkpoints remain zero.
Recent logs progressed through `test/drawer-open/00396` on rank 0 and nearby
rank-specific episodes, with expert success reported. These are preparation
records, not planner test results or an exact aggregate count.

Fresh W&B readback succeeded at 15:32:46 UTC and is preserved from the successful
SSH response in `online-readback.json`. All three subsequent snapshot requests
(primary, fallback, primary) failed with SSH status 255; the final attempt
reported a banner-exchange timeout. `preparation-progress.json` remains the
last successful 15:16 snapshot: 12,036 files = 7,800 training + 1,300 validation
candidates + 2,936 test-goal candidates. Those counts are stale, not evidence of
a stalled campaign. No processes, code, data or budgets were changed. Retry
snapshot retrieval on the next heartbeat and continue the registered bounded
training/validation queue after fixed goal preparation completes.


### 2026-10-08 15:46 UTC heartbeat — snapshot retrieval recovered

The retrieved snapshot confirms 12,423 completed files, up from the last
successful 15:16 snapshot's 12,036. All 7,800 training episodes and 1,300
validation candidates remain cached. Fixed test-goal candidates increased from
2,936 to 3,323: the first eight tasks through drawer-open have 400 each;
faucet-open has 123. More than half of the registered 6,400-candidate pool is
now prepared. Recent expert collection reports success; these are goal-data
outcomes, not planner test scores.

The live health inspection and fresh W&B readback at 15:46:59 UTC succeeded
through the fallback alias. Its snapshot request later failed with SSH status
255; the primary-alias retry succeeded and replaced the stale preparation
snapshot. Supervisor 3126314 and all four workers remain active, frozen
execution source is clean at `16747bb`, and only GPUs 0–3 are allocated.
Four methods, seed 3072 and compute limits remain unchanged; full training
runs/checkpoints remain zero. No campaign intervention was required. Continue
fixed test-goal preparation before the registered bounded training/validation
queue.


### 2026-10-08 16:01 UTC heartbeat

Preparation advanced from 12,423 to 12,574 completed files. All 7,800 training
episodes and 1,300 validation candidates remain cached. Fixed test-goal
candidates increased from 3,323 to 3,474: the first eight tasks through
drawer-open have 400 each; faucet-open has 274. The registered pool remains
6,400 candidates. Recent expert collection reports success; these are goal-data
outcomes, not planner test scores.

Health inspection, fresh W&B online readback at 16:02:02 UTC and snapshot
retrieval succeeded through the configured fallback alias. Supervisor 3126314
and all four workers remain active, frozen execution source is clean at
`16747bb`, and only GPUs 0–3 are allocated. Four methods, seed 3072 and compute
limits remain unchanged; full training runs/checkpoints remain zero. No campaign
intervention was required. Continue fixed test-goal preparation before the
registered bounded training/validation queue.


### 2026-10-08 16:16 UTC heartbeat

Preparation advanced from 12,574 to 12,749 completed files. All 7,800 training
episodes and 1,300 validation candidates remain cached. Fixed test-goal
candidates increased from 3,474 to 3,649: the first nine tasks through
faucet-open have 400 each; handle-press has 49. The registered pool remains
6,400 candidates. Recent expert collection reports success; these are goal-data
outcomes, not planner test scores.

Health inspection, fresh W&B online readback at 16:16:55 UTC and snapshot
retrieval succeeded through the configured fallback alias. Supervisor 3126314
and all four workers remain active, frozen execution source is clean at
`16747bb`, and only GPUs 0–3 are allocated. Four methods, seed 3072 and compute
limits remain unchanged; full training runs/checkpoints remain zero. No campaign
intervention was required. Continue fixed test-goal preparation before the
registered bounded training/validation queue.


### 2026-10-08 16:31 UTC heartbeat

Preparation advanced from 12,749 to 12,920 completed files. All 7,800 training
episodes and 1,300 validation candidates remain cached. Fixed test-goal
candidates increased from 3,649 to 3,820: the first nine tasks through
faucet-open have 400 each; handle-press has 220. The registered pool remains
6,400 candidates. Recent expert collection reports success; these are goal-data
outcomes, not planner test scores.

Health inspection, fresh W&B online readback at 16:32:22 UTC and snapshot
retrieval succeeded through the configured fallback alias. Supervisor 3126314
and all four workers remain active, frozen execution source is clean at
`16747bb`, and only GPUs 0–3 are allocated. Four methods, seed 3072 and compute
limits remain unchanged; full training runs/checkpoints remain zero. No campaign
intervention was required. Continue fixed test-goal preparation before the
registered bounded training/validation queue.


### 2026-10-08 16:46 UTC heartbeat

Preparation advanced from 12,920 to 13,086 completed files. All 7,800 training
episodes and 1,300 validation candidates remain cached. Fixed test-goal
candidates increased from 3,820 to 3,986: the first nine tasks through
faucet-open have 400 each; handle-press has 368 and pick-place has 18. The
registered pool remains 6,400 candidates. Recent expert collection reports
success; these are goal-data outcomes, not planner test scores.

Health inspection, fresh W&B online readback at 16:46:52 UTC and snapshot
retrieval succeeded through the configured fallback alias. Supervisor 3126314
and all four workers remain active, frozen execution source is clean at
`16747bb`, and only GPUs 0–3 are allocated. Four methods, seed 3072 and compute
limits remain unchanged; full training runs/checkpoints remain zero. No campaign
intervention was required. Continue fixed test-goal preparation before the
registered bounded training/validation queue.


### 2026-10-08 17:01 UTC heartbeat

Preparation advanced from 13,086 to 13,264 completed files. All 7,800 training
episodes and 1,300 validation candidates remain cached. Fixed test-goal
candidates increased from 3,986 to 4,164: the first ten tasks through
handle-press have 400 each; pick-place has 164. The registered pool remains
6,400 candidates. Recent expert collection reports success; these are goal-data
outcomes, not planner test scores.

Health inspection, fresh W&B online readback at 17:01:55 UTC and snapshot
retrieval succeeded through the configured fallback alias. Supervisor 3126314
and all four workers remain active, frozen execution source is clean at
`16747bb`, and only GPUs 0–3 are allocated. Four methods, seed 3072 and compute
limits remain unchanged; full training runs/checkpoints remain zero. No campaign
intervention was required. Continue fixed test-goal preparation before the
registered bounded training/validation queue.


### 2026-10-08 17:16 UTC heartbeat

Preparation advanced from 13,264 to 13,441 completed files. All 7,800 training
episodes and 1,300 validation candidates remain cached. Fixed test-goal
candidates increased from 4,164 to 4,341: the first ten tasks through
handle-press have 400 each; pick-place has 333 and plate-slide has 8. The
registered pool remains 6,400 candidates. Recent expert collection reports
success; these are goal-data outcomes, not planner test scores.

Health inspection, fresh W&B online readback at 17:16:57 UTC and snapshot
retrieval succeeded through the configured fallback alias. Supervisor 3126314
and all four workers remain active, frozen execution source is clean at
`16747bb`, and only GPUs 0–3 are allocated. Four methods, seed 3072 and compute
limits remain unchanged; full training runs/checkpoints remain zero. No campaign
intervention was required. Continue fixed test-goal preparation before the
registered bounded training/validation queue.


### 2026-10-08 17:31 UTC heartbeat

Preparation advanced from 13,441 to 13,620 completed files. All 7,800 training
episodes and 1,300 validation candidates remain cached. Fixed test-goal
candidates increased from 4,341 to 4,520: the first eleven tasks through
pick-place have 400 each; plate-slide has 120. The registered pool remains
6,400 candidates. Recent expert collection reports success; these are goal-data
outcomes, not planner test scores.

Health inspection, fresh W&B online readback at 17:31:57 UTC and snapshot
retrieval succeeded through the configured fallback alias. Supervisor 3126314
and all four workers remain active, frozen execution source is clean at
`16747bb`, and only GPUs 0–3 are allocated. Four methods, seed 3072 and compute
limits remain unchanged; full training runs/checkpoints remain zero. No campaign
intervention was required. Continue fixed test-goal preparation before the
registered bounded training/validation queue.


### 2026-10-08 17:46 UTC heartbeat

Preparation advanced from 13,620 to 13,794 completed files. All 7,800 training
episodes and 1,300 validation candidates remain cached. Fixed test-goal
candidates increased from 4,520 to 4,694: the first eleven tasks through
pick-place have 400 each; plate-slide has 294. The registered pool remains
6,400 candidates. Recent expert collection reports success; these are goal-data
outcomes, not planner test scores.

Health inspection, fresh W&B online readback at 17:46:55 UTC and snapshot
retrieval succeeded through the configured fallback alias. Supervisor 3126314
and all four workers remain active, frozen execution source is clean at
`16747bb`, and only GPUs 0–3 are allocated. Four methods, seed 3072 and compute
limits remain unchanged; full training runs/checkpoints remain zero. No campaign
intervention was required. Continue fixed test-goal preparation before the
registered bounded training/validation queue.

### 2026-10-08 18:18 UTC — lossless throughput implementation verified

The user explicitly authorized optimizing hardware throughput. Implemented sparse
exact-action-replay goal collection, bounded parallel CPU producers, larger GPU
batches, asynchronous atomic cache writes, incremental clip materialization and
parallel cache statistics/hash checks with original reduction order. No methods,
training seeds, split/candidate IDs, image resolution, model precision or compute
allowances changed. Implementation/evidence are in `docs/THROUGHPUT.md` and the
compact throughput benchmark report.

All 20 real equivalence cases (all 16 tasks and four random cases) passed exact
array checks. Two end-to-end preparation configurations matched original cached
arrays, logical attributes, means and standard deviations bit-for-bit. All 43
server tests passed. The measured configuration selected for deployment is 12 CPU
producers per GPU (48 total), 24 prefetched jobs per GPU, batch 64 for both dense
features and goals, one writer per GPU, and eight manifest workers. Batch 128 did
not outperform 64. Warm 48-worker goal collection reached 29.61 episodes/second;
this is a CPU-stage measurement, not a claim about live end-to-end throughput.

The old supervisor is still running preparation while the verified revision is
published and transferred. Migration remains in progress; the existing heartbeat
is restricted to read-only checks until deployment and cache preservation are
verified. Next: switch to a new frozen checkout, preserve completed files, verify
actual throughput and the transition into training with W&B logs.

### 2026-10-08 18:24 UTC — optimized supervisor deployed; cache hashes verified

Frozen execution revision `56419ed14cf127d4b7a8ab09a9d68e6681d2edc0` is deployed
in `repo-throughput`. The dry run confirmed the scientific execution and original
data protocol digests are unchanged. The old preparation-only supervisor and its
verified process tree were stopped before any training run existed. All 14,230
completed files were retained; every file was hashed before launch and checked
again afterward with no changes, including unselected goal candidates.

New supervisor PID 142087 uses updated `launch.sh` with 12 producers/GPU, prefetch
24/GPU, encoder/goal batches 64, eight manifest readers, and only GPUs 0–3. Its log
is `launcher-throughput.log`. Previous metadata and complete cache hash baseline
are archived under `throughput-scope-change-20261008` in the server record.
`campaign/throughput-scope-change.json` records the switch and verified hashes.
Fresh W&B API readback confirmed revision, methods, seed, budgets and preparation
settings: https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/8eme1wa7.
GPU gates and live throughput are being verified; the monitor remains read-only
until these checks complete. No optimizer training had started at this check.

### 2026-10-08 18:32 UTC — migration complete; real training verified online

The new four-rank GPU preflight passed. Preparation then wrote all remaining
1,270 files with a maximum rank elapsed time of 42.200 seconds, approximately
30.09 files/second across the four ranks. All 15,500 candidates are now present.
This is around 150 times the preceding live goal-collection rate, with differing
nearby task mixes; the separate same-case equivalence timings establish the
per-episode improvement. Startup/GPU gating and final manifest checks are outside
that 42-second worker timing and are not hidden inside the throughput claim.

Every one of the 14,230 pre-existing files was SHA-256 checked again after
preparation, all unchanged. Goal screening produced the complete registered
650 validation and 3,200 test cases, and the manifest contains 11,650 entries.
The supervisor automatically advanced into shared world training. At 18:32 UTC
(11:32 a.m. America/Los_Angeles), W&B API readback confirmed real run `fbcl549w`
active at step 1,400, loss 0.0302401, training time 193.46 seconds, four GPUs,
seed 3072 and the new frozen code. A durable `last.pt` exists. There are no
validation reports yet; the first evaluation occurs at the first budget/update
milestone. These loss metrics are not benchmark success results.

The existing heartbeat was updated to the optimized checkout, launcher and
provenance. Its temporary read-only migration restriction is removed. Next:
continue world training and periodic CEM/world validation, then the three selected
learned heads and their periodic evaluations, followed by the fixed paired final
test. No extra seeds, methods, ablations or budget increases are scheduled.

### 2026-10-08 18:46–18:52 UTC — first periodic validation preserved; training resumed

Read the campaign instructions and both scope-change records, then checked the
actual supervisor, training and evaluation processes over `target_server_2_cf`.
Supervisor 142087 is healthy; frozen `repo-throughput` remains clean at
`56419ed14cf127d4b7a8ab09a9d68e6681d2edc0`. Evaluation used GPUs 0–3 and completed
normally, after which the existing four training ranks resumed optimization.
GPUs 4–7 were empty. No restart, source edit or scope change was required.

At world step 5,000 the first registered CEM validation completed all 104 unique
cases (eight per training task), using seed 3072 and the fixed controller cap.
CEM achieved 10/104 successes (9.615% macro): drawer-close 3/8 and handle-press
7/8; the other 11 tasks had no successes. Success within 50 primitive actions was
7/104; within 100 and 200 it was 10/104. All 94 unsuccessful episodes exhausted
the 10-second controller allowance and were retained as failures. Mean controller
time was 9.598 seconds/episode, mean step latency 280.62 ms and p95 latency
287.77 ms. The mean safe-boundary budget overrun was recorded as 0.01347 seconds.
This early failure pattern is dominated by controller-budget exhaustion; it does
not yet establish why planning fails or how the learned heads will compare.

World validation dynamics loss was 0.021569 versus persistence loss 0.062804;
action identification among 16 choices was 81.445% versus 6.25% chance. These
diagnostics do not establish final task success. W&B API readback at
18:49:04 UTC confirmed the running real training run `fbcl549w`, step 5,450,
finite training loss 0.025889, the first validation round and its complete metrics.
The later downloaded ledger reached step 6,769, charging 893.436 optimization
seconds (0.992706 GPU-hours) and separately 451.743 validation seconds
(0.501937 GPU-hours). The latest sampled loss at step 6,750 was 0.022437.
Both durable checkpoints are present, 44,360,165 bytes each.

Ran `scripts/snapshot_flow_campaign.py --host target_server_2_cf --root
/home/mtxu/adam/LeFlow-experiments/20261007-joint-flow/campaign --output
docs/reports/20261007-joint-flow`. Compact evidence includes the complete paired
episode records in `runs/world_3072/validation/cem_step_0005000.json`, loss samples,
checkpoint metadata, charged compute and fresh online readback. Verified 104
unique validation IDs, 13 tasks with eight cases each, seed 3072, a 10-second cap,
and retention of every timeout as a failure. The throughput provenance update is
JSON formatting only; all preparation preservation evidence remains intact.

The learned heads have not started and final test remains sealed. Continue the
remaining world validation rounds, then the three registered learned heads and
their matched validations. Any follow-up experiment proposal should use the
completed comparison and failure analysis; no ablation or budget extension has
been queued.

### 2026-10-08 19:01 UTC — world step 10,000; second validation active

Supervisor 142087 and its four training ranks (148180–148183 under launcher
148077) remain healthy in clean frozen `repo-throughput` at `56419ed`. Verified
each rank's working directory, rank identity and `CUDA_VISIBLE_DEVICES=0,1,2,3`.
Only GPUs 0–3 are occupied; GPUs 4–7 are empty. Periodic evaluation runs inside
the existing training ranks; the preceding entry's wording that evaluation
workers exited has been corrected. NVIDIA's reported process IDs are not visible
as process IDs inside this container, so they are not used to infer worker exits.

The latest training step is 10,000 with finite loss 0.021481 and gradient norm
0.012737. Its second 104-case CEM validation is advancing through the task logs;
there is no second completed result yet. Charged optimization is 1,321.442 seconds
(1.468268 GPU-hours), with no last-update overrun. The first validation cost
remains separately recorded at 451.743 seconds (0.501937 GPU-hours); the active
round's cost is added at completion. Checkpoints are preserved. No process or
source intervention was needed.

Fresh W&B API readback at 19:02:22 UTC confirmed `fbcl549w` running, seed 3072,
frozen code and step 9,950 (the online summary lags the local step). Downloaded
the updated snapshot through `target_server_2_cf`, preserving loss samples,
checkpoint metadata, compute ledger and the online evidence. No learned head or
final evaluation has started. Next: preserve the completed second validation and
continue the unchanged registered queue; final testing stays sealed until all
registered learned models finish.

### 2026-10-08 19:16 UTC — second validation preserved; world at 15,000 updates

The second CEM validation at world step 10,000 completed all 104 registered
cases: 9 successes (8.654% macro) and 95 controller-budget timeouts (91.346%),
all retained as failures. Drawer-close improved from 3/8 to 5/8; handle-press
decreased from 7/8 to 4/8; the other 11 tasks remained at zero. Against the first
round's identical reset IDs and episode hashes, six cases succeeded in both,
three changed from failure to success, four changed from success to failure, and
91 failed in both. These intermediate paired observations do not establish a
method ranking or variation across training seeds.

Seven cases succeeded within 50 primitive actions and nine within 100/200.
Mean controller time was 9.603 seconds/episode; step latency was 280.23 ms mean
and 287.89 ms p95; recorded mean safe-boundary overrun was 0.01215 seconds.
World validation dynamics loss improved from 0.021569 to 0.019287, with
persistence loss unchanged at 0.062804. Action identification among 16 choices
rose from 81.445% to 87.891% (6.25% chance). Better prediction diagnostics have
not translated into improved task success in this round; timeout-dominated
failure remains the observed limitation, without an established causal diagnosis.

Supervisor 142087, launcher 148077 and ranks 148180–148183 remain active in clean
frozen `repo-throughput` at `56419ed`. Only GPUs 0–3 are occupied; GPUs 4–7 remain
empty. Scope-change records still specify exactly four methods, seed 3072 and
the original budgets. No restart, code change or budget adjustment was needed.
Fresh W&B readback at 19:17:04 UTC verified run `fbcl549w` online at step 14,750,
with two completed validation rounds. The later snapshot reached step 15,000,
the third validation milestone, with finite loss 0.018723 and gradient norm
0.009581. Its ledger charges 1,984.127 optimization seconds (2.204585 GPU-hours)
and separately 890.382 seconds (0.989313 GPU-hours) for two completed validation
rounds. No last-update overrun is recorded; both checkpoints are preserved.

Downloaded the campaign snapshot through `target_server_2_cf`. Verified that
`runs/world_3072/validation/cem_step_0010000.json` contains 104 unique validation
IDs, eight per task, identical episode hashes/reset seeds to round one, seed 3072,
the 10-second cap and every timeout counted as a failure. Saved the complete
paired records, updated loss samples, compute ledger and online readback.
Next: complete the remaining world validations and then the three registered
heads under their matched allowances. Final testing remains sealed; no follow-up
ablation, extra seed or experiment has been launched.

### 2026-10-08 19:31 UTC — third validation preserved; approaching world update cap

World step 15,000 produced the third complete 104-case CEM validation: seven
successes (6.731% macro), with all 97 controller timeouts (93.269%) retained as
failures. Drawer-close and handle-press each achieved 3/8; reach achieved 1/8;
the other 10 tasks achieved zero. Compared with round two on identical cases,
five remained successful, two changed from failure to success, four changed
from success to failure and 93 failed both rounds. Five cases succeeded within
50 primitive actions and seven within 100/200. Mean controller time was
9.767 seconds/episode, mean step latency 280.37 ms and p95 287.78 ms; the recorded
mean safe-boundary overrun was 0.01286 seconds.

Validation dynamics loss improved again to 0.017964 versus persistence 0.062804;
action identification among 16 choices was 86.133% (6.25% chance). CEM success
has decreased across these three checkpoints despite improving dynamics loss.
This is a negative intermediate outcome, not a causal diagnosis or a comparison
with the untrained heads. The frozen implementation selects the shared world
checkpoint using validation dynamics loss; that registered rule remains intact.

Supervisor 142087, launcher 148077 and training ranks 148180–148183 remain
healthy, using only GPUs 0–3; GPUs 4–7 are empty. Execution source is clean at
`56419ed` in `repo-throughput`. Both scope-change records retain the authorized
methods, seed and limits. No recovery, source edit or additional computation was
requested. W&B API readback at 19:31:57 UTC verified `fbcl549w` running at step
18,050, with all three validation rounds online. The later snapshot ledger
reached step 18,493, charging 2,452.177 optimization seconds (2.724641 GPU-hours)
and separately 1,335.186 validation seconds (1.483539 GPU-hours). No last-update
overrun is recorded; the latest sampled loss at step 18,450 was 0.020641 and
both checkpoints remain present.

Downloaded the snapshot through `target_server_2_cf` and verified all three
104-case reports share identical episode IDs, hashes and reset seeds, with
eight cases per task, seed 3072, the 10-second cap and all timeouts retained as
failures. Added `runs/world_3072/validation/cem_step_0015000.json` and updated
the loss samples, checkpoint metadata, charged compute and online evidence.
Next: allow the world model to reach its earlier update/time cap and finish its
fourth validation, then continue the three registered heads. Final test is
still sealed; no scope or budget expansion has been made.

### 2026-10-08 19:46 UTC — shared world complete; joint-flow training verified

The shared world finished normally at step 20,000, stopping on the update cap
before its 7,200-second optimization allowance. `complete.json` records four
validation rounds and 2,653.535 optimization seconds (2.948373 GPU-hours).
Validation was separately charged at 1,780.427 seconds (1.978253 GPU-hours).
No last-update overrun is recorded. W&B API readback at 19:47:00 UTC confirmed
`fbcl549w` finished, with all four rounds and the final metrics online.

The fourth 104-case CEM validation at step 20,000 achieved seven successes
(6.731%) and 97 controller timeouts (93.269%), all retained as failures.
Drawer-close achieved 5/8 and handle-press 2/8; the other 11 tasks had zero.
Six successes occurred within 50 primitive actions and seven within 100/200.
Against round three, five cases remained successful, two changed to success,
two changed to failure and 95 failed both rounds. Mean controller time was
9.718 seconds/episode, mean step latency 280.27 ms and p95 287.81 ms; the mean
recorded safe-boundary overrun was 0.01396 seconds. The complete validation
success sequence is 10, 9, 7, 7 out of the same 104 cases. These are not final
test outcomes and cannot yet rank the four methods.

Final world validation dynamics loss was 0.017506 versus persistence 0.062804;
action identification among 16 choices was 88.281% versus 6.25% chance. The
registered dynamics-loss selection chose the step-20,000 world checkpoint.
Read the saved checkpoints on CPU and verified that the new head's `world_hash`
matches the selected world's SHA-256, with identical manifest, protocol, code,
seed 3072 and world size four. Compact provenance is saved in
`runs/world_3072/checkpoint-provenance.json`; large binaries remain on the server.

Supervisor 142087 automatically advanced to `joint_flow_consistent_3072`, using
launcher 228925 and training ranks 229056/229059/229060/229061. Actual processes
point to clean frozen `repo-throughput` at `56419ed` and the selected
`campaign/runs/world_3072/best.pt`. Only GPUs 0–3 are occupied; GPUs 4–7 are
empty. No restart, source change or scope adjustment was needed. Online run
`32710958` was verified running at step 200; the later snapshot reached step 451
with 128.055 charged optimization seconds (0.142284 GPU-hours). The latest
sampled loss at step 450 was finite at 2.38756. Its first durable checkpoint
exists; the consistency term is still in the registered initial warmup.

Downloaded the updated snapshot through `target_server_2_cf`. Verified all four
validation reports contain the same 104 IDs, episode hashes and reset seeds,
eight per task, with seed 3072, the 10-second cap and every timeout retained as
a failure. Preserved the fourth validation, world completion, new head metadata,
loss samples, checkpoints, compute ledgers and online evidence. Next: continue
joint-flow training and its four validations, then LeFlow and HWM under the
same allowances. Keep the 3,200-reset final tests sealed until all registered
models finish; no additional experiments are queued.

### 2026-10-08 20:01 UTC — joint-flow consistency active; training healthy

Supervisor 142087, launcher 228925 and the four joint-flow ranks remain healthy
in clean frozen `repo-throughput` at `56419ed`, using only GPUs 0–3. GPUs 4–7 are
empty; the four-method, seed-3072 scope and original compute limits are unchanged.
No restart or source change was needed. Shared world completion and all four
CEM validations remain preserved; no learned-head validation or final test has
run yet.

The snapshot reached step 3,704 with 1,006.832 charged optimization seconds
(1.118702 GPU-hours), no validation charge and no last-update overrun. All saved
numeric metrics are finite, and training steps/charged seconds are monotonic.
At step 3,700, total loss was 1.422912, action objective 0.189655, state objective
1.053448 and generated consistency 2.116640 with weight 0.08495. The first sampled
positive consistency weight occurs at step 2,050, after the registered 10% warmup;
the ramp is operating as configured. These are training diagnostics, not task
success results. The durable `last.pt` remains present at 45,396,953 bytes.

W&B API readback at 20:02:00 UTC verified `32710958` running at step 3,550 with
active consistency weight; `fbcl549w` remains finished. Downloaded updated loss
samples, charged compute, checkpoint metadata and online evidence through
`target_server_2_cf`. Next: continue to the first registered 104-case joint-flow
validation milestone, then the remaining unchanged queue. Final testing stays
sealed until all registered models finish.

### 2026-10-08 20:16 UTC — first joint-flow validation completed and verified online

Joint-flow step 5,000 completed its first registered 104-case validation with
23 successes (22.115% macro), retaining all 81 controller timeouts (77.885%) as
failures. Drawer-close achieved 8/8, door-close 6/8, handle-press 5/8 and dial-turn
4/8; the other nine tasks had zero successes. Success within 50/100/200 primitive
actions was 12/19/23 out of 104. Mean controller time was 9.012 seconds/episode,
mean step latency 173.47 ms and p95 177.43 ms; recorded mean safe-boundary overrun
was 0.02690 seconds. The per-episode report preserves every outcome.

Verified identical episode IDs, hashes, reset seeds, training seed 3072 and the
10-second cap against CEM using the selected step-20,000 shared world. All seven
CEM-success cases also succeeded under this joint-flow checkpoint, another 16
changed to success, and 81 failed under both. This is useful early validation
evidence under the matched controller allowance, not a final test result or an
established advantage over the other learned methods. Joint-flow is still
training and LeFlow/HWM have not yet trained. No selection rule, budget or
follow-up experiment was changed in response to these results.

The first round charged 527.507 validation seconds (0.586118 GPU-hours), separate
from optimization. Training resumed normally and the later snapshot reached
step 5,355, charging 1,457.400 optimization seconds (1.619333 GPU-hours), with no
last-update overrun. At step 5,350, finite loss was 1.407499, generated consistency
1.921402 and consistency weight 0.1. Both `best.pt` and `last.pt` are present at
45,396,953 bytes each. Fresh W&B API readback at 20:18:23 UTC verified run
`32710958` online at step 5,100 with the 104-episode result and separate validation
cost. The completed world run remains finished.

Supervisor 142087 and the same four training ranks remain healthy in clean
frozen `repo-throughput` at `56419ed`; only GPUs 0–3 are occupied, with 4–7 empty.
Both scope-change records still match the authorized protocol. Downloaded the
snapshot through `target_server_2_cf` and verified finite metrics, monotonic
charged compute, 104 unique paired cases and all timeout failures. Preserved
`runs/joint_flow_consistent_3072/validation/step_0005000.json`, updated checkpoints,
loss samples, compute ledger and fresh online readback. Next: complete the
remaining three joint-flow validations and the registered LeFlow/HWM training
queue. Final testing remains sealed until all registered models finish.

### 2026-10-08 20:31 UTC — joint-flow advancing toward second validation

Supervisor 142087, launcher 228925 and all four joint-flow ranks remain healthy
in clean frozen `repo-throughput` at `56419ed`. Only GPUs 0–3 are occupied;
GPUs 4–7 are empty. Both scope-change records retain the four methods, seed 3072
and original limits. No intervention was needed.

The snapshot reached step 8,297, charging 2,273.521 optimization seconds
(2.526135 GPU-hours), with the first validation cost unchanged at 527.507 seconds
(0.586118 GPU-hours). No last-update overrun is recorded. At step 8,250, finite
training loss was 1.345198, generated consistency 1.829613 and consistency weight
0.1. Both saved checkpoints remain present; the only completed head validation
is still the preserved 23/104 result at step 5,000.

W&B API readback at 20:31:58 UTC verified `32710958` running at step 8,150 and
`fbcl549w` finished. Downloaded updated metrics, compute usage, checkpoint metadata
and online evidence through `target_server_2_cf`; checked finite saved metrics
and nondecreasing charged optimization. Next: continue to the second registered
validation and preserve its full paired outcomes. LeFlow/HWM and final testing
remain pending; the test stays sealed and scope is unchanged.

### 2026-10-08 20:46 UTC — joint-flow step 10,000; second validation active

Supervisor 142087, launcher 228925 and all four joint-flow ranks remain healthy
in clean frozen `repo-throughput` at `56419ed`, using only GPUs 0–3 while 4–7 are
empty. Scope-change records still match the authorized four methods, seed 3072
and compute limits. The second validation is advancing through its episode logs;
no second completed result is available in this snapshot. No intervention was
needed.

At step 10,000, finite training loss is 1.305469, generated consistency 1.811364
and consistency weight 0.1. The ledger charges 2,742.554 optimization seconds
(3.047282 GPU-hours), with no last-update overrun. The first completed validation
remains separately charged at 527.507 seconds (0.586118 GPU-hours); the active
round's cost will be added at completion. Both checkpoints remain preserved.
W&B API readback at 20:46:59 UTC confirmed `32710958` running at step 9,950 and
`fbcl549w` finished.

Downloaded and checked the updated snapshot through `target_server_2_cf`, with
finite numeric metrics and nondecreasing charged compute. Saved the new loss
samples, checkpoint metadata, compute ledger and online evidence. Next: preserve
the completed second joint-flow validation and continue the registered queue.
LeFlow/HWM are pending, and final testing remains sealed.

### 2026-10-08 21:01 UTC — second joint-flow validation declined; training resumed

The step-10,000 joint-flow validation completed with 17/104 successes (16.346%),
down from 23/104 at step 5,000. Drawer-close achieved 6/8, door-close 5/8,
handle-press 5/8 and dial-turn 1/8; the other nine tasks again had zero successes.
All 87 failures exhausted the controller allowance and remain in the denominator.
Success within 50/100/200 primitive actions was 11/15/17 out of 104. Mean
controller time was 9.129 seconds/episode, mean step latency 173.01 ms and p95
177.35 ms; mean safe-boundary overrun was 0.02842 seconds. Mean return increased
to 120.107 while success declined; observed subgoal cosine decreased to 0.263758.

Paired failure inspection found 15 successes under both checkpoints, eight
success-to-timeout transitions, two failure-to-success transitions and 79
failures under both. The eight lost successes were dial-turn IDs 00001/00004/
00005/00007, door-close 00000/00005 and drawer-close 00001/00007. The two gains
were dial-turn 00002 and door-close 00001. Against CEM with the selected shared
world, all seven CEM successes remain successful and ten additional cases
succeed. These intermediate results neither establish the final ranking nor
isolate the cause of the decline. The registered selection rule still favors
the step-5,000 checkpoint; its saved file metadata is unchanged. No tuning,
rerun, extra seed or ablation was launched in response.

Training resumed normally. The snapshot reached step 13,307 with 3,655.222
charged optimization seconds (4.061357 GPU-hours); both validation rounds total
1,057.890 seconds (1.175433 GPU-hours), including 530.383 seconds for round two.
At step 13,300, finite training loss was 1.282893, generated consistency 1.782080
and consistency weight 0.1. No last-update overrun is recorded; both checkpoints
remain present at 45,396,953 bytes. W&B API readback at 21:01:58 UTC verified
`32710958` running at step 12,650 with the second validation metrics online;
the shared world run remains finished.

Supervisor 142087, launcher 228925 and the same four training ranks remain
healthy in clean frozen `repo-throughput` at `56419ed`. Only GPUs 0–3 are occupied,
with 4–7 empty, and both scope-change records retain the original limits.
Downloaded the snapshot through `target_server_2_cf`; verified 104 unique paired
IDs, episode hashes, reset seeds, 13 tasks with eight cases each, training seed
3072, the 10-second cap and timeout retention against both the first joint-flow
round and selected-world CEM. Saved the complete second validation, finite loss
samples, monotonic charged compute, checkpoint metadata and online evidence.
Next: complete the remaining two joint-flow validations and the unchanged
LeFlow/HWM queue. Final testing stays sealed until all registered models finish.

### 2026-10-08 21:16 UTC — joint-flow step 15,000; third validation active

Supervisor 142087, launcher 228925 and all four joint-flow ranks remain active
in clean frozen `repo-throughput` at `56419ed`, using only GPUs 0–3 while 4–7 are
empty. Scope-change records retain the four methods, seed 3072 and original
limits. The third validation is advancing through its episode logs; only the
first two completed validation reports exist in this snapshot. No intervention
was needed.

At step 15,000, finite training loss is 1.258740, generated consistency 1.771354
and consistency weight 0.1. The ledger charges 4,121.427 optimization seconds
(4.579363 GPU-hours), with no last-update overrun. The first two validations
remain separately charged at 1,057.890 seconds (1.175433 GPU-hours); the active
round will be added at completion. Both checkpoints remain preserved, and the
step-5,000 best checkpoint's file metadata is unchanged. W&B API readback at
21:17:05 UTC confirmed `32710958` running at step 14,950 and `fbcl549w` finished.

Downloaded the updated snapshot through `target_server_2_cf`; verified finite
metrics, increasing training steps and nondecreasing charged optimization.
Preserved loss samples, checkpoint metadata, compute ledger and fresh online
evidence. Next: preserve the completed third validation and continue the
registered queue. LeFlow/HWM are pending, and no final test report exists;
testing remains sealed until all registered models finish.

### 2026-10-08 21:31 UTC — third joint-flow validation complete; first remains best

Joint-flow step 15,000 completed its third validation with 16/104 successes
(15.385%), compared with 23/104 and 17/104 in the first two rounds. Drawer-close
and handle-press each achieved 7/8; door-close achieved 2/8, and the other ten
tasks had no successes. All 88 controller timeouts remain failures. Success
within 50/100/200 primitive actions was 13/16/16 out of 104. Mean controller
time was 8.986 seconds/episode, mean step latency 173.17 ms and p95 177.24 ms;
mean safe-boundary overrun was 0.02290 seconds. Mean return was 114.161 and
observed subgoal cosine was 0.253515.

Paired failure inspection against round two found 13 successes under both,
four success-to-timeout transitions, three failure-to-success transitions and
84 failures under both. Lost successes were dial-turn 00002 and door-close
00001/00002/00007; gains were drawer-close 00007 and handle-press 00005/00007.
Against round one there were nine lost successes and two gains. All seven
selected-world CEM successes remain successful, plus nine other cases. The
step-5,000 checkpoint still has the highest validation success and its file
metadata is unchanged. The current results do not identify the cause of the
decline or establish a ranking against the untrained LeFlow/HWM methods.

A possible follow-up after the four-method first pass is a matched comparison
with generated-plan consistency disabled, keeping the joint planner and all
other settings fixed. That would test the regularizer's contribution; these
checkpoint comparisons cannot isolate it. This is a proposal only, requiring
the user's authorization; no ablation or additional training is queued.

Training resumed and the snapshot reached step 17,531, charging 4,820.500
optimization seconds (5.356111 GPU-hours), with no last-update overrun. Three
validations total 1,574.092 seconds (1.748991 GPU-hours), including 516.202 seconds
for round three. At step 17,500, finite training loss was 1.318035, generated
consistency 1.769762 and consistency weight 0.1. Both checkpoints remain present
at 45,396,953 bytes. W&B API readback at 21:32:02 UTC verified `32710958` running
at step 17,300 with all third-round validation metrics; the world run remains
finished.

Supervisor 142087, launcher 228925 and ranks 229056/229059/229060/229061 remain
healthy in clean frozen `repo-throughput` at `56419ed`, with rank assignments
0–3 and `CUDA_VISIBLE_DEVICES=0,1,2,3`. Only those four GPUs are occupied; 4–7
remain empty. Both scope-change records retain the original limits. Downloaded
the snapshot through `target_server_2_cf` and verified finite metrics, monotonic
charged compute, 104 unique paired IDs/hashes/reset seeds, 13 tasks with eight
cases each, seed 3072, the controller cap and timeout retention against all
earlier joint-flow rounds and selected-world CEM. Preserved the complete third
validation and updated compute, checkpoint, loss and online records. Next:
finish joint-flow's fourth validation, then the unchanged LeFlow/HWM queue.
No final test report exists; final testing remains sealed.

### 2026-10-08 21:46 UTC — joint-flow reached update cap; fourth validation active

Joint-flow reached its 20,000-update ceiling after 5,498.695 optimization seconds
(6.109661 GPU-hours), below the 7,200-second allowance and with no last-update
overrun. The fourth validation is advancing through its episode logs; the
completion record and fourth result are not yet available. The previous three
validations remain separately charged at 1,574.092 seconds (1.748991 GPU-hours).
The active validation's cost will be recorded at completion.

At step 20,000, finite training loss is 1.242981, generated consistency 1.765469
and consistency weight 0.1. Both checkpoints remain present at 45,396,953 bytes;
the first validation's best checkpoint metadata is unchanged. W&B API readback
at 21:47:03 UTC confirmed `32710958` running at step 19,950 with the three
completed validation rounds online; `fbcl549w` remains finished.

Supervisor 142087, launcher 228925 and the same four ranks remain healthy in
clean frozen `repo-throughput` at `56419ed`, assigned only GPUs 0–3, while 4–7
remain empty. Both scope-change records retain the authorized methods, seed and
budgets. No restart or source change was needed. Downloaded the snapshot through
`target_server_2_cf`; checked finite metrics, the update ceiling, monotonic
charged compute and unchanged best checkpoint. Preserved updated losses,
checkpoint metadata, compute usage and online evidence. Next: preserve the
fourth validation and completion record, then verify the supervisor proceeds to
LeFlow on the same allocation. HWM follows; final testing stays sealed and no
test report exists.

### 2026-10-08 22:01 UTC — joint-flow complete; LeFlow started on the same four GPUs

Joint-flow finished at 20,000 updates with stop reason `update_limit`, all four
validations complete and no last-update overrun. Its final optimization charge
is 5,498.695 seconds (6.109661 GPU-hours), below the 7,200-second allowance.
Validation totals 2,102.580 seconds (2.336200 GPU-hours), including 528.489 seconds
for round four. W&B API readback at 22:02:07 UTC verified `32710958` finished with
all four validation rounds online; `fbcl549w` remains finished.

The fourth validation achieved 19/104 successes (18.269%): drawer-close 7/8,
handle-press 6/8, door-close 4/8, dial-turn 1/8 and faucet-open 1/8. The other
eight tasks had zero successes. All 85 controller timeouts remain failures.
Success within 50/100/200 primitive actions was 12/18/19 out of 104. Mean
controller time was 8.934 seconds/episode, mean step latency 172.61 ms and p95
176.70 ms; mean safe-boundary overrun was 0.02170 seconds. Mean return was
116.887 and observed subgoal cosine was 0.251598.

Against round three, 15 cases succeeded under both, four failures became
successes, one success became a timeout and 84 failed under both. Gains were
dial-turn 00003, door-close 00001/00007 and faucet-open 00002; the lost success
was handle-press 00005. Relative to the selected first round, eight successes
were lost and four gained, leaving the first checkpoint best at 23/104. All
seven selected-world CEM successes remained successful in round four, with
twelve additional successes. These validation records do not establish the
four-method ranking or identify the cause of checkpoint differences. The
previously proposed consistency ablation remains unqueued.

CPU-only checkpoint reads and SHA-256 verification confirmed joint-flow's
`best.pt` is step 5,000, with hash
`6ce71f76ea30006d4af0b9fde9f98dd8bcd8c7915445f2057fbe221c905d61ff`.
Its `last.pt` is step 20,000 with validation round four and final compute usage.
Both are preserved at 45,396,953 bytes. The selected world hash remains
`fc55d9fb694813b4f7543c71ed6477c849f10fbf46e05f88369583555dfbc87a`.
The new LeFlow checkpoint matches that world hash and the shared manifest,
protocol, code, seed 3072 and world size four. Saved compact metadata and hashes
in `runs/joint_flow_consistent_3072/checkpoint-provenance.json`; no model binaries
were transferred to the reporting checkout.

Supervisor 142087 automatically advanced to `leflow_adapted_3072`, launcher
362482 and ranks 362602/362603/362604/362605. The completed joint-flow workers
have exited. The new ranks run in clean frozen `repo-throughput` at `56419ed`,
assigned to GPUs 0–3; GPUs 4–7 remain empty. Both scope-change records retain
the original methods, one seed and limits. LeFlow's snapshot reached step 2,157
with 589.563 charged optimization seconds (0.655070 GPU-hours), no validation
charge and no last-update overrun. At step 2,150, finite loss was 1.012480,
inverse objective 0.012926 and observed consistency 0.026058. Its `last.pt` is
present at 73,951,425 bytes. W&B verified run `9jhufn8j` running at step 1,950:
https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/9jhufn8j.

Downloaded the snapshot through `target_server_2_cf`; verified completion and
budget accounting, finite metrics, increasing steps and the fourth validation's
104 unique paired IDs/hashes/reset seeds, eight per task, cap and timeout
retention against all earlier joint-flow rounds and selected-world CEM. Saved
the final head validation, completion, charged compute, checkpoints, online
records and new LeFlow training evidence. Next: complete LeFlow's four registered
validations, then HWM. Final testing remains sealed until all registered models
finish; no test report exists and no extra experiment was started.

### 2026-10-08 22:16 UTC — LeFlow step 5,000; first validation active

LeFlow reached step 5,000 and is running its first registered 104-case validation.
The episode logs are advancing, but no complete validation report exists in this
snapshot. Its ledger charges 1,356.000 optimization seconds (1.506667 GPU-hours),
with no last-update overrun; validation is charged separately at completion.
At step 5,000, finite training loss is 0.934611, inverse objective 0.007828,
observed consistency 0.027117 and state objective 0.924072. These training
diagnostics are not evaluation scores. Its `last.pt` remains present at
73,951,425 bytes.

Supervisor 142087, launcher 362482 and ranks 362602/362603/362604/362605 remain
healthy in clean frozen `repo-throughput` at `56419ed`, using only GPUs 0–3 while
4–7 remain empty. Both scope-change records retain the four methods, seed 3072
and unchanged budgets. W&B API readback at 22:17:01 UTC verified `9jhufn8j`
running at step 4,950; completed world and joint-flow runs remain finished with
their checkpoints, all validation results and final compute charges preserved.
No restart or source change was needed.

Downloaded the snapshot through `target_server_2_cf`, checked finite metrics and
monotonic training steps/charged time, and saved updated loss samples, checkpoint
metadata, compute usage and online evidence. Next: preserve LeFlow's first
complete validation and its paired outcomes, then continue the remaining
registered validations and HWM queue. Final testing stays sealed; no test report
exists and no follow-up experiment is queued.

### 2026-10-08 22:31 UTC — first LeFlow validation: 26/104; paired outcomes preserved

LeFlow step 5,000 completed its first validation with 26/104 successes (25.00%),
retaining all 78 controller timeouts as failures. Door-close achieved 8/8,
handle-press 7/8, drawer-close 6/8 and coffee-button 5/8; the other nine tasks
had zero successes. Success within 50/100/200 primitive actions was 14/26/26 out
of 104. Mean controller time was 8.985 seconds/episode, mean step latency 212.52
ms and p95 216.39 ms; mean safe-boundary overrun was 0.01794 seconds. Mean return
was 107.185 and observed subgoal cosine was 0.329316.

Compared with joint-flow's selected step-5,000 validation checkpoint (23/104),
17 cases succeeded under both, nine only under LeFlow, six only under joint-flow
and 72 failed under both. LeFlow-only successes were five coffee-button cases,
two door-close and two handle-press cases; joint-flow-only successes were four
dial-turn and two drawer-close cases. Thus LeFlow's early point estimate is
three successes higher, with differing failure profiles. This small validation
difference is not a final test result or evidence of a stable advantage across
training seeds. LeFlow has three validations remaining and HWM is untrained.
Against CEM using the selected shared world, all seven CEM successes also
succeeded under LeFlow, with nineteen additional successes and 78 shared
failures. No method, selection rule, data or budget was changed in response.

The first validation charged 473.608 seconds (0.526232 GPU-hours), separately
from optimization. Training resumed and the snapshot reached step 7,032 with
1,904.169 charged optimization seconds (2.115743 GPU-hours) and no last-update
overrun. At step 7,000, finite loss was 0.925023, inverse objective 0.006535 and
observed consistency 0.026325. Both `best.pt` and `last.pt` are present at
73,951,425 bytes each. W&B API readback at 22:32:03 UTC verified `9jhufn8j`
running at step 6,900 with the complete first validation online. The shared
world and joint-flow runs remain finished with their final charges preserved.

Supervisor 142087, launcher 362482 and ranks 362602/362603/362604/362605 remain
healthy in clean frozen `repo-throughput` at `56419ed`, using only GPUs 0–3;
4–7 remain empty. Both scope-change records retain the original authorization.
Downloaded the snapshot through `target_server_2_cf`; verified finite metrics,
monotonic training steps/charged time and 104 unique paired validation IDs,
episode hashes and reset seeds against selected joint-flow and selected-world
CEM. All 13 tasks have eight cases, seed 3072 and the 10-second cap; every
timeout remains a failure. Saved the complete first LeFlow validation, loss
samples, compute usage, checkpoint metadata and online evidence. Next: finish
LeFlow's remaining three validations, then HWM. Final testing remains sealed;
no test report exists and no follow-up experiment is queued.

### 2026-10-08 22:46 UTC — LeFlow step 10,000; second validation active

LeFlow reached step 10,000 and is running its second registered validation.
The episode logs are advancing; no complete second-round result exists in this
snapshot. Its ledger charges 2,702.708 optimization seconds (3.003009 GPU-hours),
with no last-update overrun. The first validation remains separately charged at
473.608 seconds (0.526232 GPU-hours); the active round is charged at completion.
At step 10,000, finite loss is 0.909729, inverse objective 0.003329 and observed
consistency 0.027048. Both checkpoints remain preserved at 73,951,425 bytes each,
and the first validation's best checkpoint metadata is unchanged.

Supervisor 142087, launcher 362482 and ranks 362602/362603/362604/362605 remain
healthy in clean frozen `repo-throughput` at `56419ed`, assigned only GPUs 0–3;
4–7 remain empty. Both scope-change records retain the original authorization.
W&B API readback at 22:47:01 UTC verified `9jhufn8j` running at step 9,950 with
the first validation online. The world and joint-flow runs remain finished,
with their results and final compute charges preserved. No intervention was
needed.

Downloaded the snapshot through `target_server_2_cf`; verified finite metrics,
increasing training steps, nondecreasing charged time and unchanged best
checkpoint metadata. Saved updated losses, checkpoints, compute usage and online
evidence. Next: preserve the completed second LeFlow validation and continue the
registered queue. HWM remains pending; final testing stays sealed, no test
report exists and no additional experiment is queued.

### 2026-10-08 23:01 UTC — second LeFlow validation: 20/104; first checkpoint retained

LeFlow step 10,000 completed its second validation with 20/104 successes
(19.23%), down from 26/104 (25.00%). All 84 controller timeouts remain failures.
Drawer-close and handle-press achieved 7/8 each, door-close 3/8, coffee-button
2/8 and dial-turn 1/8; the other eight tasks had zero successes. Success within
50/100/200 primitive actions was 10/20/20 out of 104. Mean controller time was
9.120 seconds/episode, mean step latency 212.24 ms and p95 216.57 ms; mean
safe-boundary overrun was 0.01800 seconds. Mean return was 118.329 and observed
subgoal cosine was 0.321150.

The paired first-to-second-round outcomes were 17 shared successes, nine lost
successes, three gains and 75 shared failures. The nine losses all became
timeouts: coffee-button cases 00001/00002/00003, door-close
00000/00001/00003/00004/00007 and drawer-close 00003. Gains were dial-turn 00000
and drawer-close 00001/00007. Training loss decreased while validation success
decreased; this observation alone does not establish a cause. The registered
selection rule retains the first, step-5,000 checkpoint at 26/104, whose best
checkpoint metadata is unchanged. Against selected joint-flow, this second
round had 15 shared successes, eight joint-only and five LeFlow-only successes,
with 76 shared failures. Against selected-world CEM it had six shared successes,
one CEM-only and fourteen LeFlow-only successes, with 83 shared failures. These
are validation diagnostics, not a final ranking; no experiment was changed.

Round two charged 473.457 seconds. The two completed validations total 947.065
seconds (1.052295 GPU-hours), separately from optimization. Training resumed and
the snapshot reached step 11,969 with 3,229.996 charged optimization seconds
(3.588885 GPU-hours) and no last-update overrun. At step 11,950, finite loss was
0.908079, inverse objective 0.003550 and observed consistency 0.027068. Both
checkpoints remain present at 73,951,425 bytes each. W&B API readback at 23:02:06
UTC verified `9jhufn8j` running at step 11,800 with both validation results
online. The world and joint-flow runs remain finished with their results and
final compute charges preserved.

Supervisor 142087, launcher 362482 and ranks 362602/362603/362604/362605 remain
healthy in clean frozen `repo-throughput` at `56419ed`, using only GPUs 0–3;
4–7 remain empty. Both scope-change records retain the original authorization.
Downloaded the snapshot through `target_server_2_cf`; verified finite metrics,
monotonic training steps/charged time and all 104 unique paired IDs, episode
hashes and reset seeds against the first LeFlow round, selected joint-flow and
selected-world CEM. All 13 tasks retain eight cases, seed 3072 and the 10-second
cap; every timeout remains a failure. Saved the second validation, losses,
compute usage, checkpoint metadata and online evidence. Next: complete LeFlow's
remaining two validations, then HWM. Final testing stays sealed; no test report
exists and no follow-up experiment is queued.

### 2026-10-08 23:16 UTC — LeFlow step 15,000; third validation active

LeFlow reached step 15,000 and is running its third registered validation.
Episode logs are advancing; the completed third-round report is not yet present
in this snapshot. Optimization has charged 4,038.653 seconds (4.487392 GPU-hours),
with no last-update overrun. The first two validations remain separately charged
at 947.065 seconds (1.052295 GPU-hours); the active round is charged at completion.
At step 15,000, finite training loss is 0.904686, inverse objective 0.002967 and
observed consistency 0.026358. These diagnostics are not task success scores.
Both checkpoints remain present at 73,951,425 bytes each; the first validation's
best checkpoint metadata remains unchanged, retaining its 26/104 selection.

Supervisor 142087, launcher 362482 and ranks 362602/362603/362604/362605 remain
healthy in clean frozen `repo-throughput` at `56419ed`, assigned only GPUs 0–3;
4–7 remain empty. Both scope-change records retain the four authorized methods,
seed 3072 and unchanged budgets. W&B API readback at 23:17:27 UTC verified
`9jhufn8j` running at step 14,950 with both completed validations online. The
world and joint-flow runs remain finished, with results and final compute
charges preserved. No restart or code change was needed.

Downloaded the snapshot through `target_server_2_cf`; verified finite metrics,
increasing training steps and charged time, unchanged best checkpoint metadata
and the fixed compute allowance. Saved updated losses, compute usage, checkpoint
metadata and online evidence. Next: preserve LeFlow's completed third validation
and continue the registered queue. HWM remains pending. Final testing stays
sealed; no test report exists and no follow-up experiment is queued.

### 2026-10-08 23:31 UTC — third LeFlow validation: 25/104; first checkpoint retained

LeFlow step 15,000 completed its third validation with 25/104 successes (24.04%),
up from 20/104 in round two but below the selected first checkpoint's 26/104.
All 79 controller timeouts remain failures. Handle-press achieved 8/8,
coffee-button and door-close 5/8 each, drawer-close 4/8 and dial-turn 3/8; the
other eight tasks had zero successes. Success within 50/100/200 primitive
actions was 10/25/25 out of 104. Mean controller time was 9.017 seconds/episode,
mean step latency 212.40 ms and p95 216.71 ms; mean safe-boundary overrun was
0.01723 seconds. Mean return was 117.880 and observed subgoal cosine 0.317288.

Paired against round two, 16 cases succeeded under both, nine previous failures
succeeded, four successes became timeouts and 75 failed under both. Gains were
coffee-button cases 00001/00002/00003, dial-turn 00001/00005, door-close
00003/00004/00007 and handle-press 00005. Losses were door-close 00006 and
drawer-close 00001/00005/00006. Relative to round one, 20 cases succeeded under
both, five were new successes, six successes were lost and 73 failed under
both. The recovery therefore includes different cases; it does not surpass the
registered best score, and the first checkpoint remains selected unchanged.

Against selected joint-flow, round three had 15 shared successes, ten LeFlow-only
and eight joint-only successes, with 71 shared failures. Against selected-world
CEM it had four shared successes, 21 LeFlow-only and three CEM-only successes,
with 76 shared failures. All three CEM-only cases were drawer-close
00003/00005/00006. These are paired validation diagnostics; the selected LeFlow
checkpoint still has 26/104, and no final method ranking is available. No method,
selection rule, data or compute allowance changed in response to these results.

The third validation charged 461.582 seconds. Three completed validations total
1,408.647 seconds (1.565163 GPU-hours), separately from optimization. Training
resumed and the snapshot reached step 16,988 with 4,569.058 charged optimization
seconds (5.076732 GPU-hours), with no last-update overrun. At step 16,950, finite
training loss was 0.913270, inverse objective 0.003934 and observed consistency
0.027793. Both checkpoints remain present at 73,951,425 bytes each; the best
checkpoint metadata is unchanged. W&B API readback at 23:32:07 UTC verified
`9jhufn8j` running at step 16,750 with all three validation results online.
The world and joint-flow runs remain finished with their final charges preserved.

Supervisor 142087, launcher 362482 and ranks 362602/362603/362604/362605 remain
healthy in clean frozen `repo-throughput` at `56419ed`, using only GPUs 0–3;
4–7 remain empty. Both scope-change records retain the original authorization.
Downloaded the snapshot through `target_server_2_cf`; verified finite metrics,
monotonic training steps/charged time, and all 104 unique paired IDs, episode
hashes and reset seeds against prior LeFlow rounds, selected joint-flow and
selected-world CEM. All 13 tasks retain eight cases, seed 3072 and the 10-second
cap; every timeout remains a failure. Saved the third validation, losses,
compute usage, checkpoint metadata and online evidence. Next: finish LeFlow's
fourth validation, then HWM. Final testing stays sealed; no test report exists
and no follow-up experiment is queued.

### 2026-10-08 23:46 UTC — LeFlow at update cap; fourth validation active

LeFlow reached its 20,000-update cap and is running its fourth registered
validation. Episode logs are advancing; the complete fourth-round result and
completion marker are not yet present in this snapshot. Optimization has charged
5,371.992 seconds (5.968880 GPU-hours), below the 7,200-second allowance, with
no last-update overrun. Three completed validations remain separately charged at
1,408.647 seconds (1.565163 GPU-hours); the active round is charged at completion.
At step 20,000, finite training loss is 0.905439, inverse objective 0.002930 and
observed consistency 0.026815. Both checkpoints remain present at 73,951,425
bytes each; the first validation's best checkpoint metadata remains unchanged.
Its 26/104 remains selected while the fourth validation is incomplete.

Supervisor 142087, launcher 362482 and ranks 362602/362603/362604/362605 remain
healthy in clean frozen `repo-throughput` at `56419ed`, assigned only GPUs 0–3;
4–7 remain empty. Both scope-change records retain the original methods, seed
and limits. W&B API readback at 23:47:07 UTC verified `9jhufn8j` running at step
19,950 with all three completed validations online. The world and joint-flow
runs remain finished with their results and final charges preserved. No restart
or source change was needed.

Downloaded the snapshot through `target_server_2_cf`; verified finite metrics,
increasing training steps and charged time, the update cap, zero last-update
overrun and unchanged best checkpoint metadata. Saved updated losses, compute
usage, checkpoint metadata and online evidence. Next: preserve LeFlow's fourth
validation and completion, then follow HWM on the same four GPUs. Final testing
stays sealed; no test report exists and no follow-up experiment is queued.
