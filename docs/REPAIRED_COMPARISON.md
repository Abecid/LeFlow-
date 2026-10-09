# Fresh repaired comparison — October 9, 2026 UTC

The user's latest correction explicitly authorizes fresh training after the
repairs, periodic evaluation, comparison against the three major baselines, and
failure analysis. This supersedes the previous no-additional-training instruction
for this new campaign. It does not authorize extra seeds, ablations or tuning
against test outcomes.

## Registered execution

- Server root: `/home/mtxu/adam/LeFlow-experiments/20261009-repaired-comparison`.
- Frozen execution checkout: `repo`; records: `campaign`; recipe:
  `config/flow_metaworld_repair.json`; seed **3072**, GPUs **0–3** only.
- Sequential fresh initialization: shared world, `joint_flow_consistent`,
  `leflow_adapted`, `hwm_adapted`. `cem_long` uses the same shared world.
- Each learned model: maximum **7,200 optimization seconds or 20,000 updates**,
  whichever occurs first, global batch 64, microbatch 4, four GPUs. Validation is
  charged separately. Actual time and update counts remain in durable ledgers.
- Four periodic validation rounds at 25/50/75/100% of the earlier training cap;
  the same **104 resets** (8 per training task), with online W&B and local JSON.
  Training metrics every 50 updates. World prediction diagnostics use the same
  fixed validation sampler and world training includes the original CEM checks.
- Final test only after all training: **3,200 identical resets per method**,
  200 per each of 16 tasks, **12,800 total executions**. Same cumulative
  **10-second controller limit** and **200 primitive actions** per episode.
- Test outcomes generate task-balanced paired comparisons and automatic
  `failure-analysis.json`, both attached to the W&B comparison artifact.

## What changed and what is held fixed

The full-dimensional endpoint sampler removes the confirmed retained-noise
restriction in both flow heads. Static repeated-current-frame features align
states, hindsight goals and image goals across every method. Continuous proposal
scoring rolls actions from the actual start without resetting at imaginary
waypoints; both flow methods use it. Our consistency sampler uses eight steps,
matching inference. The endpoint objective has changed to clean-state MSE and
static inputs give up temporal history; improvement is not assumed.

All models are freshly initialized; no old trained world or planner weights,
optimizer state, completion files or compute ledgers enter the new campaign.
The official frozen V-JEPA encoder and existing feature cache are reused.
The original **11,650 selected entries** (7,800 train, 650 validation, 3,200 test),
13 training tasks, 3 held-out tasks, exact resets, goal screening, episode file
hashes, encoder fingerprint and original train-only normalization are unchanged.
Planner training uses the same successful expert trajectory windows as before;
the shared world uses the same expert/random mixture. Cache reuse is verified
with all episode hashes and an explicit manifest-equality record.

The original root `20261007-joint-flow` remains held with its frozen source,
selected checkpoints and ledgers intact. Its 236 partial test records are
preserved and were not read for repair development. Prior validation values
(ours 23/104, LeFlow 27/104, HWM 8/104, CEM 7/104) are historical references;
the definitive new comparison uses all freshly trained repaired-protocol models.

## Failure analysis fixed before results

For every method, report per-task success and distinguish controller-time
exhaustion, primitive-action exhaustion and early environment endings. Compare
steps, controller latency, predicted costs and observed subgoal distance for
successful versus failed episodes. Pair ours with every baseline on identical
reset IDs, retaining all failed case IDs, reset seeds and episode hashes,
especially baseline-success/ours-failure cases. Separate seen and held-out tasks
in the comparison. These records describe operational failures; they cannot
establish grasp, collision or contact causes without visual/physical evidence.
Proxy-score associations do not prove useful executable progress.

Report negative findings without changing this campaign. Any proposed later
fixes or ablations must be justified by these results and separately authorized;
no additional training or rollouts are launched by the analysis.

## Launch status

Preparing a clean frozen checkout and running CPU regression checks plus the
registered four-GPU preflight. Live status and first evaluation evidence will
be appended here and in `PROGRESS.md` after actual execution is verified.
