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

Launched 2026-10-09 **04:26 UTC**, supervisor PID 778158, frozen code
`75e0815351eb25c63e495f87458f139bd5062259`. All 59 CPU checks and the four-A800 GPU preflight passed. All cached file hashes
were verified before world training began at approximately **04:29 UTC**.
By 04:30 UTC the world had exceeded **300 optimizer updates**, with finite losses,
a saved checkpoint and online W&B metrics independently read back from the API.
The original campaign remains held.

Live world run: [8vwksovs](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/8vwksovs).
Training metrics are present; the first periodic evaluation has **not** completed
at this launch checkpoint. It triggers at update 5,000 (or 25% of the time cap,
whichever comes first). The initial measured throughput was about 460 samples/s,
so the first trigger was roughly 11 minutes after this checkpoint, followed by
simulation evaluation. This is an estimate, not a completion assertion.

The fresh manifest has SHA-256
`ce6e190b9b777ad34807c47c2dbeebae810f9cca52b440dbaf9e05043d69f1bc`;
its unchanged entry list has digest
`632005d660cdf000ee244461c5ffacee020d8a9d3321535ae0e3e15deb4508a8`.
The new execution protocol is
`563c7d7d8b10664aa1ebf50db8f8a24fc015b50325ff1de82bcb736c7d411d51`.

The existing 15-minute campaign monitor is active with the new authorization,
new root and frozen revision. It will preserve meaningful evaluation results,
complete final comparison/failure analysis, publish compact evidence to main,
and pause when complete. It must not launch a duplicate or another campaign.
