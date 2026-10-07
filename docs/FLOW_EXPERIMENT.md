# Joint flow planning on V-JEPA 2.1 / MetaWorld

This is the active experiment. BTM is not part of its model, training, or campaign.
Historical BTM files remain for reproducibility; do not launch `run_comparison.py`
for this experiment.

## Fixed scientific target

Primary outcome: closed-loop MetaWorld v3 success within 200 primitive actions
and 10 seconds of cumulative controller time per episode,
averaged equally over 16 tasks. Report the 13 training-task and 3 task-held-out
groups separately. The task list follows *The Planning Limits of Latent World
Models* (arXiv:2609.39235), but this code is an independent implementation, not
the authors' released experiment or a claim to reproduce their scores.

The official V-JEPA 2.1 ViT-L encoder is frozen. Causal 16-frame image histories
are encoded and pooled to a 2 x 4 x 4 spatial-temporal token grid: [32,1024].
Training-only feature statistics standardize every method's inputs and targets.
The downstream dynamics model uses one such history embedding as its state.
This differs from the paper's four-embedding history predictor; comparisons in
this campaign share the same implementation. No simulator state enters a learned
planner; privileged state is used only by the scripted data-collection expert.

The joint planner predicts intermediate visual embeddings and action chunks.
For M=12, K=5 control steps, and two primitive actions per control step, outputs
are subgoals [B,11,32,1024] and actions [B,12,5,8]. Endpoints are observations.
Generated-plan consistency compares each chunk's frozen-world-model rollout to
its next proposed subgoal; both endpoint and action gradients reach the planner.
This regularizer is not a certificate of simulator reachability.

## Data and evaluation rules

- 600 independent training resets per training task (80% expert / 20% random).
  The one shared fine world model sees both collection modes. All three learned
  planners use the same successful expert episodes and sampled trajectory windows,
  with hindsight goals and starts before task completion. HWM learns its coarse
  transitions from every five-step chunk in those same windows. No training episodes from the three held-out
  tasks. The success rate of the collecting expert is logged, not assumed.
- 50 validation resets per training task, separate from all training resets.
  Intermediate evaluation uses a fixed 8 per task (104 episodes). Only validation outcomes
  select checkpoints or tune hyperparameters.
- 200 **distinct test resets per task**, with valid single-image goal annotations.
  Before any model is trained, collect a fixed stream of 400 candidate resets
  per task and retain the first 200 for which the scripted expert reaches the
  task goal within 200 primitive actions. Use the last successful RGB frame.
  Validation similarly retains 50 valid goals from a fixed pool of 100 resets.
  Record every goal-construction failure and the selection in `goal_screening`.
  If the pool is insufficient, preparation fails instead of shrinking the test.
  This measures the goal-constructible subset; it is not an unconditional success
  rate over all MetaWorld resets. No selection uses a learned method's outcomes.
  The manifest is frozen before training and is shared by all comparisons.
- Exactly **one training seed, 3072**, per the user's correction. All methods
  share reset IDs and goal images: 3,200 test episodes per method. The launcher
  explicitly registers `execution_seeds: [3072]`; no additional seeds are run.
- Report environment success, first-success time, episode return, full-controller
  mean/p95 latency, world-model calls, and predicted/observed subgoal distance.
  Goal images are allowed task specifications; expert subgoals are never exposed.
- The primary comparison is joint_flow_consistent versus leflow_adapted. Report
  all comparisons even if the method loses. Confidence intervals resample paired
  resets within tasks; they do not measure variation across training runs. Correct secondary comparisons for multiple
  comparisons. With 200 resets, a per-task success CI can still be about +/-6.9 pp
  near 50%; do not infer small per-task improvements from point estimates.

## Comparators and attribution

Exactly four methods run in this first pass:

1. **Proposed joint flow + generated-plan consistency** (`joint_flow_consistent`).
2. **LeFlow adaptation** (`leflow_adapted`): latent flow paths, separate inverse
   model, ranking by predicted action outcomes. Primary baseline.
3. **HWM adaptation** (`hwm_adapted`): learned macro-actions, coarse/fine planning.
4. **JEPA/CEM, horizon 30** (`cem_long`): direct action search through the same
   fine world model; the stronger long-horizon reference in the planning-limits study.

No ablations, additional seeds, or extra methods are queued. The evidence and
selection rationale are in [FIRST_PASS.md](FIRST_PASS.md). LeFlow/HWM are method
adaptations to a common encoder and dataset, not exact published-SOTA reproductions.

Each learned model gets the same 7,200-second optimization allowance on the same
four GPUs, with a 20,000-update ceiling. The fine world model is trained once with
that same cap and shared by all methods. CEM has no extra learned head. This gives
an upper training allowance of 32 GPU-hours plus the final atomic update, excluding
shared preprocessing and separately metered validation. Caps are checked between
optimizer updates; the last update's overrun and actual usage are logged. Usage
is durably recorded each update and retained across resume. Methods may use fewer
updates or less time; no dummy work is added to make consumption identical.

Each controller gets 10 seconds cumulatively per evaluation episode, including
image-history preparation, encoding, planning, and action conversion. Environment
rendering/stepping is separate. Checks inside flow/CEM/rollout loops stop work at
safe boundaries. A running tensor operation can finish beyond the deadline; that
overrun is logged and its late action is not executed. Such episodes remain in
the denominator. All methods also share the 200-primitive-action limit.

Periodic validation occurs at 25%, 50%, 75%, and 100% of the earlier time/update
cap, with the same 104 resets. Loss/system metrics are logged every 50 updates.
CEM validation runs at those milestones while the shared world is being trained;
its final test uses exactly the same selected frozen world as all learned heads.
The final comparison contains 12,800 episode executions: four methods times
3,200 paired resets. There is no automatic ablation phase after it.

## Launch and monitoring

The active resumed campaign passes the preserved original configuration through
`--data-config` solely for feature preparation/cache compatibility. Its current
`--config` contains exactly the four methods, seed 3072 and the explicit budgets.
After preparation, the original manifest is archived as `data-manifest.json`,
and the execution manifest retains all episode hashes, reset IDs, goal screening
and feature statistics while recording the new execution protocol. Changing an
encoder, camera, reset distribution, or split fails the compatibility check.

Use branch `research/joint-flow-metaworld`. On the authorized server, run:

```bash
bash scripts/bootstrap_flow_server.sh
```

The script creates the `flow-jepa` Conda environment and pins PyTorch 2.8 with
CUDA 12.6. It needs an installed NVIDIA driver compatible with that build, Conda,
outbound access to GitHub/model weights/W&B, and an existing W&B login with access
to project `flow-jepa-metaworld`. It does not print or copy credentials. Set
`FLOW_DATA_DIR` before launching to use a volume with at least 150 GiB free;
the default is `$HOME/flow-jepa-data/v1`. This is a full research campaign, not
a short demonstration. Actual throughput is recorded on the target hardware.

The supervisor checks online W&B, then waits up to seven days for 1, 2, or 4 GPUs
to be idle on three consecutive 30-second checks. It never stops another process.
Its per-user file locks coordinate this launcher only; they are not a reservation
against other users. On Slurm hosts, run it inside a GPU allocation. Resume uses
the original GPU count to retain checkpoint/RNG compatibility.

Before data preparation, the CUDA preflight loads the official encoder, renders
real MetaWorld observations, checks repeatable resets, and runs two distributed
optimizer steps through each distinct training architecture. A failure stops the
campaign. Data preparation then creates the frozen train/validation/test manifest.
It downloads the pretrained encoder and collects simulator data; it does not
silently substitute another dataset if collection fails.

Training logs online to W&B and local JSONL every 50 optimizer steps. World
validation reports prediction loss, persistence loss and action identification.
Closed-loop validation uses 104 fixed episodes at four budget milestones. The
selected checkpoint maximizes validation macro success. World checkpoints use
validation prediction loss. All models finish before any test results are used.

Inspect these files beneath `FLOW_DATA_DIR`:

| File | Meaning |
|---|---|
| `launcher.log`, `logs/*.log` | Supervisor and stage output |
| `queue.json`, `status.json` | GPU waiting state and active/failed/completed stage |
| `gpu_preflight.json` | Actual CUDA/NCCL/encoder wiring results |
| `campaign.json`, `manifest.json` | Frozen code/configuration and episode identity |
| `runs/*/run.json` | Online W&B run links |
| `runs/*/metrics.jsonl` | Intermediate training and evaluation metrics |
| `runs/*/compute_usage.json` | Resumable training and validation time accounting |
| `test/*.json`, `test/*.json.episodes/` | Final reports and durable individual episodes |
| `comparison.json` | Paired single-seed comparison and reset confidence intervals |

Rerun the same bootstrap command to resume. Do not change a registered campaign's
code or configuration. For a changed method, create a separate checkout and data
root; do not tune against the frozen test set. Completed evaluation episodes,
including failures, survive interruption. Hash checks reject reuse after changes
to code, checkpoint, data, or hardware. Final reports are saved before W&B upload;
an upload failure can be retried without executing the episodes again.

## Verification and current execution status

CPU verification covers model gradients, controller paths, split/goal-screening
rules, paired-comparison guards, evaluation resume, fixture training and exact
single-process checkpoint resume. The official 5.15 GB checkpoint loads strictly
and produces finite `[1,32,1024]` features from real MetaWorld RGB frames. A real
scripted expert reaches all 16 task goals on one smoke-test reset per task; this
is a data-collection check, not a learned-method benchmark.

The campaign was recovered on 2026-10-07 in a desktop chat with working SSH to
`target_server_2`. All 29 CPU tests pass on the server, online W&B upload is
verified, and the official encoder checksum is verified. The initial GPU
preflight exposed missing NVIDIA EGL graphics libraries; Mesa software EGL now
renders repeatable observations. The corrected four-GPU preflight passed and real data preparation is running
under a detached supervisor. See [PROGRESS.md](PROGRESS.md) and the
versioned [runtime reports](reports/20261007-joint-flow/) for the latest verified
stage, failures, run links, and metrics. Historical runtime access blockers above
must not be assumed to describe the current desktop session. There are no
measured success improvements or SOTA claims yet.
