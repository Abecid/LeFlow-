# Joint flow planning on V-JEPA 2.1 / MetaWorld

This is the active experiment. BTM is not part of its model, training, or campaign.
Historical BTM files remain for reproducibility; do not launch `run_comparison.py`
for this experiment.

## Fixed scientific target

Primary outcome: closed-loop MetaWorld v3 success within 200 primitive actions,
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
  Fine and coarse world dynamics see both collection modes. Planners use successful expert
  episodes with hindsight goals, sampling starts before task completion. No training episodes from the three held-out
  tasks. The success rate of the collecting expert is logged, not assumed.
- 50 validation resets per training task, separate from all training resets.
  Intermediate evaluation uses a fixed 32 per task. Only validation outcomes
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
- Three training seeds (3072/3073/3074). All methods share reset IDs and goal
  images. 3,200 test episodes/model, 9,600 executions/method over three seeds;
  repeated model-seed evaluations are **not** 9,600 independent environments.
- Report environment success, first-success time, episode return, full-controller
  mean/p95 latency, world-model calls, and predicted/observed subgoal distance.
  Goal images are allowed task specifications; expert subgoals are never exposed.
- The primary comparison is joint_flow_consistent versus leflow_adapted. Report
  all comparisons even if the method loses. Confidence intervals resample training
  seeds and paired resets within tasks; correct secondary comparisons for multiple
  comparisons. With 200 resets, a per-task success CI can still be about +/-6.9 pp
  near 50%; do not infer small per-task improvements from point estimates.

## Comparators and attribution

1. CEM over the same JEPA dynamics, horizons 5 and 30.
2. LeFlow-style flow paths plus a separate inverse model and rollout selection
   (arXiv:2608.24855), adapted to the shared V-JEPA token representation.
3. HWM-style learned macro-actions and coarse/fine CEM (arXiv:2604.03208), adapted
   to the shared representation and data.
4. A matched 2 x 2: deterministic / flow, each with / without generated consistency.

The LeFlow/HWM ports are method-family comparisons, **not exact reproductions of
published SOTA checkpoints**. Published results use different data and protocols
and must not be inserted into this measured comparison table. The campaign logs
parameter counts, training updates, controller settings and actual latency.
Equal candidate counts alone do not establish equal compute. A separate
validation-calibrated latency-matched comparison is required for efficiency claims.

## Launch and monitoring

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

Training logs online to W&B and to local JSONL every 50 optimizer steps. World
validation reports prediction loss, no-motion persistence loss, and action
identification among 16 candidates. Planner validation runs 32 fixed episodes per
training task (416 total) every 5,000 updates, including the final update. The
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
| `test/*.json`, `test/*.json.episodes/` | Final reports and durable individual episodes |
| `comparison.json` | Paired three-seed comparison and confidence intervals |

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
renders repeatable observations. The corrected four-GPU preflight and subsequent
stages run under a detached supervisor. See [PROGRESS.md](PROGRESS.md) and the
versioned [runtime reports](reports/20261007-joint-flow/) for the latest verified
stage, failures, run links, and metrics. Historical runtime access blockers above
must not be assumed to describe the current desktop session. There are no
measured success improvements or SOTA claims yet.
