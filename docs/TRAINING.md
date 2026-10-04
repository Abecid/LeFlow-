# Train BTM and flow subgoal planners

This fork implements the **LeWM stage** of the JEPA subgoal-planning project:
matched BTM/flow/deterministic waypoint models, hierarchical execution, and actual
environment evaluation. **V-JEPA 2.1 and Meta-World are not implemented in this
release.** The current encoder returns one 192-dimensional vector per image;
adapting a video encoder with spatial patch tokens is a separate representation
and dynamics-training change, not a checkpoint-name substitution.

The default experiment compares two models, trained sequentially, on the same
cached observations, episode split, inverse-dynamics targets, controller, and
candidate budget. The new flow model is a matched baseline for this fork, not an
exact reproduction of the published LeFlow network/training protocol. Original
LeFlow files remain available below the fork instructions in the main README.

Use the [paired campaign launcher](COMPARISON.md) to freeze these shared settings,
group online W&B runs, resume both jobs, and audit final paired evaluation reports.

## 1. Environment on the GPU machine

Use Linux with NVIDIA GPUs and a driver compatible with the CUDA runtime in
PyTorch 2.8. Run these commands on that machine, not a Mac's integrated GPU.
Miniforge/Conda must already be installed.

```bash
git clone https://github.com/Abecid/LeFlow-.git
cd LeFlow-
bash scripts/setup_conda.sh
conda activate btm-jepa

# Choose a disk with room for BOTH compressed and extracted datasets.
export STABLEWM_HOME=/data/leflow
mkdir -p "$STABLEWM_HOME"
wandb login
# Optional team/entity; otherwise W&B uses the authenticated user's default.
export WANDB_ENTITY=your_entity
```

`environment.yml` installs the pinned requirements. `stable-worldmodel` is pinned
to commit `06b1f4c24ccb237a9b2d6bb65bf010a6a5b41e83`, which has the
`World.evaluate_from_dataset` API used by this release. Installing its current
main branch can break the evaluator. PyMunk 7.1 and MuJoCo 3.3.7 are pinned too.

The original `train.py` / `train_latent_planner.py` additionally require
`stable-pretraining`; the new `train_subgoals.py` does not. Use the new trainer for
the distributed BTM/flow experiments below.

## 2. Download, inspect, split, and encode

```bash
# Prints verified HF revisions, compressed size, and intended paths first.
python scripts/prepare_data.py --task pusht --dry-run
python scripts/prepare_data.py --task pusht --remove-archive

# Encode once, using four GPUs. For one GPU, use nproc_per_node=1.
torchrun --nnodes=1 --nproc_per_node=4 \
  --master-addr=127.0.0.1 --master-port=29501 \
  scripts/cache_latents.py --task pusht --batch-size 64

python scripts/preflight.py --task pusht --gpus 4
```

PushT's compressed archive alone is about 13.1 GB. Reacher's is about 23.8 GB,
Cube's about 46.2 GB, and TwoRoom's about 3.4 GB. Extracted sizes are additional;
the downloader checks free space and tar member sizes. Keep a large data volume.
Use `--source /absolute/path/to/existing.h5` to reuse an existing SWM dataset.

Preparation downloads the official LeWM state dict and records the HF revisions
and checkpoint SHA-256. It checks frame counts, episode offsets, RGB layout, and
action statistics. It saves an 80/10/10 train/validation/test **episode** split.
Caching distributes whole episodes across workers and writes one HDF5 shard per
rank; only complete caches receive `manifest.json`. A failed cache build can be
rerun; unfinished shards are recomputed. Raw images are not read during planner
training.

The cache paths are:

```text
$STABLEWM_HOME/prepared/pusht/source.json
$STABLEWM_HOME/checkpoints/pusht/weights.pt
$STABLEWM_HOME/latents/pusht/manifest.json
$STABLEWM_HOME/latents/pusht/rank_00_of_04.h5
```

Other supported source tasks are `tworoom`, `reacher`, and `cube`. Use the same
task in preparation, caching, preflight, and training. These download paths are
implemented; complete training/evaluation on those three tasks has not been run
in the development workspace.

**Split scope:** the released frozen LeWM may already have been trained on these
episodes. The split holds out data from the *new planner*. It does not establish
an unseen-environment or encoder-level generalization result. Action normalization
uses the full source's mean and sample standard deviation to match the pretrained
LeWM convention; those statistics are saved and reused by evaluation.

## 3. Train the matched comparison, at most four GPUs

```bash
# Sequential jobs: never eight GPUs at once.
GPUS=4 bash scripts/train.sh task=pusht method=flow
GPUS=4 bash scripts/train.sh task=pusht method=btm
```

Defaults: 10 epochs, AdamW lr=1e-4, global batch 128, microbatch 32 per GPU,
float32, gradient clipping at 1.0. The global batch stays 128 on 1, 2, or 4 GPUs
through gradient accumulation. With three GPUs, choose a divisible global batch,
for example `global_batch_size=96`. More than four ranks is rejected.

For a first short GPU run:

```bash
GPUS=4 bash scripts/train.sh task=pusht method=btm max_steps=100 \
  validate_every=25 checkpoint_every=25 evaluation.every_steps=100 \
  evaluation.episodes=5 run_dir="$STABLEWM_HOME/runs/pusht/btm_gpu_check" \
  wandb.name=pusht_btm_gpu_check
```

BTM's JVP is implemented with forward-mode AD and a detached target; it does not
build a Hessian graph. Explicit transformer attention avoids unsupported
forward-AD Flash Attention kernels. Float32 is currently required; mixed
precision is deliberately not advertised as verified. Both methods use the same
main transformer dimensions; FM has a small additional time-conditioning MLP.

`method=deterministic` enables the additional MSE waypoint control. Start with
flow and BTM. The default BTM objective has no source-endpoint MSE augmentation
and no distillation teacher.

## 4. W&B metrics and intermediate evaluation

Only rank zero starts a W&B run. `wandb.mode=online` is the default. Authentication
or online initialization failure stops the job rather than silently discarding
metrics. A local `metrics.jsonl` is also written. Every log includes the optimizer
step; training losses are averaged across ranks and accumulated microbatches.

| Metrics | Meaning |
|---|---|
| `train/generative_loss`, `train/inverse_loss`, `train/consistency_loss` | Objective components; FM and BTM generative losses have different meanings |
| `train/transport_loss`, `train/boundary_loss`, `train/jvp_rms` | BTM stationary-map diagnostics |
| `train/grad_norm`, `train/lr`, `system/gpu_peak_memory_gib` | Optimization and memory |
| `val/path_mse_single`, `val/candidate_variance` | First-sample waypoint error and sample diversity; not oracle best-of-N task success |
| `val/boundary_mse` | Checks the clamped endpoints only; not evidence of a correct plan |
| `eval/offset_*/success_rate`, `success_ci95_low`, `success_ci95_high` | Real environment success in **[0,1]**, with Wilson 95% intervals |
| `eval/offset_*/selected_rollout_goal_mse` | Model-predicted goal error after executing decoded actions in the world model |
| `eval/offset_*/observed_subgoal_mse_after_chunk` | Observed latent distance to the previous subgoal after an executed chunk; not a binary physical-reachability metric |
| `eval/offset_*/planning_batch_latency_ms_*`, `world_model_state_predictions` | Full planning cost including verification/refinement |

Validation runs every 500 optimizer steps. Environment evaluation runs every
1,000 steps and at epoch ends, using 20 validation episodes at goal offsets
25/50/100 primitive actions and a 200-action budget. Adjust `evaluation.episodes`
if too few held-out episodes support the longest offset; the trainer checks this
before training. Evaluation runs on rank zero's GPU while the other ranks wait,
so it never needs a fifth GPU. It intentionally pauses training to keep model
versions and metrics synchronized. Subprocess logs and exact evaluation episode
IDs/start indices are saved under the run's `eval/` directory.

`best.pt` is selected by mean validation **task success**, not generative loss.
`last.pt` includes model, optimizer, progress, per-rank RNG state, data fingerprint,
configuration, and W&B run ID. Resume preserves global batch and rank count:

```bash
GPUS=4 bash scripts/train.sh task=pusht method=btm \
  resume="$STABLEWM_HOME/runs/pusht/btm_3072/last.pt"
```

Existing run directories are protected against accidental fresh-run overwrite.
Use a new `run_dir` and W&B name/group for a new experiment.

## 5. Final evaluation and fair controls

```bash
python eval_subgoals.py \
  --checkpoint "$STABLEWM_HOME/runs/pusht/btm_3072/best.pt" \
  --split test --episodes 50 --goal-offset 50 --budget 200 \
  --mode hierarchical --spacing 2 --candidates 64 \
  --output "$STABLEWM_HOME/results/pusht_btm_test50.json"
```

Repeat with the flow checkpoint using the same seed, goals, candidate count,
inverse architecture, and CEM settings. Sweep flow inference over 2/4/8/16 steps
with `--flow-steps`; BTM always makes one path-network call per candidate. The
whole controller still includes inverse prediction, dynamics rollouts, and CEM.
Use `--mode flat --spacing 1` for the no-hierarchy control. Use `--save-video`
only when needed. The final test split is not used for checkpoint selection.

This implementation does not automatically enforce an equal wall-clock budget
between samplers; it logs latency and world-model calls so that a measured
compute-matched evaluation can be designed on the actual hardware. Success gains
must be measured; a lower generator NFE alone is not the research result.

## 6. Validation status and remaining work

- Verified: pinned Python requirements resolve/install; official PushT LeWM
  checkpoint loads strictly; image encoding produces `[B,1,192]`.
- Verified: core mathematical/data/planning tests and an exact single-process
  checkpoint-resume comparison, **9 tests passed**.
- Verified: random PushT fixture collection, image preprocessing, latent caching,
  BTM and FM optimizer steps, world-model consistency gradients, closed-loop
  evaluation, and W&B **offline** training/evaluation metrics.
- Not verified here: CUDA/NCCL or multi-process Gloo execution. This workspace has
  no NVIDIA GPU and denies Gloo socket operations. The distributed test was
  attempted and blocked with `Operation not permitted`.
- Not run: full expert-dataset training, online W&B synchronization, published
  benchmark reproduction, or V-JEPA 2.1/Meta-World experiments. No performance
  improvement is claimed from the tiny wiring fixture.

Run on the GPU server before committing a long job:

```bash
OMP_NUM_THREADS=1 python -m pytest tests -q
python scripts/preflight.py --task pusht --gpus 4
```

The socket-limited development environment can run
`python -m pytest tests -m 'not distributed' -q`; this explicitly excludes the
distributed test and does not certify multi-GPU execution.

The next research milestone is the V-JEPA 2.1/Meta-World port with patch-token
preservation and an action-conditioned predictor trained on the chosen robot
data, followed by the matched subgoal and no-subgoal comparisons. It should not
be represented as implemented by this LeWM release.
