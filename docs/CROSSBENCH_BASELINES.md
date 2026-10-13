# MetaWorld + PushT + Cube: eight-controller comparison

## Status and scope

The cross-benchmark implementation now includes data preparation, native environment
adapters, eight executable controllers, checkpointed/DDP training, periodic real
simulation evaluation, restart journals, paired comparisons, and incremental GitHub
publication. **No full PushT/Cube dataset training has been launched from this chat.**
The available development runtime has no CUDA/SSH access to the user's existing GPU
server. GitHub read/write and GitHub Actions work. CPU/CI tests are not task-success
results. Read the CI run for the exact source revision before launching.

The authorized scope is **16 new jobs**: eight methods on PushT and Cube. Reuse the
completed MetaWorld results from `docs/results-ledger.json`; do not retrain them.
This request does not authorize opening the previously sealed MetaWorld final test,
adding another method, new seed, dynamics dataset, or inference-scaling sweep.

| Row name | Controller | Important fidelity limit |
|---|---|---|
| `leflow` | Released latent-path flow/inverse modules; 64 candidates, 16 Euler steps, frozen-world endpoint ranking | Shared execution prefix of five primitives, not unchanged author deployment |
| `vanilla_action_flow` | Goal-conditioned action flow, eight integration steps, one sample | Standard behavior cloning, not a reasoning contribution |
| `execution_revision` | Four-component GMM, observed-history context, retrieval, prefix/terminal corrections, three CEM rounds | Compact-feature adaptation of the previous GMM controller |
| `action_flow_search` | Local-goal flow, retrieved actions, three fixed sample/rank pools | No workspace revision, adaptive fit, or residual flow |
| `sage_port` | Duration-conditioned predicted subgoals, eight-component GMM and CEM | Independent paper-inspired port, not an author reproduction |
| `diffusion_policy_port` | Conditional temporal U-Net, 100-step DDPM training, 16-step DDIM inference | Same previously specified image-goal adaptation |
| `feedback_wm_port` | Same diffusion network plus factual residual feedback and action-aware gradient guidance | Independent paper-inspired port, not an author reproduction |
| `retrieval_repair_ttt` | Initial action flow, episode-local ridge fit, candidate attention revision, bounded residual flow, training retrieval | Compact-feature adaptation of the retained best proposed method, not a claim of beating vanilla98 |

The five newly completed MetaWorld baselines are included exactly once; “naive flow
matching” is `vanilla_action_flow`, not another ninth method. The old experimental
BTM planner is not revived. Its existing download/cache utilities are reused.

## Frozen protocol

Each new dataset has its own released frozen LeWM encoder and predictor. All eight
controllers on that dataset share its checkpoint and cache. This is intentionally
not the 32-token V-JEPA MetaWorld backbone. Released backbone training may overlap
the source dataset: splits hold out data from new policy heads, not necessarily
from backbone pretraining. Report this exposure rather than calling it fully
unseen end-to-end generalization.

| Setting | PushT | Cube |
|---|---:|---:|
| Native primitive action dimension | 2 | **5** |
| Primitives per world/action block | 5 | 5 |
| Action plan `[blocks,block_coordinates]` | `[5,10]` | `[5,25]` |
| Proposed primitive horizon | 25 | 25 |
| Executed prefix before observing again | 5 | 5 |
| Goal offset in source trajectory | 25 | 25 |
| Episode action allowance | 50 | 50 |
| Episode controller allowance | 10 seconds | 10 seconds |
| Fixed validation cases per checkpoint | 50 | 50 |

Cube's five controls include rotation; the earlier four-dimensional assumption
was wrong. The adapter reads bounds from the native simulator and checks the
released model's action-encoder width. PushT's pinned SWM interface uses normalized
relative actions, not absolute pixel coordinates. Policy outputs, simulator
commands, and pretrained-world normalized actions have explicit tested conversions.
Every world-scored action is bounded exactly as the action physically executed.

Training: seed3072, global batch64, maximum20,000 optimizer steps OR28,800 aggregate
optimizer GPU-seconds, whichever stops first. Validation at quarter-budget
milestones (normally5k/10k/15k/20k), earliest tie retained. Each dataset/method trains
from scratch except the frozen backbone. Preparation, preflight, and validation
costs are separate from optimization. Equal caps do not imply equal consumed time,
number of world calls, parameter count, or convergence.

This initial campaign uses native goal-offset/action-budget settings with a shared
short execution prefix and the explicit controller clock. It is a **registered
comparison port**, not native published SOTA. It does not yet add Cube hard-start
perturbations or change the training distribution. Those require a separately
registered follow-up after the initial results.

## Data and exact success

`crossbench.prepare` consumes the existing `prepared/TASK/source.json`, downloads
nothing when it is already prepared, and can reuse compatible cached LeWM latents.
It writes `cache.h5` with float32 latent states and bounded policy-coordinate
primitive actions. Only TRAIN and validation episode data are read into the new
cache. The source metadata contains test IDs for exclusion; test images/actions are
not used by the new policy, route bank, or evaluator. The original released-world
action-statistic convention is retained rather than silently renormalizing it.

All methods draw identical stateless TRAIN windows. Actual five-block trajectories
supervise actions. Only factual actions receive observed outcome labels. Repair
training excludes the query demonstration from retrieved action evidence.

Evaluation restores the source's native state and goal using the existing task
configuration's setter methods. The first observation uses the recorded initial
image, matching SWM dataset evaluation; later observations come from actual physics.
Both recorded and rendered reset hashes are saved. Source variation metadata is
restored where present. No privileged state is supplied to the policy.

- PushT: use the pinned simulator's native goal-state success/termination predicate.
  This is **not** interchangeable with another PushT implementation's IoU metric.
- Cube: require `info['success']`; do not infer success from predicted image distance.
- Stop immediately when native success occurs, even inside the five-action prefix.
- No action computed after the shared controller deadline is executed.

Before optimization, the launcher checks actual TRAIN resets/rendering, executes a
bounded TRAIN action block, compares a freshly encoded image against its cached
features, and checks finite synchronized gradients without optimizer updates.
Final-test cases are inaccessible through this campaign's data/evaluation APIs.

## Run on the existing GPU server

Use a clean clone and an environment with the repository's pinned `requirements.txt`.
Do not silently upgrade the simulator or modify a checkout used by an active run.
`stable-worldmodel` is checked against commit
`06b1f4c24ccb237a9b2d6bb65bf010a6a5b41e83`. Online W&B authentication and GitHub push
credentials must already work in that server environment. Do not paste secrets into
code, launch arguments, logs, or this document.

From the repository root, set `STABLEWM_HOME` to the **existing** data/checkpoint
volume. The following selects a new isolated experiment directory:

```bash
git pull --ff-only origin main
export STABLEWM_HOME=/data/leflow  # replace only when the existing cache is elsewhere
export LEFLOW_PYTHON="$(command -v python)"  # active pinned GPU environment
export WANDB_ENTITY=attentionx2023
WORKSPACE="$HOME/adam/LeFlow-experiments/crossbench-baselines-v1"

# Dry run: writes 16-job plan, launches no models/GPU work.
"$LEFLOW_PYTHON" -m crossbench.campaign \
  --workspace "$WORKSPACE" --data-root "$STABLEWM_HOME"

# Actual launch: sequential jobs, at most four idle GPUs, automatic evaluation.
mkdir -p "$WORKSPACE"
nohup bash scripts/operations/crossbench_run.sh "$WORKSPACE" "$STABLEWM_HOME" \
  > "$WORKSPACE/coordinator.log" 2>&1 &
echo $! > "$WORKSPACE/coordinator.pid"
```

The coordinator acquires only idle GPUs via existing ownership locks. It freezes
source in `$WORKSPACE/runtime`, prepares each dataset once, and launches exactly one
method at a time. To use a Slurm server, run inside a matching GPU allocation; the
existing allocator refuses an unallocated Slurm launch.

Every completed method's metrics, four case-level evaluations, paired wins/losses,
and ten fixed trajectory contact sequences per milestone are committed and pushed
from `$WORKSPACE/publication`. Full trajectories, datasets, and model binaries stay
on the server. Push failure stops the campaign with saved evidence; no force push
is used. A publication receipt verifies remote HEAD. Incomplete measurements are
preserved on failure and never presented as complete scores.

### Inspect or resume

```bash
cat "$WORKSPACE/status.json"
cat "$WORKSPACE/analysis/matrix.csv"
tail -f "$WORKSPACE/logs/pusht-vanilla_action_flow.log"

# Resume from the ORIGINAL runtime, not a later edited main checkout.
cd "$WORKSPACE/runtime"
bash scripts/operations/crossbench_run.sh "$WORKSPACE" "$STABLEWM_HOME"
```

The resume path checks source, dataset/world identity, rank count, optimizer/RNG
state, and evaluation journal identities. Completed jobs are audited and reused.
Lost optimizer work after a crash is still charged from the durable usage ledger.
Never run two coordinators for the same workspace.

For one explicitly selected method instead of the entire matrix:

```bash
python -m crossbench.campaign --workspace "$WORKSPACE" \
  --data-root "$STABLEWM_HOME" --tasks pusht --methods vanilla_action_flow \
  --gpus 4 --execute --publish
```

### Standalone evaluation of a saved checkpoint

Run from its original frozen source; this reuses the identical registered validation
cases and provides no final-test access:

```bash
python -m crossbench.cli_evaluate \
  --cache "$WORKSPACE/cache/pusht" \
  --checkpoint "$WORKSPACE/runs/pusht/vanilla_action_flow/best.pt" \
  --output "$WORKSPACE/recheck/pusht-vanilla"
```

Repeated rechecks are additional evaluations, not new independent test estimates.
The normal campaign already performs all registered milestone evaluations.

## Verification and reporting boundary

CPU synthetic tests execute every loss/backward path and every controller in both
native action dimensions. They cover bounds, world-call counts, causal history,
TRAIN-only retrieval, identical case plans, success within a prefix, deadline
exhaustion, checkpoint selection, and interruption/resume parameter equality.
A two-process Gloo integration executes all eight training paths with synchronized
gradients. These do not substitute for the CUDA/NCCL preflight on actual data.

The separate native CI job executes real PushT/Pymunk and Cube/MuJoCo reset,
rendering, action, and self-goal success checks. Another small test loads released
LeWM checkpoints and checks encoder/dynamics dimensions and action gradients. Neither
runs a benchmark policy or produces a success-rate claim.

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_crossbench*.py' -v
OMP_NUM_THREADS=1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tests/crossbench_ddp_check.py
python -m crossbench.report --workspace "$WORKSPACE"
```

The matrix contains the eight preserved MetaWorld results and `not_started` or
`incomplete` native rows until factual case reports exist. It never fills missing
scores with estimates. Native results include successes, failure reasons, action
counts, controller timeouts, per-decision timing, world-call counts, paired case IDs,
and sampled actual frames. This supports subsequent failure analysis; lower training
loss or a successful unit test is not evidence of better robot control.
