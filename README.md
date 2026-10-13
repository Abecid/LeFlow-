# New baseline result: vanilla conditional action flow98/104

The first two newly authorized comparisons are complete: vanilla conditional
flow77/84/89/98 selects20k98/104, and Diffusion Policy43/69/70/74 selects20k74/104.
Both have zero controller timeouts at all four fixed104 evaluations. Their
source/data/bank/checkpoint audits pass. These are single-seed development
results; the new98 baseline exceeds the retained proposed repair controller80.
Fixed-world action search, Feedback-WM and SAGE remain in the authorized batch.

[Vanilla full findings](docs/VANILLA_ACTION_FLOW_RESULTS.md) ·
[Diffusion Policy full findings](docs/DIFFUSION_POLICY_PORT_RESULTS.md) ·
[Live five-method batch](docs/ADDITIONAL_BASELINE_RUNS_20261012.md).

Historical project overview follows; its80 best-reference statements are superseded.

# Execution-calibrated visual planning with V-JEPA 2.1 (LeFlow fork)

**New baseline implementations (not yet trained/evaluated):** vanilla action-flow BC,
fixed-world action-flow search, SAGE-inspired subgoal planning, adapted Diffusion
Policy and Feedback-WM. See [run protocol, code, and preflight instructions](docs/ADDITIONAL_BASELINES_20261012.md).
Their results are **not** part of the completed sixteen-method ledger below.


**Current status, October 12, 2026:** all sixteen registered shared-world
comparisons are complete. The latest LaCT-inspired nonlinear episodic memory
run scores **73 / 66 / 67 / 72 out of 104**, selecting 73 at 5k updates. It does
not improve on the retained **80/104 (76.92%)** retrieval-repair or
execution-revision references. No training run is active.

Start with the [complete experiment index](docs/RESULTS_INDEX.md), which links
all methods, training curves, baseline comparisons, implementation sources,
negative results, failure analyses and earlier campaigns. The
[machine-readable ledger](docs/results-ledger.json) preserves 16 current
comparisons and 61 complete development evaluations with artifact hashes.

The [paper blueprint](docs/PAPER_BLUEPRINT.md) states the central thesis, exact
formulation, contribution/evidence boundaries, related-work positioning, paper
outline and figure designs, and the missing experiments for stronger claims.

| Reference | Best development success | Scope |
| --- | ---: | --- |
| Retrieval-conditioned bounded flow repair | **80/104** | Action flow, bounded residual repairs, linear episodic correction |
| Execution-revision controller | **80/104** | GMM proposer; no flow/diffusion generator |
| Latest nonlinear episodic flow repair | **73/104** | LaCT-inspired memory trained through factual prefix updates |
| LeFlow release-module/spatial port | 21/104 | Released core modules with documented shared-world adaptation |
| HWM paper port | 5/104 | Robotics implementation unavailable; time-limited paper port |
| Released CEM | 24/104 | Released solver on the shared frozen world |

## Research direction and observed behavior

The research direction is **execution-calibrated bounded flow repair**: use
observed action-prefix outcomes to correct predicted costs, condition action
generation and bounded repairs, and retain the incumbent during search.
Whether the correction mechanism improves physical decisions and provides useful
inference scaling remains unestablished. This is a V-JEPA 2.1 / MetaWorld study;
the current method uses no BTM and is not an unchanged published LeFlow model.

- [Retrieval-repair](docs/RETRIEVAL_REPAIR_TTT_RESULTS.md) reaches 80/104 but
  does not exceed the preserved GMM reference. Their selected checkpoints have
  different case-level successes and failures.
- [Nonlinear memory](docs/NONLINEAR_REPAIR_TTT_RESULTS.md) improves aggregate
  next-prefix prediction error while task success remains below 80. Assembly
  and pick-place remain 0/8 throughout; reach falls from 8/8 to 4/8. The learned
  empty-history prior worsens prediction error relative to the raw world.
- [Goal-progress supervision](docs/PROGRESS_OBJECTIVE_TTT_RESULTS.md),
  [Bayesian repair acceptance](docs/BAYESIAN_REPAIR_TTT_RESULTS.md), and
  [shifted-plan training](docs/SHIFTED_REPAIR_TTT_RESULTS.md) select 78, 77,
  and 79 respectively. These negative results are retained.
- Lower action losses, smaller repair targets and shrinking proposal spread do
  not establish better execution. Route provenance and sparse trajectory images
  support diagnostics, not causal claims about rejected actions or contact forces.
- [Released baseline findings](docs/RELEASE_BASELINE_RESULTS.md) distinguish
  implementation provenance, continuing loss improvement, and severe controller
  time limits for HWM/CEM from unrestricted method capability.

## Data, budget and claim limits

Policy training uses **6,222 successful expert episodes**, frozen cached
V-JEPA 2.1 features and a shared frozen spatial world, seed 3072 and global batch
64. Each trained policy stops at 20,000 updates or eight optimizer GPU-hours;
consumed compute differs. Four fixed **104-case development evaluations** cover
13 tasks under 200 primitive actions and ten cumulative controller seconds.
[The protocol audit](docs/RELEASE_BASELINES.md) documents baseline adaptations.

These are repeatedly used development cases with one training seed. Best-checkpoint
selection is not independent testing; convergence of all baselines is not
established. The reserved 3,200-case final evaluation remains unused by the current
comparison. No native-benchmark SOTA, general superiority, component causality or
inference-scaling result is claimed.

## Implementations and reproducibility

- [Current controllers and training](flow_jepa/execution/),
  [baseline ports and pinned upstream provenance](flow_jepa/baselines/),
  [direct LeFlow TTT](flow_jepa/leflow_ttt/), and [experiment configurations](config/).
- [Latest complete results, behavior and visual evidence](docs/NONLINEAR_REPAIR_TTT_RESULTS.md).
- [All experiments, source mapping and failure findings](docs/RESULTS_INDEX.md).
- [Chronological decisions, repairs and runtime history](docs/PROGRESS.md).
- [Historical sampler repairs](docs/REPAIR_RESULTS.md) and
  [different-world experiments](docs/REPAIRED_VALIDATION_RESULTS.md).

Full training metrics, evaluation records, compute accounting, selected trajectory
contact sheets, source identities and checkpoint paths/hashes are preserved in
[the evidence archive](docs/reports/). Model binaries and raw datasets remain on
the server; they are not stored in Git. Run
`python3 scripts/analysis/verify_result_publications.py` to verify the current
ledger and paired development-case identities without running models or opening
final-test outcomes. Every completed run must be published to `origin/main`,
including unsuccessful experiments.

Original LeFlow release documentation and attribution follow.

---

# LeFlow: Latent Path Flow Planning for LeWorldModel

LeFlow adds a learned goal-conditioned planner on top of a frozen
[LeWorldModel](https://arxiv.org/pdf/2603.19312v1) backbone. Instead of running
test-time CEM over actions, LeFlow first generates a latent future path and then
uses inverse dynamics to convert latent transitions into action chunks.

This repository builds on the original LeWorldModel codebase, plus
`stable-worldmodel` for datasets, environments, policies, and evaluation.

## Checkpoints

Main H=5 LeFlow checkpoints are hosted on [`Hugging Face`](https://huggingface.co/hsiangwei0903/LeFlow)

The repo contains:

| Benchmark | Checkpoint |
|---|---|
| TwoRoom | `tworoom/latent_planner.pt` |
| PushT | `pusht/latent_planner.pt` |
| Reacher | `reacher/latent_planner.pt` |
| OGBench Cube | `cube/latent_planner.pt` |

The LeFlow checkpoint stores only lightweight planner state:

- flow planner state dict
- inverse dynamics state dict
- architecture metadata
- reference to the frozen LeWM checkpoint


## Setup

Create an environment following the original LeWorldModel setup:

```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

Set the stable-worldmodel cache root:

```bash
export STABLEWM_HOME=/path/to/stable-wm
```

You also need the original LeWM datasets and frozen LeWM checkpoints from the
official LeWM collection:

[`quentinll/lewm`](https://huggingface.co/collections/quentinll/lewm)

Expected frozen LeWM checkpoint references:

| Benchmark | LeWM reference |
|---|---|
| TwoRoom | `tworoom/lewm` |
| PushT | `pusht/lewm` |
| Reacher | `reacher/lewm` |
| OGBench Cube | `cube/lewm` |

These should resolve under `$STABLEWM_HOME`, exactly as in the original LeWM
evaluation code.

## Evaluation

Use `solver=latent_flow` and pass the downloaded LeFlow checkpoint as `policy`.

TwoRoom:

```bash
python eval.py --config-name=tworoom.yaml \
  solver=latent_flow \
  policy=leflow/tworoom/latent_planner.pt \
  plan_config.horizon=5 \
  plan_config.receding_horizon=5 \
  plan_config.action_block=5 \
  eval.num_eval=50
```

PushT:

```bash
python eval.py --config-name=pusht.yaml \
  solver=latent_flow \
  policy=leflow/pusht/latent_planner.pt \
  plan_config.horizon=5 \
  plan_config.receding_horizon=5 \
  plan_config.action_block=5 \
  eval.num_eval=50
```

Reacher:

```bash
python eval.py --config-name=reacher.yaml \
  solver=latent_flow \
  policy=leflow/reacher/latent_planner.pt \
  plan_config.horizon=5 \
  plan_config.receding_horizon=5 \
  plan_config.action_block=5 \
  eval.num_eval=50
```

OGBench Cube:

```bash
python eval.py --config-name=cube.yaml \
  solver=latent_flow \
  policy=leflow/cube/latent_planner.pt \
  plan_config.horizon=5 \
  plan_config.receding_horizon=5 \
  plan_config.action_block=5 \
  eval.num_eval=50
```

Default latent-flow inference settings are in
[`config/eval/solver/latent_flow.yaml`](config/eval/solver/latent_flow.yaml):

```yaml
num_samples: 64
flow_steps: 16
score_mode: rollout_goal
```

The reranking score uses frozen-LeWM autoregressive rollout final latent
distance to the encoded goal. It does not score candidates by the clamped
generated endpoint.

## Training

LeFlow training freezes LeWM and trains two modules jointly:

- `LatentPathFlow`: rectified-flow model over latent path interiors
- `InverseDynamics`: maps `[z_t, z_{t+1}, z_{t+1} - z_t]` to normalized action chunks

Main loss weights:

| Loss | Weight |
|---|---:|
| flow matching | 1.0 |
| inverse dynamics MSE | 1.0 |
| LeWM one-step consistency | 0.1 |
| latent smoothness | 0.0 |

All main models use:

```text
horizon = 5
action_block = 5
epochs = 10
batch_size = 128
```

TwoRoom:

```bash
python train_latent_planner.py \
  lewm_checkpoint=tworoom/lewm \
  data.dataset.name=tworoom \
  data.dataset.keys_to_load='[pixels,action,proprio]' \
  data.dataset.keys_to_cache='[action,proprio]' \
  planner.horizon=5 \
  planner.max_horizon=20 \
  planner.action_block=5 \
  epochs=10 \
  loader.batch_size=128 \
  subdir=latent_planner/tworoom_h5_ep10
```

PushT:

```bash
python train_latent_planner.py \
  lewm_checkpoint=pusht/lewm \
  data.dataset.name=pusht_expert_train \
  data.dataset.keys_to_load='[pixels,action,proprio,state]' \
  data.dataset.keys_to_cache='[action,proprio,state]' \
  planner.horizon=5 \
  planner.max_horizon=20 \
  planner.action_block=5 \
  epochs=10 \
  loader.batch_size=128 \
  subdir=latent_planner/pusht_h5_ep10
```

Reacher:

```bash
python train_latent_planner.py \
  lewm_checkpoint=reacher/lewm \
  data.dataset.name=reacher \
  data.dataset.keys_to_load='[pixels,action,observation]' \
  data.dataset.keys_to_cache='[action,observation]' \
  planner.horizon=5 \
  planner.max_horizon=20 \
  planner.action_block=5 \
  epochs=10 \
  loader.batch_size=128 \
  subdir=latent_planner/reacher_h5_ep10
```

OGBench Cube:

```bash
python train_latent_planner.py \
  lewm_checkpoint=cube/lewm \
  data.dataset.name=ogbench/cube_single_expert \
  data.dataset.keys_to_load='[pixels,action,observation]' \
  data.dataset.keys_to_cache='[action,observation]' \
  planner.horizon=5 \
  planner.max_horizon=20 \
  planner.action_block=5 \
  epochs=10 \
  loader.batch_size=128 \
  subdir=latent_planner/cube_h5_ep10
```

Training saves:

```text
$STABLEWM_HOME/<subdir>/latent_planner.pt
```

## SLURM Helpers

Example one-GPU training:

```bash
sbatch --export=ALL,TASK=pusht,HORIZON=5,EPOCHS=10,RUN_NAME=pusht_h5_ep10 \
  train_latent_planner.slurm
```

Example one-GPU evaluation:

```bash
sbatch --export=ALL,CONFIG_NAME=pusht.yaml,SOLVER=latent_flow,\
PLANNER_CHECKPOINT=leflow/pusht/latent_planner.pt,HORIZON=5,NUM_EVAL=50 \
  eval_latent_planner.slurm
```

## Key Files

| File | Purpose |
|---|---|
| [`latent_planner.py`](latent_planner.py) | LeFlow modules and solver wrapper |
| [`train_latent_planner.py`](train_latent_planner.py) | Training entry point |
| [`config/train/latent_planner.yaml`](config/train/latent_planner.yaml) | Training config |
| [`config/eval/solver/latent_flow.yaml`](config/eval/solver/latent_flow.yaml) | Evaluation solver config |
| [`eval.py`](eval.py) | LeWM-compatible evaluation entry point |

## Citation

This code builds on LeWorldModel:

```bibtex
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}

@misc{huang2026leflowgenerativelatentflow,
      title={LeFlow: Generative Latent Flow Planning for World Models}, 
      author={Hsiang-Wei Huang and Jianxu Shangguan and Junbin Lu and Jenq-Neng Hwang},
      year={2026},
      eprint={2608.24855},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2608.24855}, 
}
```
