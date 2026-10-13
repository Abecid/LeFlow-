# Execution-calibrated visual planning with V-JEPA 2.1

**Current status: all five additional baseline runs are trained, evaluated, audited and published.** The project retains 21 completed shared-world comparisons and 81 complete development evaluation reports. No training run is active.

| New comparison | 5k /10k /15k /20k successes | Selected /104 |
| --- | --- | ---: |
| [vanilla_action_flow](docs/VANILLA_ACTION_FLOW_RESULTS.md) | 77 / 84 / 89 / 98 | **98** |
| [action_flow_search](docs/ACTION_FLOW_SEARCH_RESULTS.md) | 74 / 79 / 84 / 78 | **84** |
| [sage_port](docs/SAGE_PORT_RESULTS.md) | 62 / 77 / 77 / 75 | **77** |
| [diffusion_policy_port](docs/DIFFUSION_POLICY_PORT_RESULTS.md) | 43 / 69 / 70 / 74 | **74** |
| [feedback_wm_port](docs/FEEDBACK_WM_PORT_RESULTS.md) | 42 / 65 / 75 / 76 | **76** |

**Vanilla conditional action flow is the strongest measured development controller at 98/104.** Our retained execution-calibrated retrieval-repair method scores 80/104. These results do not establish a gain from our TTT/reasoning/repair mechanism over the stronger BC baseline. The plain baseline has no retrieval, world search, workspace, TTT or residual repair.

Read the [completed five-method report](docs/ADDITIONAL_BASELINE_RUNS_20261012.md), [full experiment index](docs/RESULTS_INDEX.md), [results ledger](docs/results-ledger.json) and [paper blueprint](docs/PAPER_BLUEPRINT.md). They preserve negative results, full curves, implementation differences, paired failures and contribution boundaries.

## Research direction

The proposed mechanism uses observed action-prefix outcomes to correct predicted costs, condition planning, and generate bounded action-flow repairs around a retained incumbent. Its scientific target is better physical decisions per unit of controller compute. Additional inference, lower prediction MAE and retained predicted-cost incumbents do not by themselves demonstrate useful scaling or better execution. The current strongest baseline changes what this mechanism must beat.

## Fixed setup and limits

Training uses 6,222 successful expert episodes, frozen cached V-JEPA 2.1 ViT-L features/shared spatial world, seed 3072 and global batch 64. Each learned policy has a 20k-update OR eight optimizer-GPUh ceiling. Four fixed 104-case development evaluations cover 13 tasks under 200 primitive actions AND ten cumulative controller seconds. The reserved 3,200-case final benchmark remains sealed. One seed and repeatedly selected development cases do not establish native-benchmark SOTA, independent generalization or component causality. All method ports retain their explicit fidelity caveats.

## Implementation and evidence

- [Controllers/training](flow_jepa/execution/), [baseline ports](flow_jepa/baselines/), [LeFlow TTT](flow_jepa/leflow_ttt/) and [configurations](config/).
- [Exact new-method adaptations](docs/ADDITIONAL_BASELINES_20261012.md), [released baseline fidelity](docs/RELEASE_BASELINES.md) and [all measured results](docs/RESULTS_INDEX.md).
- [Chronological progress](docs/PROGRESS.md), [historical sampler repairs](docs/REPAIR_RESULTS.md) and [earlier different-world campaign](docs/REPAIRED_VALIDATION_RESULTS.md).

Full periodic metrics, physical case records, compute, measured selected-trajectory contact sheets, source identities and checkpoint hashes remain in [the report archive](docs/reports/). Raw datasets and large checkpoints remain on the server. Run `python3 scripts/analysis/verify_result_publications.py` to check the ledger and shared case identities without running models or opening final tests. Completed runs and their negative findings must be pushed to `origin/main`.

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
