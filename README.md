# Execution-aware visual planning with V-JEPA 2.1 (LeFlow fork)

**Current status, October 9 Pacific / October 10 UTC:** one fresh
**candidate-conditioned action-flow reasoner is training on all eight A800 GPUs**.
It updates its workspace after inspecting predicted candidate outcomes, with
fixed causal scoring and paired flow supervision. All 21 behavioral checks
passed; full-controller preflight averaged 98.39 ms. Same seed, data, global
batch and training/controller ceilings. The first 5k validation scores **79/104**
versus the preserved GMM best of 80/104; three registered checkpoints remain.
[Registered method and run](docs/FLOW_REASONING_RUN.md),
[verified launch evidence](docs/reports/20261010-flow-reasoning/), and
[live metrics](https://wandb.ai/attentionx2023/flow-jepa-metaworld/runs/6orbayjr).
No baseline retraining, extra seed or sealed final test is running.

The corrected baselines are
completed, verified and frozen. Selected development scores are **LeFlow21/104,
HWM paper port5/104 and released CEM24/104**, versus our preserved80/104.
[Full results, learning curves and compute](docs/RELEASE_BASELINE_RESULTS.md) and
[the source/adaptation audit](docs/RELEASE_BASELINES.md) record the exact scope.
These are shared-backbone MetaWorld ports under fixed training/controller limits.
HWM robot code is unavailable, convergence is not established, and CEM/HWM are
strongly limited by controller time. This is not an authors' native-benchmark
reproduction or an independent final-test SOTA claim. All20 tests passed and
all baseline workers finished. These frozen references are reused below.

The execution-revision iteration is complete.
Its four fixed validation rounds score **80, 78, 79 and 76 out of 104** at 5k,
10k, 15k and 20k updates. The selected 5k checkpoint scores **76.92%**, versus
the previous controller-grounded **79/104 (75.96%)** and latent-revision
**77/104 (74.04%)**. The one-case best-checkpoint gain is not reliable superiority.
Later training improves demonstration fit while initial proposal diversity falls
about 51%; the causes of the task-success regression remain unisolated.

One fresh seed-3072 policy was trained on all eight A800 GPUs using the same
6,222 successful expert episodes, global batch 64, 20k-update/8-optimization-GPUh
ceilings, fixed 104 validation cases and 10-second/200-action controller limits.
Actual optimization used 3.233375 GPUh; registered validation used 1.945972 GPUh.
The cached V-JEPA 2.1 encoder features and fine world were frozen. Baselines
were preserved; no baseline rerun, extra seed, ablation or final test was added.
The run finished normally and all owned workers exited.

The previous best measured candidate uses an execution-error-conditioned latent workspace,
a GMM action proposer, supported target retrieval and bounded CEM verification.
It scores the executed prefix and terminal window, reuses shifted plans and
penalizes repeated observed stalls. It has **no flow-matching or diffusion
generator**. The `flow_jepa` package and W&B project names are historical; this
is neither the published Flow-JEPA dynamics model nor an unchanged LeFlow model.
Historical LeFlow/HWM adaptations retain documented implementation differences;
these development validation comparisons are not corrected-SOTA or final-test
claims. The corrected baseline work is complete; the new flow-reasoning run is
separate from this preserved GMM reference.

- [Corrected baseline results and resource-limit findings](docs/RELEASE_BASELINE_RESULTS.md)
- [Independent project, comparison and contribution audit](docs/PROJECT_AUDIT_20261010.md)
- [Latest results, training-decline diagnosis and method lineage](docs/EXECUTION_REVISION_RESULTS.md)
- [Registered implementation and resource contract](docs/EXECUTION_REVISION_RUN.md)
- [Previous latent-revision results](docs/LATENT_REVISION_RESULTS.md)
- [Previous strongest controller-grounded result](docs/CONTROLLER_GROUNDED_RESULTS.md)
- [Latent-reasoning literature and code review](docs/LARC_APPLICATION_20261009.md)
- [Progress and preserved experiment evidence](docs/PROGRESS.md)
- [Historical repair and comparison protocol](docs/REPAIRED_COMPARISON.md)
- [Implementation and literature-fidelity audit](docs/EVAL_AUDIT_20261009.md)

The research uses frozen V-JEPA 2.1 features for MetaWorld, with **no BTM**.
Historical BTM scripts remain only for reproducibility. Earlier scheduling
records are historical. The old all-method queue stays cancelled; the new baseline
launchers have separate frozen source/configuration/checkpoint registries. Final
tests stay reserved.

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
