"""Frozen native/shared adapters and CLI contracts, using small CPU fixtures."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from leflow_refinement.core import RefinementConfig
from leflow_refinement.noise_search import NoiseSearchConfig, noise_variants
from leflow_refinement.runtime import NativeRuntime, variants
from leflow_refinement.crossbench import RefinementController
from leflow_refinement.native_eval import Solver


class Flow(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(d, d)
        self.register_buffer('calls_buffer', torch.tensor(0))
    def forward(self, x, t, start, goal):
        return .1 * self.linear(x) + .01 * (goal-start)[:, None]


class Inverse(nn.Module):
    def __init__(self, d, ad):
        super().__init__()
        self.linear = nn.Linear(3*d, ad)
        self.norm = nn.BatchNorm1d(ad)
    def forward(self, a, b):
        return 4*self.norm(self.linear(torch.cat((a, b, b-a), -1)))


class World(nn.Module):
    def __init__(self, d, ad):
        super().__init__()
        self.action_encoder = nn.Linear(ad, d)
        self.last_actions = []
    def predict(self, states, actions):
        return states + .1 * actions
    def from_world(self, actions):
        return (actions * .3).clamp(-1, 1)
    def rollout(self, z, actions, *, budget=None):
        self.last_actions.append(actions.detach().clone())
        if budget is not None:
            budget.check()
        states = z + .1 * self.action_encoder(actions).cumsum(1)
        return states[:, :, None]


class Model(nn.Module):
    def __init__(self, d=3, ad=10):
        super().__init__()
        self.flow = Flow(d)
        self.inverse = Inverse(d, ad)
        self.lewm = World(d, ad)
        self.action_block = 5
    def decode_actions(self, paths):
        d = paths.shape[-1]
        return self.inverse(paths[:, :-1].reshape(-1, d), paths[:, 1:].reshape(-1, d)).reshape(len(paths), paths.shape[1]-1, -1)
    def encode_current_and_goal(self, info):
        return info['z'], info['goal']


def snapshot(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


def assert_unchanged(model, before):
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize('ad', [10, 25])
@pytest.mark.parametrize('name', list(noise_variants()))
def test_native_adapter_all_variants(ad, name):
    torch.manual_seed(42)
    model = Model(ad=ad)
    engine = NativeRuntime(model, lower=[-.7]*ad, upper=[.7]*ad)
    before = snapshot(model)
    cfg = noise_variants()[name]
    z = torch.zeros(2, 3); goal = torch.ones_like(z)*.3
    with torch.inference_mode():
        result = engine.plan(z, goal, cfg, seed=123)
    assert result['actions'].shape == (2, 5, ad)
    assert result['actions'].abs().max() <= .7
    assert result['telemetry']['world_candidate_transitions'] == 2*960
    assert result['telemetry']['flow_candidate_evaluations'] == 2*3072
    assert result['telemetry']['noise_search']['rounds_completed'] == cfg.rounds
    assert result['telemetry']['planner_seconds'] > 0
    assert_unchanged(model, before)
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    json.dumps(result['telemetry'], allow_nan=False)


@pytest.mark.parametrize('ad', [10, 25])
@pytest.mark.parametrize('projected', [False, True])
def test_shared_controller_returns_evaluated_bounded_prefix(ad, projected):
    torch.manual_seed(42)
    model = Model(ad=ad)
    world = model.lewm.eval().requires_grad_(False)
    spec = SimpleNamespace(method='leflow', horizon=5, ad=ad)
    cfg = NoiseSearchConfig(samples=8, elites=2, steps=3, project_shell=projected)
    controller = RefinementController(model, world, spec, cfg)
    before = snapshot(model)
    with torch.inference_mode():
        action = controller.plan(torch.zeros(1, 1, 3), torch.ones(1, 1, 3)*.3)
    all_actions = torch.cat(world.last_actions)
    index = controller.trace[-1]['selected_index']
    assert action.shape == (ad,) and action.abs().max() <= 1
    assert torch.equal(action, all_actions[index, 0])
    assert (all_actions.abs() <= 1).all()
    assert controller.trace[-1]['noise_search']['rounds_completed'] == 3
    assert controller.trace[-1]['world_candidate_transitions'] == 3*8*5
    controller.observe(torch.ones(1, 1, 3))
    assert_unchanged(model, before)
    controller.begin_episode()
    assert not controller.trace and controller.decisions == 0


def test_native_solver_accepts_noise_config_and_serializes_telemetry():
    engine = NativeRuntime(Model(), lower=[-1.]*10, upper=[1.]*10)
    cfg = NoiseSearchConfig(samples=8, elites=2, steps=3)
    solver = Solver(engine, cfg, 42)
    solver.configure(action_space=SimpleNamespace(shape=(1, 2)), n_envs=1,
                     config=SimpleNamespace(horizon=5, action_block=5))
    out = solver.solve({'z': torch.zeros(1, 3), 'goal': torch.ones(1, 3)})
    assert out['actions'].shape == (1, 5, 10)
    assert solver.rollout_count == 24
    assert solver.trace[0]['encoding_and_planning_seconds'] > 0
    assert solver.trace[0]['noise_search']['rounds_completed'] == 3
    json.dumps(solver.trace, allow_nan=False)


def test_old_six_variants_and_defaults_do_not_expand_existing_campaign():
    expected = ('leflow', 'more_samples', 'more_steps', 'pnp', 'world_gradient', 'world_pnp')
    assert tuple(variants()) == expected
    assert len(variants(include_noise=True)) == 12
    assert all(isinstance(v, RefinementConfig) for v in variants().values())
    assert all(isinstance(v, NoiseSearchConfig) for v in noise_variants().values())
    text = Path('leflow_refinement/native_eval.py').read_text()
    assert "names = list(variants()) if args.variant == 'all'" in text


@pytest.mark.parametrize('module', ['crossbench', 'native_eval'])
def test_existing_eval_cli_lists_explicit_noise_variants(module):
    result = subprocess.run([sys.executable, '-m', 'leflow_refinement.'+module, '--help'],
                            capture_output=True, text=True, check=True)
    for name in noise_variants():
        assert name in result.stdout


def test_registry_roundtrip_records_all_hyperparameters():
    for cfg in noise_variants().values():
        assert NoiseSearchConfig(**json.loads(json.dumps(asdict(cfg)))) == cfg


def test_shared_rejects_wrong_model_without_running_it():
    with pytest.raises(ValueError, match='matching LeFlow'):
        RefinementController(Model(), None, SimpleNamespace(method='vanilla_action_flow', horizon=5), NoiseSearchConfig())
