"""Structural regression checks, with no optimizer updates or benchmark training."""
import json
from copy import deepcopy

import h5py
import numpy as np
import pytest
import torch

from flow_jepa.common import digest, require_execution
from flow_jepa.data import Segments
from flow_jepa.models import JointPlanner, System
from flow_jepa.scoring import continuous_plan_score
from flow_jepa.vision import state_clip


@pytest.mark.parametrize("mode", ["velocity", "endpoint"])
def test_production_dimension_noise_support(mode):
    torch.set_num_threads(2)
    torch.manual_seed(98)
    # Production dimensionality, with fewer tokens/layers to keep this unit test cheap.
    p = JointPlanner(1024, 8, 2, 5, width=256, depth=1, state_parameterization=mode)
    torch.nn.init.normal_(p.zout.weight, std=0.01)
    torch.nn.init.normal_(p.zout.bias, std=0.01)
    start, goal = torch.randn(1, 2, 1024), torch.randn(1, 2, 1024)
    noise = torch.randn(1, 1, 2, 1024)
    actions = torch.randn(1, 2, 5, 8)
    basis = torch.cat([p.zout.weight.detach(), p.zout.bias.detach()[:, None]], 1)
    q, _ = torch.linalg.qr(basis.double(), mode="reduced")
    def residual(x):
        x = x.double()
        return x - (x @ q) @ q.T
    with torch.no_grad():
        sampled, _ = p.sample(start, goal, steps=8, noise=(noise.clone(), actions))
    anchor = (start[:, None] + goal[:, None]) / 2
    target = residual(noise) if mode == "velocity" else residual(anchor)
    assert torch.allclose(residual(sampled), target, atol=2e-6, rtol=0)
    if mode == "endpoint":
        assert not torch.allclose(sampled, noise)


@pytest.mark.parametrize("steps", [1, 4, 8])
def test_endpoint_integrator_reaches_known_clean_endpoint(steps):
    from types import MethodType
    p = JointPlanner(8, 4, 2, 2, width=16, depth=1, state_parameterization="endpoint")
    target = torch.randn(1, 1, 2, 8)
    def constant(self, z, actions, start, goal, time):
        return target, torch.zeros_like(actions)
    p.forward = MethodType(constant, p)
    sampled, _ = p.sample(torch.randn(1, 2, 8), torch.randn(1, 2, 8), steps=steps)
    assert torch.equal(sampled, target)


@pytest.mark.parametrize("method", ["joint_flow_consistent", "leflow_adapted"])
def test_endpoint_objective_and_consistency_have_gradients(small_config, method):
    c = deepcopy(small_config)
    c["model"]["state_parameterization"] = "endpoint"
    system = System(c, method)
    world = System(c, "world").world.eval().requires_grad_(False)
    torch.nn.init.normal_(world.out.weight, std=0.02)
    batch = {"z": torch.randn(2, 3, 2, 8), "a": torch.randn(2, 2, 2, 4),
             "local": torch.randn(2, 2, 2, 8)}
    loss, _ = system(batch, world, consistency_weight=0.1 if method.endswith("consistent") else 0,
                     consistency_steps=2)
    loss.backward()
    assert torch.isfinite(loss)
    assert system.planner.zout.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in world.parameters())


def test_static_view_uses_one_frame_for_state_and_goal():
    frames = np.arange(5 * 2 * 2 * 3, dtype=np.uint8).reshape(5, 2, 2, 3)
    state = state_clip(frames, 3, 16, "static")
    goal = np.repeat(frames[3:4], 16, axis=0)
    assert np.array_equal(state, goal)
    assert not np.array_equal(state, state_clip(frames, 3, 16, "causal"))


@pytest.mark.parametrize("stage", ["world", "joint_flow_consistent", "leflow_adapted", "hwm_adapted"])
def test_all_models_use_same_cached_static_states(tmp_path, small_config, stage):
    c = deepcopy(small_config)
    c["model"]["state_representation"] = "static"
    # Distinguishable causal and static views; a single valid window removes sampling ambiguity.
    causal = np.full((5, 2, 8), -9, np.float32)
    static = np.arange(5 * 2 * 8, dtype=np.float32).reshape(5, 2, 8)
    with h5py.File(tmp_path / "episode.h5", "w") as f:
        f["z"], f["image_goals"] = causal, static
        f["actions"] = np.zeros((4, 4), np.float32)
    row = dict(task="reach", split="train", steps=4, mode="expert", expert_success=True,
               first_success_action=0, path="episode.h5")
    (tmp_path / "manifest.json").write_text(json.dumps(dict(protocol=digest(c),
        mean=[0] * 8, std=[1] * 8, entries=[row])))
    b = Segments(tmp_path, c, stage, 3072, 1)[0]
    assert not (b["z"] == -9).any()
    if stage != "world":
        assert torch.equal(b["z"], torch.from_numpy(static[::2]))
        assert torch.equal(b["local"], torch.from_numpy(static[:3:2]))
        assert torch.equal(b["coarse_z"], b["z"])


def test_scoring_never_restarts_from_imagined_subgoals():
    class AdditiveWorld:
        def rollout(self, start, actions, budget=None):
            self.received_start = start.clone()
            return start[:, None] + actions.cumsum(1)[:, :, None]
    world = AdditiveWorld()
    start = torch.tensor([[[1.0, 0.0]]])
    goal = torch.tensor([[[1.0, 2.0]]])
    # Candidate zero describes teleportation but takes no actions; candidate one moves correctly.
    paths = torch.tensor([[[[1., 0.]], [[1., 1.]], [[1., 2.]]]]).expand(2, -1, -1, -1)
    actions = torch.tensor([[[[0., 0.]], [[0., 0.]]], [[[0., 1.]], [[0., 1.]]]])
    costs = continuous_plan_score(world, start, goal, paths, actions)
    assert torch.equal(world.received_start, start.expand(2, -1, -1))
    assert costs[1] < costs[0]
    assert costs[1].abs() < 1e-6


@pytest.mark.parametrize("operation", ["training", "test"])
def test_repair_and_user_hold_reject_new_execution(tmp_path, operation):
    with pytest.raises(RuntimeError, match="disabled"):
        require_execution({"execution": {operation + "_enabled": False}}, tmp_path, operation)
    (tmp_path / "post-training-hold.json").write_text('{"restart_authorized": false}')
    with pytest.raises(RuntimeError, match="held"):
        require_execution({}, tmp_path, operation)
