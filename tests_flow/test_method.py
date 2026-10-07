import pytest
import torch

from flow_jepa.models import System
from flow_jepa.evaluate import Controller


@pytest.mark.parametrize(
    "method",
    [
        "joint_flow_consistent",
        "joint_flow",
        "joint_deterministic_consistent",
        "joint_deterministic",
        "leflow_adapted",
        "hwm_adapted",
        "world",
    ],
)
def test_losses_and_backward(small_config, method):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    c = small_config
    model = System(c, method)
    world = System(c, "world").world.eval().requires_grad_(False)
    # Nonconstant action-conditioned test dynamics, to exercise action derivatives.
    torch.nn.init.normal_(world.out.weight, std=0.1)
    batch = dict(
        z=torch.randn(2, 3, 2, 8),
        a=torch.randn(2, 2, 2, 4).clamp(-1, 1),
        local=torch.randn(2, 2, 2, 8),
    )
    if method in ("world", "hwm_adapted"):
        batch["a"] = batch["a"][:, 0]
    loss, parts = model(
        batch,
        world,
        consistency_weight=0.1 if method.endswith("_consistent") else 0.0,
        consistency_steps=2,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.norm() > 0 for p in model.parameters())
    assert all(p.grad is None for p in world.parameters())
    if method.endswith("_consistent"):
        assert "generated_consistency" in parts


def test_generated_bridge_has_action_and_state_gradients(small_config):
    torch.manual_seed(53)
    system = System(small_config, "joint_flow_consistent")
    world = System(small_config, "world").world.eval().requires_grad_(False)
    torch.nn.init.normal_(world.out.weight, std=0.2)
    start, goal = torch.randn(2, 2, 8), torch.randn(2, 2, 8)
    z, a = system.planner.sample(start, goal, steps=2)
    z.retain_grad()
    a.retain_grad()
    path = torch.cat([start[:, None], z, goal[:, None]], 1)
    ends = world.rollout(path[:, :-1].flatten(0, 1), a.reshape(4, 2, 4))[:, -1]
    (ends - path[:, 1:].flatten(0, 1)).square().mean().backward()
    assert z.grad.abs().sum() > 0
    assert a.grad.abs().sum() > 0
    assert system.planner.zout.weight.grad.abs().sum() > 0
    assert system.planner.aout.weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "method",
    [
        "joint_flow_consistent",
        "joint_deterministic",
        "leflow_adapted",
        "hwm_adapted",
        "cem_short",
        "cem_long",
    ],
)
def test_all_controllers_produce_legal_actions(small_config, method):
    world = System(small_config, "world").world.eval()
    model = None if method.startswith("cem_") else System(small_config, method).eval()
    controller = Controller(model, world, small_config, method)
    action, goal, cost, horizon = controller.plan(
        torch.randn(1, 2, 8), torch.randn(1, 2, 8)
    )
    assert action.shape == (4,)
    assert action.abs().max() <= 1
    assert torch.isfinite(action).all()
    assert goal.shape == (2, 8)
    assert world.calls > 0 and horizon > 0
