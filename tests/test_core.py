import h5py
from datetime import timedelta
import numpy as np
import pytest
import torch

from btm_jepa.data import LatentSegments, atomic_json, episode_split
from btm_jepa.distributed import EvalShard, initialize
from btm_jepa.models import PathModel, PlannerTrainingModel, btm_loss, sample_paths
from btm_jepa.runtime import SubgoalRuntime


def test_btm_stopped_gradient_is_not_squared_jvp_gradient():
    class LinearMap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, x, *args):
            return self.a * x

    model = LinearMap()
    target, noise = torch.tensor([[[3.0]]]), torch.tensor([[[1.0]]])
    dummy = torch.zeros(1, 1)
    loss, _ = btm_loss(
        model,
        target,
        dummy,
        dummy,
        torch.ones(1),
        noise=noise,
        time=torch.tensor([0.5]),
        boundary_weight=0,
    )
    loss.backward()
    # x=2, direction=2, Jd=4. Stopped target gives 2*(-4)*2 = -16.
    # Differentiating ||Jd||² would instead give +16 and reverse the update.
    assert model.a.grad.item() == pytest.approx(-16)


@pytest.mark.parametrize("method", ["btm", "flow", "deterministic"])
def test_generator_shapes_boundaries_and_gradients(method):
    path = PathModel(8, method, hidden_dim=16, heads=2, depth=1)
    model = PlannerTrainingModel(path, action_dim=6, inverse_hidden=16)
    b = dict(
        z_path=torch.randn(3, 6, 8),
        spacing=torch.ones(3),
        local_z=torch.randn(3, 5, 8),
        actions=torch.randn(3, 4, 6),
    )
    out = model(b)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    z0, zg = b["z_path"][:, 0], b["z_path"][:, -1]
    samples = sample_paths(
        path, z0, zg, horizon=5, spacing=1, num_samples=4, flow_steps=2
    )
    assert samples.shape == (3, 4, 6, 8)
    torch.testing.assert_close(samples[:, :, 0], z0[:, None].expand(3, 4, 8))
    torch.testing.assert_close(samples[:, :, -1], zg[:, None].expand(3, 4, 8))
    if method != "flow":
        assert path.time is None
        with pytest.raises(ValueError, match="diffusion time"):
            path(torch.randn(3, 4, 8), z0, zg, torch.ones(3), torch.ones(3))


def test_episode_split_and_nonduplicated_validation():
    source = dict(episodes=100, episode_ids=list(range(100)), signature="abc")
    split = episode_split(source)
    groups = [set(split[k]) for k in ["train", "val", "test"]]
    assert len(set.union(*groups)) == 100
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert split == episode_split(source)
    indices = [i for rank in range(4) for i in EvalShard(7, rank, 4)]
    assert sorted(indices) == list(range(7))


@pytest.mark.parametrize("evaluation_seconds, expected", [(0, 7200), (10800, 11400)])
def test_collective_wait_covers_sequential_evaluation(
    monkeypatch, evaluation_seconds, expected
):
    calls = []
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda backend, timeout: calls.append((backend, timeout)),
    )
    initialize("cpu", evaluation_timeout_seconds=evaluation_seconds)
    assert calls == [("gloo", timedelta(seconds=expected))]


def test_cuda_initialization_selects_rank_device_before_collectives(monkeypatch):
    calls = []
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "3")

    def reject_early_cuda_probe():
        pytest.fail("Do not probe CUDA availability/count before rank device selection")

    monkeypatch.setattr(torch.cuda, "is_available", reject_early_cuda_probe)
    monkeypatch.setattr(torch.cuda, "device_count", reject_early_cuda_probe)
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda index: calls.append(("set_device", index))
    )
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda backend, timeout: calls.append(("collectives", backend)),
    )
    rank, world, device = initialize("cuda")
    assert (rank, world, device) == (3, 4, torch.device("cuda", 3))
    assert calls == [("set_device", 3), ("collectives", "nccl")]


def test_cache_action_blocks_and_physical_spacing(tmp_path):
    with h5py.File(tmp_path / "cache.h5", "w") as f:
        for ep in range(3):
            g = f.create_group(f"episodes/{ep}")
            g["z"] = np.arange(41, dtype=np.float32)[:, None]
            g["action"] = np.arange(41, dtype=np.float32)[:, None]
    atomic_json(
        tmp_path / "manifest.json",
        dict(
            split=dict(train=[0], val=[1], test=[2]),
            source=dict(lengths=[41] * 3),
            episode_shards={str(i): "cache.h5" for i in range(3)},
        ),
    )
    ds = LatentSegments(
        tmp_path / "manifest.json",
        horizon=2,
        action_block=2,
        local_horizon=2,
        spacing_blocks=[3],
        clip_stride=1,
    )
    b = ds[0]
    assert b["z_path"][:, 0].tolist() == [0, 6, 12]
    assert b["local_z"][:, 0].tolist() == [0, 2, 4]
    assert b["actions"].tolist() == [[0, 1], [2, 3]]
    assert ds.index[-1][1] + 12 < 41


def test_verification_uses_executed_actions_not_clamped_goal():
    class Dynamics(torch.nn.Module):
        def action_encoder(self, a):
            return a

        def predict(self, z, a):
            return z + a

    class Planner(torch.nn.Module):
        def inverse_actions(self, z, nxt):
            return torch.ones_like(z)

    runtime = SubgoalRuntime(Planner(), Dynamics())
    # All generated endpoints equal 10, but two unit actions only reach state 2.
    paths = torch.tensor([[[[0.0], [5.0], [10.0]], [[0.0], [7.0], [10.0]]]])
    _, goal_error, _ = runtime.decode_and_verify(
        paths, 1, torch.tensor([-10.0]), torch.tensor([10.0])
    )
    torch.testing.assert_close(goal_error, torch.tensor([[64.0, 64.0]]))


def test_cem_refinement_reaches_an_analytic_target():
    class Dynamics(torch.nn.Module):
        def action_encoder(self, a):
            return a

        def predict(self, z, a):
            return z + a

    runtime = SubgoalRuntime(torch.nn.Linear(1, 1), Dynamics())
    initial = torch.zeros(1, 2, 1)
    actions = runtime.refine(
        torch.zeros(1, 1),
        torch.ones(1, 1),
        initial,
        torch.tensor([-1.0]),
        torch.tensor([1.0]),
        candidates=64,
        iterations=4,
        elites=8,
        generator=torch.Generator().manual_seed(0),
    )
    error = (runtime.rollout(torch.zeros(1, 1), actions)[:, -1] - 1).square().item()
    assert error < 0.01
