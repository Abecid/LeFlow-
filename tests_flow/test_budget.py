import copy
import json

import pytest
import torch

from flow_jepa.budget import PlanningBudget, PlanningBudgetExceeded, training_progress
from flow_jepa.campaign import adopt_data_manifest
from flow_jepa.common import digest, file_hash, save_json
from flow_jepa.data import Segments
from flow_jepa.evaluate import Controller
from flow_jepa.models import System
from flow_jepa.train import train
from test_training import fixture_data


def test_exhausted_planning_budget_stops_before_model_work(small_config):
    world = System(small_config, "world").world.eval()
    controller = Controller(None, world, small_config, "cem_long")
    now = [0.0]
    budget = PlanningBudget(1.0, clock=lambda: now[0])
    now[0] = 1.0
    with pytest.raises(PlanningBudgetExceeded):
        controller.plan(torch.randn(1, 2, 8), torch.randn(1, 2, 8), budget=budget)
    assert world.calls == 0


def test_planning_budget_interrupts_a_rollout(small_config):
    world = System(small_config, "world").world.eval()
    controller = Controller(None, world, small_config, "cem_long")
    values = iter(range(100))
    budget = PlanningBudget(4, clock=lambda: next(values))
    with pytest.raises(PlanningBudgetExceeded):
        controller.plan(torch.randn(1, 2, 8), torch.randn(1, 2, 8), budget=budget)
    assert world.calls == small_config["evaluation"]["cem_candidates"]


def test_late_action_is_discarded_and_failure_stays_in_evaluation(tmp_path, small_config, monkeypatch):
    import importlib
    import h5py
    import numpy as np

    evaluation = importlib.import_module("flow_jepa.evaluate")
    c = copy.deepcopy(small_config)
    c["heldout_tasks"] = []
    c["data"]["test_episodes_per_task"] = 1
    initial = np.zeros((2, 2, 3), dtype=np.uint8)
    path = tmp_path / "episode.h5"
    with h5py.File(path, "w") as f:
        f["initial_rgb"] = initial
        f["goal"] = np.zeros((2, 8), dtype=np.float32)
    save_json(tmp_path / "manifest.json", dict(
        protocol=digest(c), encoder="test", mean=[0.0] * 8, std=[1.0] * 8,
        entries=[dict(id="test/reach/0", task="reach", split="test", index=0,
                      seed=17, path=path.name, sha256=file_hash(path), expert_success=True)],
    ))

    class Encoder:
        fingerprint = "test"
        def __call__(self, clip):
            return torch.zeros(1, 2, 8)

    class Environment:
        steps = 0
        def reset(self): pass
        def render(self): return initial.copy()
        def step(self, action):
            self.steps += 1
            raise AssertionError("Late action reached simulator")
        def close(self): pass

    class ExpireAfterPlanning:
        def __init__(self, seconds): self.checks = 0
        def check(self):
            self.checks += 1
            if self.checks == 2:
                raise PlanningBudgetExceeded("Late result")

    def plan(self, start, goal, **kwargs):
        return torch.zeros(4), goal[0], 0.0, 1

    env = Environment()
    monkeypatch.setattr(evaluation, "PlanningBudget", ExpireAfterPlanning)
    monkeypatch.setattr(evaluation, "make_env", lambda *args: env)
    monkeypatch.setattr(evaluation, "distributed", lambda: (0, 1, torch.device("cpu")))
    monkeypatch.setattr(Controller, "plan", plan)
    world = System(c, "world").world.eval()
    records, metrics = evaluation.evaluate(None, world, c, tmp_path, "cem_long", 3072,
                                            "test", Encoder(), torch.device("cpu"))
    assert env.steps == 0
    assert len(records) == 1 and not records[0]["success"]
    assert records[0]["controller_budget_exhausted"]
    assert metrics["episodes"] == 1 and metrics["success_macro"] == 0.0


def test_budget_stop_validates_checkpoints_and_never_resumes_extra_updates(tmp_path, small_config, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setenv("WANDB_SILENT", "true")
    c = copy.deepcopy(small_config)
    c["heldout_tasks"] = []
    c["training"].update(budget_seconds=1e-9, world_steps=100)
    fixture_data(tmp_path, c)
    out = tmp_path / "world"
    train(c, tmp_path, out, "world", 3072, allow_fixture=True)
    done = json.loads((out / "complete.json").read_text())
    assert done["step"] == 1 and done["stop_reason"] == "compute_budget"
    assert (out / "best.pt").exists()
    checkpoint = torch.load(out / "last.pt", weights_only=False)
    assert checkpoint["training_seconds"] >= c["training"]["budget_seconds"]
    train(c, tmp_path, out, "world", 3072, allow_fixture=True)
    resumed = torch.load(out / "last.pt", weights_only=False)
    assert resumed["step"] == 1
    for key in checkpoint["model"]:
        torch.testing.assert_close(checkpoint["model"][key], resumed["model"][key], rtol=0, atol=0)
    assert training_progress(1, 100, 7200, 7200) == 1


def test_cache_reuse_preserves_all_episode_identity_and_rejects_changed_camera(tmp_path, small_config):
    old = copy.deepcopy(small_config)
    old["seeds"] = [3072, 3073, 3074]
    original = dict(protocol=digest(old), fixture=False,
                    entries=[dict(id="train/reach/0", sha256="unchanged")], mean=[0], std=[1])
    path = tmp_path / "manifest.json"
    save_json(path, original)
    original_hash = file_hash(path)
    new = copy.deepcopy(small_config)
    new["training"]["budget_seconds"] = 7200
    adopt_data_manifest(tmp_path, old, new)
    updated = json.loads(path.read_text())
    assert updated["entries"] == original["entries"]
    assert updated["source_manifest_sha256"] == original_hash
    assert updated["protocol"] == digest(new)
    assert file_hash(tmp_path / "data-manifest.json") == original_hash
    adopt_data_manifest(tmp_path, old, new)  # Resume is idempotent.
    new["data"]["camera"] = "different"
    with pytest.raises(ValueError, match="changing data"):
        adopt_data_manifest(tmp_path, old, new)


def test_three_learned_heads_use_identical_training_windows(tmp_path, small_config):
    fixture_data(tmp_path, small_config)
    datasets = [Segments(tmp_path, small_config, m, 3072, 10) for m in (
        "joint_flow_consistent", "leflow_adapted", "hwm_adapted")]
    for i in range(10):
        reference = datasets[0][i]
        for data in datasets[1:]:
            sample = data[i]
            for key in ("z", "a", "coarse_z", "local"):
                assert torch.equal(reference[key], sample[key])
    model = System(small_config, "hwm_adapted")
    batch = {k: torch.stack([datasets[2][0][k], datasets[2][1][k]]) for k in datasets[2][0]}
    loss, _ = model(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
