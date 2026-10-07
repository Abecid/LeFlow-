import copy
import json

import h5py
import numpy as np
import pytest
import torch

from flow_jepa.common import digest, file_hash, save_json
from flow_jepa.data import Segments
from flow_jepa.train import train


def fixture_data(root, c):
    entries = []
    rng = np.random.default_rng(567)
    for split in ("train", "validation"):
        for index in range(3):
            path = root / f"{split}_{index}.h5"
            actions = rng.uniform(-1, 1, (8, 4)).astype(np.float32)
            states = [rng.normal(size=(2, 8)).astype(np.float32)]
            for action in actions:
                states.append(states[-1] + 0.1 * np.tile(action, 2)[None])
            states = np.stack(states)
            with h5py.File(path, "w") as f:
                f["z"] = states
                f["image_goals"] = states
                f["actions"] = actions
            entries.append(
                dict(
                    id=f"{split}/reach/{index}",
                    task="reach",
                    split=split,
                    mode="expert",
                    expert_success=True,
                    path=path.name,
                    steps=8,
                    sha256=file_hash(path),
                )
            )
    save_json(
        root / "manifest.json",
        dict(
            protocol=digest(c),
            entries=entries,
            mean=[0.0] * 8,
            std=[1.0] * 8,
            fixture=True,
            encoder="synthetic-test-fixture",
        ),
    )


def test_data_sampling_and_fixture_rejection(tmp_path, small_config):
    c = copy.deepcopy(small_config)
    fixture_data(tmp_path, c)
    data = Segments(tmp_path, c, "joint_flow", 1, 10)
    item = data[0]
    assert item["z"].shape == (3, 2, 8)
    assert item["a"].shape == (2, 2, 4)
    assert torch.equal(item["z"], data[0]["z"])
    with pytest.raises((RuntimeError, ValueError)):
        train(c, tmp_path, tmp_path / "forbidden", "world", 1)


def test_train_resume_and_joint_update(tmp_path, small_config, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setenv("WANDB_SILENT", "true")
    c = copy.deepcopy(small_config)
    c["heldout_tasks"] = []
    fixture_data(tmp_path, c)
    train(c, tmp_path, tmp_path / "full", "world", 1, allow_fixture=True)
    full = torch.load(tmp_path / "full/last.pt", weights_only=False)
    assert full["step"] == 2 and full["fixture"]
    import importlib

    module = importlib.import_module("flow_jepa.train")
    original = module.checkpoint

    def interrupt_after_checkpoint(path, data):
        original(path, data)
        if data["step"] == 1:
            raise InterruptedError(
                "intentional interruption after a durable checkpoint"
            )

    monkeypatch.setattr(module, "checkpoint", interrupt_after_checkpoint)
    with pytest.raises(InterruptedError):
        train(c, tmp_path, tmp_path / "resumed", "world", 1, allow_fixture=True)
    monkeypatch.setattr(module, "checkpoint", original)
    train(c, tmp_path, tmp_path / "resumed", "world", 1, allow_fixture=True)
    resumed = torch.load(tmp_path / "resumed/last.pt", weights_only=False)
    for k in full["model"]:
        torch.testing.assert_close(
            full["model"][k], resumed["model"][k], rtol=0, atol=0
        )
    world = tmp_path / "full/best.pt"
    before = file_hash(world)
    train(
        c,
        tmp_path,
        tmp_path / "joint",
        "joint_flow_consistent",
        1,
        world,
        allow_fixture=True,
    )
    assert file_hash(world) == before
    joint = torch.load(tmp_path / "joint/last.pt", weights_only=False)
    assert joint["world_hash"] == before and joint["step"] == 2
    logs = [
        json.loads(x)
        for x in (tmp_path / "joint/metrics.jsonl").read_text().splitlines()
    ]
    assert all(np.isfinite(x["train/generated_consistency"]) for x in logs)
