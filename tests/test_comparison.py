"""Scientific comparison guards: shared training, paired tasks, and data identity."""

from argparse import Namespace
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_campaign_uses_identical_training_except_method_and_run_labels(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path))
    manifest = tmp_path / "latents/pusht/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "split": {"train": [0], "val": [1], "test": [2]},
                "checkpoint_sha256": "encoder",
            }
        )
    )
    args = Namespace(name="pilot", seeds=[3072, 3073], gpus=4, overrides=["epochs=2"])
    script = load_script("run_comparison")
    plan = script.make_plan(args)
    assert [job["method"] for job in plan["jobs"]] == ["flow", "btm", "flow", "btm"]
    for left, right in zip(plan["jobs"][::2], plan["jobs"][1::2]):
        a, b = deepcopy(left), deepcopy(right)
        for cfg in (a, b):
            assert cfg["wandb"]["mode"] == "online"
            assert cfg["epochs"] == 2
            for key in ("method", "run_dir", "hydra"):
                cfg.pop(key)
            cfg["wandb"].pop("name")
        assert a == b
    args.overrides = ["method=btm"]
    with pytest.raises(ValueError, match="Campaign controls method"):
        script.make_plan(args)
    args.overrides = []
    data = json.loads(manifest.read_text())
    data["test_fixture"] = True
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="test fixture"):
        script.make_plan(args)


def reports():
    script = load_script("compare_evaluations")
    protocol = dict.fromkeys(script.MATCHED, "same")
    protocol.update(method="flow", episode_indices=[0, 1, 2, 3])
    flow = dict(
        protocol=protocol,
        hardware={"device": "synthetic-test"},
        metrics={"success_rate": 0.5, "planning_batch_latency_ms_mean": 20},
        episode_successes=[1, 0, 1, 0],
    )
    btm = deepcopy(flow)
    btm["protocol"]["method"] = "btm"
    btm["episode_successes"] = [1, 1, 1, 0]
    btm["metrics"] = {"success_rate": 0.75, "planning_batch_latency_ms_mean": 10}
    return script, flow, btm


def test_paired_task_success_and_controller_speedup():
    script, flow, btm = reports()
    result = script.compare(flow, btm)
    assert result["btm_minus_flow_success_pp"] == 25
    assert result["btm_only_successes"] == 1
    assert result["flow_only_successes"] == 0
    assert result["full_controller_latency_speedup_flow_over_btm"] == 2


@pytest.mark.parametrize(
    "key",
    [
        "episode_indices",
        "start_steps",
        "manifest_sha256",
        "training_protocol",
        "cem_candidates",
    ],
)
def test_comparison_rejects_unmatched_tasks_data_or_training(key):
    script, flow, btm = reports()
    btm["protocol"][key] = "different"
    with pytest.raises(ValueError, match=f"Unmatched evaluation field: {key}"):
        script.compare(flow, btm)
