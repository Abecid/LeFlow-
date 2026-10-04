"""Real optimizer, checkpoint/resume, and two-process Gloo integration tests."""

import os
from pathlib import Path
import subprocess
import sys
import socket

import torch
import pytest

from make_fixture import create

ROOT = Path(__file__).resolve().parents[1]


def train(cache, output, steps, extra=(), processes=1):
    args = [sys.executable]
    if processes > 1:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        args += [
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--master-addr=127.0.0.1",
            f"--master-port={port}",
            f"--nproc_per_node={processes}",
        ]
    args += [
        str(ROOT / "train_subgoals.py"),
        f"run_dir={output}",
        f"data.manifest={cache}",
        "device=cpu",
        "wandb.enabled=false",
        "evaluation.enabled=false",
        "loss.consistency_weight=0",
        "num_workers=0",
        "global_batch_size=8",
        "micro_batch_size=2",
        "model.hidden_dim=16",
        "model.heads=2",
        "model.depth=1",
        "model.inverse_hidden=16",
        "data.horizon=2",
        "data.action_block=2",
        "data.local_horizon=2",
        "data.spacing_blocks=[1,2]",
        "val_max_samples=4",
        "val_generation_samples=1",
        "val_candidates=2",
        "evaluation.flow_steps=2",
        "validate_every=2",
        "checkpoint_every=2",
        "log_every=1",
        f"max_steps={steps}",
        *extra,
    ]
    env = {
        **os.environ,
        "STABLEWM_HOME": str(Path(cache).parent),
        "OMP_NUM_THREADS": "1",
        "GLOO_SOCKET_IFNAME": "lo",
    }
    p = subprocess.run(
        args, cwd=ROOT, env=env, text=True, capture_output=True, timeout=120
    )
    assert p.returncode == 0, p.stdout + "\n" + p.stderr
    return torch.load(Path(output) / "last.pt", map_location="cpu", weights_only=False)


def test_exact_single_process_resume(tmp_path):
    cache = create(tmp_path / "cache")
    full = train(cache, tmp_path / "full", 4)
    train(cache, tmp_path / "resumed", 2)
    resumed = train(
        cache, tmp_path / "resumed", 4, [f"resume={tmp_path / 'resumed' / 'last.pt'}"]
    )
    assert full["step"] == resumed["step"] == 4
    for key in full["model"]:
        torch.testing.assert_close(
            full["model"][key], resumed["model"][key], rtol=0, atol=0
        )


@pytest.mark.distributed
def test_two_process_ddp_training_and_resume(tmp_path):
    cache = create(tmp_path / "cache")
    first = train(cache, tmp_path / "ddp", 2, processes=2)
    final = train(
        cache,
        tmp_path / "ddp",
        3,
        [f"resume={tmp_path / 'ddp' / 'last.pt'}"],
        processes=2,
    )
    assert first["world_size"] == final["world_size"] == 2
    assert final["step"] == 3
    assert len(final["rng_states"]) == 2
    assert all(torch.isfinite(x).all() for x in final["model"].values())
