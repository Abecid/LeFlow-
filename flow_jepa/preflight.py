"""Real CUDA/NCCL, official encoder, renderer and gradient gates before a long job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from .common import barrier, config, distributed, gather, save_json, seed_all
from .environment import collect_episode, make_env
from .models import System
from .vision import Encoder, history_clip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/flow_metaworld.json")
    p.add_argument("--root", required=True)
    a = p.parse_args()
    c = config(a.config)
    rank, size, device = distributed()
    if device.type != "cuda":
        raise RuntimeError(
            "GPU preflight requires CUDA; CPU unit tests are not a substitute"
        )
    if rank == 0:
        encoder = Encoder(c["encoder"], a.root, device)
    barrier()
    if rank != 0:
        encoder = Encoder(c["encoder"], a.root, device)
    seed_all(6000 + rank)
    dc = dict(
        c["data"],
        primitive_budget=c["model"]["chunk_steps"] * c["data"]["action_repeat"],
    )
    sample = collect_episode("reach", 7000 + rank, dc)
    env = make_env("reach", 7000 + rank, dc)
    try:
        env.reset()
        if not np.array_equal(env.render(), sample["frames"][0]):
            raise AssertionError("Repeated seeded reset is not identical")
    finally:
        env.close()
    clips = [
        history_clip(sample["frames"], i, dc["history_frames"])
        for i in range(0, len(sample["frames"]), dc["action_repeat"])
    ]
    z = torch.cat(
        [
            encoder(np.stack(clips[i : i + c["encoder"]["batch_size"]]))
            for i in range(0, len(clips), c["encoder"]["batch_size"])
        ]
    ).clone()
    actions = torch.tensor(sample["actions"], device=device).reshape(
        1, c["model"]["chunk_steps"], -1
    )
    micro = c["training"]["micro_batch"]
    batch = dict(
        z=z[None].expand(micro, -1, -1, -1).clone(),
        a=actions.expand(micro, -1, -1).clone(),
    )
    base = System(c, "world").to(device)
    module = DDP(base, device_ids=[device.index]) if size > 1 else base
    optimizer = torch.optim.AdamW(base.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = module(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    world = base.world.eval().requires_grad_(False)
    model = System(c, "joint_flow_consistent").to(device)
    joint = DDP(model, device_ids=[device.index]) if size > 1 else model
    m, k = c["model"]["segments"], c["model"]["chunk_steps"]
    zpath = (
        torch.stack([torch.lerp(z[0], z[-1], i / m) for i in range(m + 1)])[None]
        .expand(micro, -1, -1, -1)
        .clone()
    )
    # Explicit wiring fixture, never an evaluation result or training dataset.
    fixture = dict(z=zpath, a=actions[:, None].expand(micro, m, k, -1).clone())
    loss, parts = joint(
        fixture,
        world,
        consistency_weight=0.1,
        consistency_batch=c["training"]["consistency_batch"],
        consistency_steps=c["training"]["consistency_steps"],
    )
    loss.backward()
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite joint loss")
    if not any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in model.parameters()
    ):
        raise AssertionError("No finite planner gradients")
    checks = gather(
        dict(
            rank=rank,
            gpu=torch.cuda.get_device_name(device),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            encoder_shape=list(z.shape),
            joint_loss_finite=True,
            generated_consistency_finite=bool(
                torch.isfinite(parts["generated_consistency"])
            ),
            peak_memory_gib=torch.cuda.max_memory_allocated(device) / 2**30,
        )
    )
    if rank == 0:
        save_json(
            Path(a.root) / "gpu_preflight.json",
            dict(
                passed=True,
                checks=checks,
                label="Wiring validation only; not an experiment result",
            ),
        )
        print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
