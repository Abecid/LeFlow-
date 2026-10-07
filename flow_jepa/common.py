from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False))
    temp.replace(path)


def config(path):
    c = json.loads(Path(path).read_text())
    if set(c["training_tasks"]) & set(c["heldout_tasks"]):
        raise ValueError("Training and task-held-out sets overlap")
    if len(set(c["seeds"])) != len(c["seeds"]):
        raise ValueError("Repeated training seeds")
    return c


def episode_seed(task, split, index):
    # Separate blocks, then task/index: no accidental cross-split seed collisions.
    base = {"train": 10000000, "validation": 1000000000, "test": 2000000000}[split]
    task_id = int(hashlib.sha256(task.encode()).hexdigest()[:6], 16) % 9000
    return base + task_id * 10000 + index


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def distributed():
    rank, size = int(os.getenv("RANK", 0)), int(os.getenv("WORLD_SIZE", 1))
    local = int(os.getenv("LOCAL_RANK", 0))
    if not 1 <= size <= 4:
        raise ValueError("This campaign permits at most four GPUs")
    requested = os.getenv("FLOW_DEVICE", "auto")
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("FLOW_DEVICE must be auto, cpu, or cuda")
    # Explicit CUDA launches must select the rank before any CUDA availability
    # query; early probing breaks NCCL on target_server_2's driver/runtime stack.
    use_cuda = requested == "cuda" or (
        requested == "auto" and torch.cuda.is_available()
    )
    device = torch.device(f"cuda:{local}" if use_cuda else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    os.environ.setdefault("MUJOCO_GL", "egl")
    egl = os.getenv("FLOW_EGL_DEVICES")
    if egl:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = egl.split(",")[local]
    if size > 1 and not dist.is_initialized():
        from datetime import timedelta

        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo", timeout=timedelta(hours=6)
        )
    return rank, size, device


def barrier():
    if dist.is_initialized():
        dist.barrier()


def gather(value):
    if not dist.is_initialized():
        return [value]
    result = [None] * dist.get_world_size()
    dist.all_gather_object(result, value)
    return result


def git_revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def checkpoint(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    torch.save(data, tmp)
    tmp.replace(path)
