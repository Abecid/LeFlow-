"""Episode-disjoint data, action normalization, and frozen-latent caches."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 -- registers source compression filters
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

TASKS = {
    "pusht": dict(repo="quentinll/lewm-pusht", archive="pusht_expert_train.h5.zst", name="pusht_expert_train"),
    "tworoom": dict(repo="quentinll/lewm-tworooms", archive="tworoom.tar.zst", name="tworoom"),
    "reacher": dict(repo="quentinll/lewm-reacher", archive="reacher.tar.zst", name="dmc/reacher_random"),
    "cube": dict(repo="quentinll/lewm-cube", archive="cube_single_expert.tar.zst", name="ogbench/cube_single_expert"),
}


def cache_root() -> Path:
    return Path(os.environ.get("STABLEWM_HOME", "~/.stable_worldmodel")).expanduser().resolve()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def image_transform(size=224):
    return v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                       v2.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]),
                       v2.Resize((size, size), antialias=True)])


def inspect_source(path):
    """Check SWM HDF5 schema and compute streaming primitive-action statistics.

    Normalization intentionally matches the frozen released LeWM's full-source
    convention (sample std, ddof=1), not new planner-only training statistics.
    Nonfinite terminal actions are excluded from stats and from sampled clips.
    """
    path = Path(path).resolve()
    with h5py.File(path, "r") as f:
        required = {"ep_len", "ep_offset", "pixels", "action", "step_idx"}
        missing = required - set(f.keys())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        lengths, offsets = f["ep_len"][:].astype(int), f["ep_offset"][:].astype(int)
        if len(lengths) < 3 or len(lengths) != len(offsets) or np.any(lengths < 1):
            raise ValueError("Need >=3 nonempty episodes with matching lengths/offsets")
        if not np.array_equal(offsets, np.r_[0, np.cumsum(lengths)[:-1]]):
            raise ValueError("Expected contiguous, ordered episode offsets")
        n = int(lengths.sum())
        if len(f["pixels"]) != n or len(f["action"]) != n:
            raise ValueError("Pixels/actions and episode metadata disagree")
        if f["pixels"].ndim != 4 or f["pixels"].shape[-1] != 3 or f["pixels"].dtype != np.uint8:
            raise ValueError("Expected uint8 RGB pixels [frames, height, width, 3]")
        if f["action"].ndim != 2:
            raise ValueError("Expected primitive actions [frames, action_dim]")
        total = np.zeros(f["action"].shape[-1], np.float64)
        square, count, invalid = total.copy(), 0, 0
        for start in range(0, n, 65536):
            a = f["action"][start:start+65536].astype(np.float64)
            good = np.isfinite(a).all(-1)
            invalid += int((~good).sum())
            a = a[good]
            total += a.sum(0)
            square += np.square(a).sum(0)
            count += len(a)
        if count < 2:
            raise ValueError("Need at least two finite action rows")
        mean = total / count
        std = np.sqrt(np.maximum((square - count * mean**2) / (count - 1), 0)).clip(1e-6)
        ep_key = "episode_idx" if "episode_idx" in f else "ep_idx"
        if ep_key not in f:
            raise ValueError("Expected episode_idx or ep_idx column")
        ids = [int(np.asarray(f[ep_key][int(off)]).item()) for off in offsets]
        if len(set(ids)) != len(ids):
            raise ValueError("Episode IDs must be unique")
        signature = hashlib.sha256(lengths.tobytes()+offsets.tobytes()+mean.tobytes()+std.tobytes()).hexdigest()
        return dict(path=str(path), bytes=path.stat().st_size, frames=n, episodes=len(lengths),
                    lengths=lengths.tolist(), offsets=offsets.tolist(), episode_ids=ids,
                    signature=signature, action_dim=len(mean), action_mean=mean.tolist(),
                    action_std=std.tolist(), action_stats_scope="frozen_lewm_full_source_ddof1",
                    nonfinite_action_rows=invalid, pixel_shape=list(f["pixels"].shape[1:]))


def episode_split(source, seed=3072, train_fraction=.8, val_fraction=.1):
    n = source["episodes"]
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1 or train_fraction + val_fraction >= 1:
        raise ValueError("Positive train/val fractions must sum to <1")
    order = np.random.default_rng(seed).permutation(n)
    train_n = min(max(1, int(n*train_fraction)), n-2)
    val_n = min(max(1, int(n*val_fraction)), n-train_n-1)
    groups = dict(train=order[:train_n], val=order[train_n:train_n+val_n], test=order[train_n+val_n:])
    return dict(seed=seed, source_signature=source["signature"],
                **{k: sorted(v.tolist()) for k,v in groups.items()},
                **{f"{k}_episode_ids": sorted(source["episode_ids"][int(i)] for i in v) for k,v in groups.items()})


class LatentSegments(Dataset):
    """Read per-episode latents without loading images during planner training.

    z_path: [H+1,D] every spacing*action_block primitive steps.
    local_z: [K+1,D] every action_block steps; actions: [K,block*A].
    The same short-step inverse model serves all coarse waypoint spacings.
    """
    def __init__(self, manifest, split="train", horizon=5, action_block=5,
                 spacing_blocks=(1,2,4), local_horizon=4, clip_stride=5):
        self.manifest_path = Path(manifest).resolve()
        self.meta = json.loads(self.manifest_path.read_text())
        self.split = split
        self.horizon, self.block, self.local_horizon = horizon, action_block, local_horizon
        self.spacings = tuple(sorted(set(int(s) for s in spacing_blocks)))
        if horizon < 2 or min(self.spacings) < 1 or min(action_block, local_horizon, clip_stride) < 1:
            raise ValueError("Invalid segment configuration")
        self._handles = {}
        self._pid = None
        self.index = []
        min_span = max(horizon*min(self.spacings), local_horizon) * action_block
        for ep in self.meta["split"][split]:
            length = self.meta["source"]["lengths"][ep]
            self.index.extend((ep, s) for s in range(0, length-min_span, clip_stride))
        if not self.index:
            raise ValueError(f"No {split} clips for requested horizon/spacing; use longer episodes")

    def __len__(self):
        return len(self.index)

    def __getstate__(self):
        return {**self.__dict__, "_handles": {}, "_pid": None}

    def __getitem__(self, index):
        if self._pid != os.getpid():
            self._handles, self._pid = {}, os.getpid()
        ep, start = self.index[index]
        shard = self.meta["episode_shards"][str(ep)]
        if shard not in self._handles:
            self._handles[shard] = h5py.File(self.manifest_path.parent / shard, "r")
        g = self._handles[shard][f"episodes/{ep}"]
        length = len(g["z"])
        eligible = [s for s in self.spacings if start + self.horizon*s*self.block < length]
        spacing = eligible[index % len(eligible)]
        ids = start + np.arange(self.horizon+1)*spacing*self.block
        local_ids = start + np.arange(self.local_horizon+1)*self.block
        a = g["action"][start:start+self.local_horizon*self.block]
        if not np.isfinite(a).all():
            raise ValueError(f"Nonfinite nonterminal action in episode {ep}, start {start}")
        return {"z_path": torch.from_numpy(g["z"][ids].astype(np.float32)),
                "local_z": torch.from_numpy(g["z"][local_ids].astype(np.float32)),
                "actions": torch.from_numpy(a.astype(np.float32).reshape(self.local_horizon, -1)),
                "spacing": torch.tensor(float(spacing)), "episode": ep, "start": start}
