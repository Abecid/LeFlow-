from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .common import (
    barrier,
    config,
    digest,
    distributed,
    episode_seed,
    file_hash,
    save_json,
)
from .collection import collected_rows, default_workers
from .vision import Encoder, history_clip


def episode_plan(c):
    rows = []
    for split in ("train", "validation", "test"):
        tasks = c["training_tasks"] + (c["heldout_tasks"] if split == "test" else [])
        count = c["data"][split + "_episodes_per_task"]
        if split != "train":
            count *= c["data"].get("goal_candidate_multiplier", 2)
        for task in tasks:
            for i in range(count):
                mode = (
                    "random"
                    if split == "train"
                    and i < int(count * c["data"]["random_fraction"])
                    else "expert"
                )
                rows.append(
                    dict(
                        id=f"{split}/{task}/{i:05d}",
                        task=task,
                        split=split,
                        seed=episode_seed(task, split, i),
                        mode=mode,
                        index=i,
                    )
                )
    return rows


def select_valid_goals(entries, c):
    """Freeze goal-constructible evaluation cases before any method is trained.

    A failed expert's final frame is not a valid task-goal annotation. Screening
    uses only the fixed collection expert, never a tested method's outcomes.
    """
    selected, screening = [], {}
    for row in entries:
        if row["split"] == "train":
            selected.append(row)
            continue
        key = row["split"] + "/" + row["task"]
        group = screening.setdefault(
            key, dict(attempted=0, valid=0, selected=0, failed_ids=[])
        )
        group["attempted"] += 1
        if not row["expert_success"]:
            group["failed_ids"].append(row["id"])
            continue
        group["valid"] += 1
        count = c["data"][row["split"] + "_episodes_per_task"]
        if group["selected"] < count:
            selected.append(
                {**row, "source_index": row["index"], "index": group["selected"]}
            )
            group["selected"] += 1
    for key, group in screening.items():
        split = key.split("/")[0]
        if group["selected"] != c["data"][split + "_episodes_per_task"]:
            raise ValueError(
                f"Insufficient valid goal images for {key}: {group}. Fix the collection protocol; do not shrink the test set."
            )
    return selected, screening


def _encode_list(encoder, clips, batch):
    result = []
    for start in range(0, len(clips), batch):
        result.append(
            encoder(np.stack(clips[start : start + batch]))
            .cpu()
            .numpy()
            .astype(np.float16)
        )
    return np.concatenate(result)


def encode_episode(episode, row, c, encoder):
    """Use the original inference shapes, order, precision and cache arrays."""
    history, repeat = c["data"]["history_frames"], c["data"]["action_repeat"]
    frames = episode.get("frames")
    goal_image = episode["goal_rgb"] if frames is None else frames[episode["goal_index"]]
    arrays = dict(initial_rgb=episode["initial_rgb"] if frames is None else frames[0],
                  goal_rgb=goal_image,
                  goal=encoder(np.repeat(goal_image[None, None], history, axis=1)).cpu().numpy()[0].astype(np.float16),
                  success=episode["successes"])
    if row["split"] != "test":
        indexes = range(0, len(frames), repeat)
        # Materialize only one batch instead of two full lists of duplicated video clips.
        batch = int(os.getenv("FLOW_CACHE_ENCODER_BATCH", str(c["encoder"]["batch_size"])))
        def encode(kind):
            result = []
            for start in range(0, len(indexes), batch):
                ids = indexes[start:start + batch]
                clips = [history_clip(frames, i, history) if kind == "history" else
                         np.repeat(frames[i][None], history, axis=0) for i in ids]
                result.append(encoder(np.stack(clips)).cpu().numpy().astype(np.float16))
            return np.concatenate(result)
        arrays["z"] = encode("history")
        arrays["image_goals"] = encode("static")
        n = len(indexes) - 1
        arrays["actions"] = episode["actions"][:n * repeat].reshape(n, repeat * 4)
    return arrays


def preparation_batches(rows, goal_batch):
    goals = []
    for item in rows:
        if item[0]["split"] == "test":
            goals.append(item)
            if len(goals) >= goal_batch:
                yield goals
                goals = []
        else:
            if goals:
                yield goals
                goals = []
            yield [item]
    if goals:
        yield goals


def encode_goal_batch(items, c, encoder):
    images = [e["goal_rgb"] for _, e, _ in items]
    clips = np.repeat(np.stack(images)[:, None], c["data"]["history_frames"], axis=1)
    goals = encoder(clips).cpu().numpy().astype(np.float16)
    return [dict(initial_rgb=e["initial_rgb"], goal_rgb=e["goal_rgb"],
                 goal=goals[i], success=e["successes"])
            for i, (_, e, _) in enumerate(items)]


def write_episode(root, row, episode, arrays, c, fingerprint,
                  collection_seconds=0.0, encoding_seconds=0.0):
    start = time.perf_counter()
    path = root / "episodes" / (row["id"] + ".h5")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    with h5py.File(tmp, "w") as f:
        f.attrs.update(**row, protocol=digest(c), encoder=fingerprint,
                       expert_success=episode["expert_success"], goal_index=episode["goal_index"])
        for name, value in arrays.items():
            compression = "gzip" if name in ("initial_rgb", "goal_rgb") else (
                "lzf" if name in ("z", "image_goals") else None)
            f.create_dataset(name, data=value, compression=compression)
        f.flush()
    tmp.replace(path)
    return dict(event="cached_episode", episode=row["id"], success=episode["expert_success"],
                collection_seconds=collection_seconds, encoding_seconds=encoding_seconds,
                write_seconds=time.perf_counter() - start)


def summarize_episode(root, row, c):
    path = root / "episodes" / (row["id"] + ".h5")
    sha = file_hash(path)
    with h5py.File(path, "r") as f:
        successes = f["success"][:]
        item = {**row, "path": str(path.relative_to(root)), "sha256": sha,
                "expert_success": bool(f.attrs["expert_success"]),
                "steps": len(f["actions"]) if "actions" in f else 0,
                "first_success_action": int(np.flatnonzero(successes)[0]) if np.any(successes) else None}
        if row["split"] == "train":
            z = f["z"][:].astype(np.float64).reshape(-1, c["encoder"]["dim"])
            if not np.isfinite(z).all():
                raise ValueError("Nonfinite training features")
            return item, z.sum(0), np.square(z).sum(0), len(z)
    return item, None, None, 0


def prepare(c, root, *, encoder_factory=Encoder):
    rank, size, device = distributed()
    root = Path(root)
    plan = episode_plan(c)
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        lock = root / "data_protocol.json"
        if lock.exists() and json.loads(lock.read_text()) != c:
            raise ValueError(
                "Data directory belongs to another protocol; use a new directory"
            )
        save_json(lock, c)
        encoder = encoder_factory(c["encoder"], root, device)
    barrier()
    if rank != 0:
        encoder = encoder_factory(c["encoder"], root, device)
    workers = int(os.getenv("FLOW_PREP_WORKERS_PER_GPU", str(default_workers(size))))
    prefetch = int(os.getenv("FLOW_PREP_PREFETCH", str(max(1, workers * 2))))
    writer_depth = 2
    goal_batch = int(os.getenv("FLOW_GOAL_ENCODER_BATCH", "16"))
    if workers < 0 or prefetch < max(1, workers) or goal_batch < 1:
        raise ValueError("Invalid preparation concurrency")
    pending = []
    for pos in range(rank, len(plan), size):
        row = plan[pos]
        path = root / "episodes" / (row["id"] + ".h5")
        if path.exists():
            with h5py.File(path, "r") as f:
                if (f.attrs.get("protocol") != digest(c)
                        or f.attrs.get("encoder") != encoder.fingerprint):
                    raise ValueError(f"Mismatched completed cache: {path}")
        else:
            pending.append(row)
    print(json.dumps(dict(event="preparation_pipeline", rank=rank,
                          workers=workers, prefetch=prefetch, writer_depth=writer_depth,
                          pending=len(pending), encoder_batch=int(os.getenv("FLOW_CACHE_ENCODER_BATCH", str(c["encoder"]["batch_size"]))),
                          goal_batch=goal_batch,
                          sparse_test_goals=True)), flush=True)
    began, completed = time.perf_counter(), 0
    writes = deque()
    def finish(future):
        nonlocal completed
        result = future.result()  # Writer exceptions must stop the campaign.
        completed += 1
        result.update(rank=rank, completed=completed,
                      elapsed_seconds=time.perf_counter() - began)
        print(json.dumps(result), flush=True)
    with ThreadPoolExecutor(max_workers=1) as writer:
        rows = collected_rows(pending, c["data"], workers, prefetch)
        for items in preparation_batches(rows, goal_batch):
            started = time.perf_counter()
            if items[0][0]["split"] == "test":
                encoded = encode_goal_batch(items, c, encoder)
            else:
                encoded = [encode_episode(items[0][1], items[0][0], c, encoder)]
            encoding_seconds = (time.perf_counter() - started) / len(items)
            for (row, episode, collection_seconds), arrays in zip(items, encoded):
                if len(writes) >= writer_depth:
                    finish(writes.popleft())
                writes.append(writer.submit(write_episode, root, row, episode, arrays,
                                            c, encoder.fingerprint, collection_seconds,
                                            encoding_seconds))
        while writes:
            finish(writes.popleft())
    barrier()
    if rank == 0:
        entries, total, squared, count = [], None, None, 0
        expert = {}
        readers = int(os.getenv("FLOW_MANIFEST_WORKERS", "8"))
        with ThreadPoolExecutor(max_workers=max(1, readers)) as pool:
            queue = deque()
            rows = iter(plan)
            for _ in range(max(1, 2 * readers)):
                row = next(rows, None)
                if row is not None:
                    queue.append(pool.submit(summarize_episode, root, row, c))
            while queue:
                item, row_total, row_squared, row_count = queue.popleft().result()
                row = next(rows, None)
                if row is not None:
                    queue.append(pool.submit(summarize_episode, root, row, c))
                entries.append(item)
                if item["mode"] == "expert":
                    expert.setdefault(item["split"] + "/" + item["task"], []).append(
                        item["expert_success"])
                if row_count:
                    total = row_total if total is None else total + row_total
                    squared = row_squared if squared is None else squared + row_squared
                    count += row_count
        mean = total / count
        std = np.sqrt(np.maximum(squared / count - mean**2, 1e-6))
        entries, screening = select_valid_goals(entries, c)
        save_json(
            root / "manifest.json",
            dict(
                protocol=digest(c),
                encoder=encoder.fingerprint,
                mean=mean.tolist(),
                std=std.tolist(),
                entries=entries,
                goal_screening=screening,
                expert_success={k: float(np.mean(v)) for k, v in expert.items()},
                statistics_scope="training_episodes_only",
                fixture=False,
            ),
        )
    barrier()


class Segments(Dataset):
    """Stateless sampling makes the data sequence reproducible across resume."""

    def __init__(self, root, c, stage, seed, length, split="train"):
        self.root, self.c, self.stage, self.seed, self.length = (
            Path(root),
            c,
            stage,
            seed,
            length,
        )
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["protocol"] != digest(c):
            raise ValueError("Configuration differs from prepared data")
        self.mean = np.asarray(self.manifest["mean"], np.float32)
        self.std = np.asarray(self.manifest["std"], np.float32)
        representation = c["model"].get("state_representation", "causal")
        if representation not in ("causal", "static"):
            raise ValueError("Unknown state representation")
        self.state_key = "image_goals" if representation == "static" else "z"
        k, m = c["model"]["chunk_steps"], c["model"]["segments"]
        self.horizon = k if stage == "world" else k * m
        self.tasks = {}
        for row in self.manifest["entries"]:
            if row["split"] != split or row["steps"] < self.horizon:
                continue
            if stage != "world" and (
                row["mode"] != "expert" or not row["expert_success"]
            ):
                continue
            self.tasks.setdefault(row["task"], []).append(row)
        if set(self.tasks) != set(c["training_tasks"]):
            raise ValueError(
                "Some training tasks have no eligible data; inspect expert success and horizon"
            )
        self.names = sorted(self.tasks)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        rows = self.tasks[self.names[int(rng.integers(len(self.names)))]]
        row = rows[int(rng.integers(len(rows)))]
        max_start = row["steps"] - self.horizon
        if (
            self.stage != "world"
            and row.get("first_success_action") is not None
        ):
            max_start = min(
                max_start,
                row["first_success_action"] // self.c["data"]["action_repeat"],
            )
        start = int(rng.integers(max_start + 1))
        k, m = self.c["model"]["chunk_steps"], self.c["model"]["segments"]
        with h5py.File(self.root / row["path"], "r") as f:
            if self.stage == "world":
                z = (
                    f[self.state_key][start : start + k + 1].astype(np.float32) - self.mean
                ) / self.std
                a = f["actions"][start : start + k].astype(np.float32)
                return {"z": torch.from_numpy(z), "a": torch.from_numpy(a)}
            z = (
                f[self.state_key][start : start + self.horizon + 1 : k].astype(np.float32)
                - self.mean
            ) / self.std
            goal = (
                f["image_goals"][start + self.horizon].astype(np.float32) - self.mean
            ) / self.std
            coarse_z = z.copy()
            z[-1] = goal
            a = (
                f["actions"][start : start + self.horizon]
                .astype(np.float32)
                .reshape(m, k, -1)
            )
            # Observed one-chunk targets for the separate LeFlow inverse control.
            local = (
                f[self.state_key][start : start + k + 1 : k].astype(np.float32) - self.mean
            ) / self.std
            return {
                "z": torch.from_numpy(z),
                "a": torch.from_numpy(a),
                "local": torch.from_numpy(local),
                "coarse_z": torch.from_numpy(coarse_z),
            }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/flow_metaworld.json")
    p.add_argument("--root", required=True)
    args = p.parse_args()
    prepare(config(args.config), args.root)
