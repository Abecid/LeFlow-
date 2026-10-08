"""Bounded hardware benchmark and exact-output gate; never changes the dataset."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def equivalence(job):
    import numpy as np
    from flow_jepa.environment import collect_episode, collect_goal_episode
    task, seed, data, mode = job
    start = time.perf_counter()
    old = collect_episode(task, seed, data, mode)
    old_seconds = time.perf_counter() - start
    start = time.perf_counter()
    new = collect_goal_episode(task, seed, data, mode)
    new_seconds = time.perf_counter() - start
    checks = {k: bool(np.array_equal(old[k], new[k])) for k in
              ("actions", "rewards", "successes", "goal_index", "expert_success")}
    checks.update(initial_rgb=bool(np.array_equal(old["frames"][0], new["initial_rgb"])),
                  goal_rgb=bool(np.array_equal(old["frames"][old["goal_index"]], new["goal_rgb"])))
    return dict(task=task, seed=seed, mode=mode, exact=all(checks.values()), checks=checks,
                old_seconds=old_seconds, new_seconds=new_seconds,
                goal_index=int(old["goal_index"]),
                rgb_sha256=hashlib.sha256(new["goal_rgb"].tobytes()).hexdigest())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=["equivalence", "workers", "batches"], required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    from flow_jepa.common import config, episode_seed, save_json
    from flow_jepa.collection import collected_rows, worker_init
    c = config("config/flow_metaworld.json")
    tasks = c["training_tasks"] + c["heldout_tasks"]
    result = dict(mode=a.mode, checks=[])
    if a.mode == "equivalence":
        jobs = [(task, episode_seed(task, "test", 0), c["data"], "expert") for task in tasks]
        jobs += [(task, episode_seed(task, "train", 0), c["data"], "random")
                 for task in ("reach", "push", "pick-place", "door-close")]
        with ProcessPoolExecutor(max_workers=a.workers, initializer=worker_init,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            for row in pool.map(equivalence, jobs):
                result["checks"].append(row)
                save_json(a.output, result)
                print(json.dumps(row), flush=True)
        result["passed"] = all(x["exact"] for x in result["checks"])
        save_json(a.output, result)
        if not result["passed"]:
            raise AssertionError("Sparse collection changed retained data")
    elif a.mode == "workers":
        # Measure warm throughput as well as startup; baseline is full rendering.
        for sparse in (True,):
            rows = [dict(id=str(i), task=tasks[i % len(tasks)], split="test", mode="expert",
                         seed=episode_seed(tasks[i % len(tasks)], "test", i // len(tasks)))
                    for i in range(max(64, 4 * a.workers))]
            began, stamps = time.perf_counter(), []
            for row, episode, seconds in collected_rows(rows, c["data"], a.workers,
                                                        sparse_goals=sparse):
                stamps.append(time.perf_counter())
            warm_start = min(a.workers, len(stamps) // 2)
            result.update(workers=a.workers, episodes=len(stamps),
                          elapsed_seconds=stamps[-1] - began,
                          warm_episodes_per_second=(len(stamps) - warm_start) /
                          (stamps[-1] - stamps[warm_start - 1]))
            save_json(a.output, result)
            print(json.dumps(result), flush=True)
    else:
        import h5py
        import numpy as np
        import torch
        from flow_jepa.vision import Encoder
        torch.cuda.set_device(0)
        encoder = Encoder(c["encoder"], a.root, torch.device("cuda:0"))
        images = []
        for task in tasks[:13]:
            path = Path(a.root) / "episodes/train" / task / "00124.h5"
            with h5py.File(path, "r") as f:
                images.extend([f["initial_rgb"][:], f["goal_rgb"][:]])
        clips = np.repeat(np.stack(images)[:, None], c["data"]["history_frames"], axis=1)
        reference = torch.cat([encoder(clips[i:i + 1]).cpu() for i in range(len(clips))]).numpy().astype(np.float16)
        clips = np.concatenate([clips] * 5)[:128]
        reference = np.concatenate([reference] * 5)[:128]
        for batch in (1, 2, 3, 4, 5, 8, 16, 32, 64, 128):
            encoder(clips[:batch]); torch.cuda.synchronize()
            start = time.perf_counter()
            arrays = []
            for i in range(0, len(clips), batch):
                arrays.append(encoder(clips[i:i + batch]).cpu().numpy().astype(np.float16))
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            actual = np.concatenate(arrays)
            row = dict(batch=batch, seconds=seconds, clips_per_second=len(clips) / seconds,
                       exact_fp16=bool(np.array_equal(reference, actual)),
                       different_values=int(np.count_nonzero(reference != actual)),
                       max_abs_diff=float(np.max(np.abs(reference.astype(np.float32) - actual.astype(np.float32)))),
                       peak_memory_GiB=torch.cuda.max_memory_allocated() / 2**30)
            result["checks"].append(row)
            save_json(a.output, result)
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
