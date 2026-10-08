"""Bounded CPU collection and ordered reduction, independent of GPU inference."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing
import os
import time

from .environment import collect_episode, collect_goal_episode


def worker_init():
    # Children never use CUDA. Keep software rasterizer and BLAS pools bounded.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)


def collect_row(row, data_config, sparse_goals=True):
    start = time.perf_counter()
    collect = collect_goal_episode if sparse_goals and row["split"] == "test" else collect_episode
    episode = collect(row["task"], row["seed"], data_config, row["mode"])
    return row, episode, time.perf_counter() - start


def collected_rows(rows, data_config, workers, prefetch=None, sparse_goals=True):
    """Finish independent episodes out of order; row IDs define all randomness.

    Bound both submitted work and retained RGB payloads. Manifest reduction later
    still follows the original registered order, preserving floating-point sums.
    """
    if workers < 1:
        for row in rows:
            yield collect_row(row, data_config, sparse_goals)
        return
    capacity = prefetch or 2 * workers
    if capacity < workers:
        raise ValueError("Prefetch must be at least the worker count")
    rows = iter(rows)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn"),
                             initializer=worker_init) as pool:
        pending = set()
        def submit():
            row = next(rows, None)
            if row is not None:
                pending.add(pool.submit(collect_row, row, data_config, sparse_goals))
        for _ in range(capacity):
            submit()
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                submit()
                yield result


def default_workers(world_size):
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    return max(1, min(8, (cpus - 2 * world_size) // (2 * world_size)))
