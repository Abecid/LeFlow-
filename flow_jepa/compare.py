"""Paired, task-balanced comparisons; no publication scores or fixtures mixed in."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import config, digest, save_json


def compare(c, reports):
    methods, seeds = c["methods"], c["seeds"]
    tasks = c["training_tasks"] + c["heldout_tasks"]
    expected = {(m, s) for m in methods for s in seeds}
    groups = {(r["method"], r["seed"]): r for r in reports}
    if len(groups) != len(reports) or set(groups) != expected:
        raise ValueError("Missing/duplicated methods or training seeds")
    if any(
        r["split"] != "test" or r.get("fixture", True) or r["protocol"] != digest(c)
        for r in reports
    ):
        raise ValueError(
            "Only complete, non-fixture, registered test reports may be compared"
        )
    for key in ("manifest", "code"):
        if len({r[key] for r in reports}) != 1:
            raise ValueError(f"Mismatched {key}")
    count = c["data"]["test_episodes_per_task"]
    arrays = {}
    reference_signature = None
    for seed in seeds:
        rows = [groups[(m, seed)] for m in methods]
        if len({r["world_hash"] for r in rows}) != 1:
            raise ValueError(
                "Comparators must use the same frozen world within a training seed"
            )
        signatures = []
        for r in rows:
            records = sorted(r["records"], key=lambda x: x["id"])
            sig = [(x["id"], x["reset_seed"], x["episode_sha256"]) for x in records]
            if len({x[0] for x in sig}) != len(sig):
                raise ValueError("Duplicate evaluation resets")
            signatures.append(sig)
        if any(s != signatures[0] for s in signatures[1:]):
            raise ValueError("Evaluation resets/goals differ between methods")
        if reference_signature is not None and signatures[0] != reference_signature:
            raise ValueError("Evaluation resets/goals differ between training seeds")
        reference_signature = signatures[0]
    for method in methods:
        out = np.empty((len(seeds), len(tasks), count), np.float64)
        for si, seed in enumerate(seeds):
            records = groups[(method, seed)]["records"]
            if len(records) != len(tasks) * count:
                raise ValueError("Incomplete test set")
            for ti, task in enumerate(tasks):
                row = sorted(
                    (r for r in records if r["task"] == task), key=lambda r: r["id"]
                )
                if len(row) != count or any(r["model_seed"] != seed for r in row):
                    raise ValueError("Incomplete task or mismatched model seed")
                out[si, ti] = [r["success"] for r in row]
        arrays[method] = out
    # The same reset ordering must also be retained between model seeds.
    reference = groups[(methods[0], seeds[0])]["records"]

    def key(rs):
        return sorted((r["id"], r["reset_seed"], r["episode_sha256"]) for r in rs)

    if any(key(r["records"]) != key(reference) for r in reports):
        raise ValueError("Model seeds evaluated different reset sets")
    rng = np.random.default_rng(40107)
    n_boot = c["evaluation"]["bootstrap_samples"]
    primary = arrays[c["primary_method"]]
    baselines = [m for m in methods if m != c["primary_method"]]
    differences = {m: primary - arrays[m] for m in baselines}
    boots = {m: np.empty(n_boot) for m in baselines}
    for i in range(n_boot):
        # Crossed resampling: a model seed is shared across tasks; each reset is
        # shared across seeds/methods. Tasks remain fixed benchmark strata.
        ss = rng.integers(len(seeds), size=len(seeds))
        ii = rng.integers(count, size=(len(tasks), count))
        for m, delta in differences.items():
            sample = delta[ss]
            sample = np.take_along_axis(sample, ii[None], axis=2)
            boots[m][i] = sample.mean()
    comparisons = {}
    for m, delta in differences.items():
        ci = np.quantile(boots[m], [0.025, 0.975]) * 100
        alpha = 0.05 / len(baselines)
        simultaneous = np.quantile(boots[m], [alpha / 2, 1 - alpha / 2]) * 100
        comparisons[m] = dict(
            delta_percentage_points=float(delta.mean() * 100),
            paired_ci95=ci.tolist(),
            familywise_ci95_bonferroni=simultaneous.tolist(),
            per_training_seed_delta_pp=(delta.mean((1, 2)) * 100).tolist(),
            wins=int((delta > 0).sum()),
            losses=int((delta < 0).sum()),
            supported_positive_difference=bool(simultaneous[0] > 0),
        )
    table = {}
    for m, x in arrays.items():
        per_seed = x.mean((1, 2))
        table[m] = dict(
            success=float(x.mean()),
            seed_std=float(per_seed.std(ddof=1)) if len(seeds) > 1 else None,
            per_seed=per_seed.tolist(),
            per_task={t: float(x[:, i].mean()) for i, t in enumerate(tasks)},
            seen_success=float(x[:, : len(c["training_tasks"])].mean()),
            heldout_success=float(x[:, len(c["training_tasks"]) :].mean())
            if c["heldout_tasks"]
            else None,
        )
    return dict(
        protocol=digest(c),
        primary_method=c["primary_method"],
        primary_baseline=c["primary_baseline"],
        table=table,
        comparisons=comparisons,
        independent_resets=len(tasks) * count,
        model_seeds=len(seeds),
        interpretation="Bootstrap uncertainty over model seeds and paired resets; tasks fixed. Three seeds provide limited training-variance precision. Ports do not establish a published-SOTA reproduction.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/flow_metaworld.json")
    p.add_argument("--reports", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    c = config(a.config)
    reports = [
        json.loads((Path(a.reports) / f"{m}_{s}.json").read_text())
        for m in c["methods"]
        for s in c["seeds"]
    ]
    result = compare(c, reports)
    save_json(a.output, result)
    print(json.dumps(result, indent=2))
