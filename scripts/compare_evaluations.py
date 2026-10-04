#!/usr/bin/env python
"""Audit paired evaluation protocols and report BTM minus flow task success."""

import argparse
import json
from pathlib import Path

import numpy as np

MATCHED = (
    "task",
    "split",
    "seed",
    "goal_offset",
    "budget",
    "mode",
    "spacing",
    "candidates",
    "flow_steps",
    "execute_blocks",
    "cem_candidates",
    "cem_iterations",
    "cem_elites",
    "episode_indices",
    "start_steps",
    "manifest_sha256",
    "encoder_sha256",
    "training_seed",
    "training_protocol",
    "training_world_size",
)


def compare(flow, btm):
    if flow.get("test_fixture") or btm.get("test_fixture"):
        raise ValueError("Fixture evaluation is not a research comparison")
    for key in MATCHED:
        if key not in flow["protocol"] or key not in btm["protocol"]:
            raise ValueError(
                f"Missing protocol field {key}; rerun with the current evaluator"
            )
        if flow["protocol"][key] != btm["protocol"][key]:
            raise ValueError(f"Unmatched evaluation field: {key}")
    if (
        flow["protocol"].get("method") != "flow"
        or btm["protocol"].get("method") != "btm"
    ):
        raise ValueError("Expected a flow evaluation followed by a BTM evaluation")
    a = np.asarray(flow["episode_successes"], dtype=float)
    b = np.asarray(btm["episode_successes"], dtype=float)
    n = len(flow["protocol"]["episode_indices"])
    if a.shape != (n,) or b.shape != (n,) or n < 2:
        raise ValueError("Need at least two paired episode outcomes")
    for values, report in ((a, flow), (b, btm)):
        if not np.isin(values, [0, 1]).all() or not np.isclose(
            values.mean(), report["metrics"]["success_rate"]
        ):
            raise ValueError("Success metrics and per-episode outcomes disagree")
    differences = b - a
    rng = np.random.default_rng(42)
    boot = differences[rng.integers(n, size=(10000, n))].mean(1)
    fm, bm = flow["metrics"], btm["metrics"]
    latency_key = "planning_batch_latency_ms_mean"
    same_hardware = bool(flow.get("hardware")) and flow.get("hardware") == btm.get(
        "hardware"
    )
    result = {
        "protocol": {key: flow["protocol"][key] for key in MATCHED},
        "flow_success_rate": float(a.mean()),
        "btm_success_rate": float(b.mean()),
        "btm_minus_flow_success_pp": 100 * float(differences.mean()),
        "paired_episode_bootstrap_delta_ci95_pp": (
            100 * np.quantile(boot, [0.025, 0.975])
        ).tolist(),
        "btm_only_successes": int(((b == 1) & (a == 0)).sum()),
        "flow_only_successes": int(((a == 1) & (b == 0)).sum()),
        "full_controller_latency_speedup_flow_over_btm": fm[latency_key]
        / bm[latency_key]
        if same_hardware and bm[latency_key] > 0
        else None,
        "latency_hardware_match": same_hardware,
        "flow_hardware": flow.get("hardware"),
        "btm_hardware": btm.get("hardware"),
        "flow_checkpoint_sha256": flow["protocol"].get("checkpoint_sha256"),
        "btm_checkpoint_sha256": btm["protocol"].get("checkpoint_sha256"),
        "flow_metrics": fm,
        "btm_metrics": bm,
        "limits": "Interval resamples evaluation episodes for this training-seed pair only; it does not include training-seed uncertainty. Latency is descriptive unless hardware and load also match. Boundary MSE and model-predicted errors are not task success.",
    }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", type=Path, required=True)
    p.add_argument("--btm", type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = compare(
        json.loads(args.flow.read_text()), json.loads(args.btm.read_text())
    )
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
