#!/usr/bin/env python3
"""Estimate matched-campaign finish times from a collector JSON snapshot.

Pure standard-library analysis: no network, GPU operations, or process signals.
Public entry point: estimate_eta(snapshot). CLI: python eta_estimator.py snapshot.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import statistics
from zoneinfo import ZoneInfo


def _timestamp(value):
    return datetime.fromisoformat(value).timestamp()


def _dates(timestamp):
    return {
        "utc": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(
            timespec="seconds"
        ),
        "america_new_york": datetime.fromtimestamp(
            timestamp, ZoneInfo("America/New_York")
        ).isoformat(timespec="seconds"),
    }


def _quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return (
        values[low]
        if low == high
        else values[low] * (high - position) + values[high] * (position - low)
    )


def _run_eta(run, observed, concurrent_start):
    cfg = (run.get("config") or {}).get("data") or run["frozen_config"]
    provenance = run["provenance"]["data"]
    checkpoint = run["checkpoints"].get("last.pt") or {}
    if "step" not in checkpoint:
        return {"state": "insufficient_data", "reason": "No readable last checkpoint"}
    rows = (run.get("metrics") or {}).get("rows", [])
    train_rows = [row for row in rows if "system/samples_per_second" in row]
    checkpoint_step = checkpoint["step"]
    logged_step = max((row["step"] for row in train_rows), default=0)
    current = max(checkpoint_step, logged_step)
    batches = (
        provenance["train_clips"] // provenance["world_size"] // cfg["micro_batch_size"]
    )
    steps_per_epoch = batches // provenance["accumulation"]
    target = min(steps_per_epoch * cfg["epochs"], cfg.get("max_steps") or math.inf)
    target = int(target)
    evaluation = cfg["evaluation"]
    offsets = evaluation["goal_offsets"]
    schedule = set()
    if evaluation["enabled"]:
        schedule.update(
            range(evaluation["every_steps"], target + 1, evaluation["every_steps"])
        )
        schedule.update(
            range(
                steps_per_epoch * evaluation["every_epochs"],
                target + 1,
                steps_per_epoch * evaluation["every_epochs"],
            )
        )
        if evaluation["at_end"]:
            schedule.add(target)
    schedule = sorted(schedule)
    groups = {}
    for item in run.get("evaluations", []):
        if not item or "error" in item:
            continue
        match = re.search(r"step_(\d+)_offset_(\d+)", item["file"])
        if match:
            groups.setdefault(int(match[1]), {})[int(match[2])] = item
    complete = [
        {
            "step": step,
            "unix": max(group[offset]["mtime_unix"] for offset in offsets),
            "offsets": group,
        }
        for step, group in sorted(groups.items())
        if set(offsets).issubset(group)
    ]
    base = {
        "current_checkpoint_step": checkpoint_step,
        "current_logged_training_step": logged_step,
        "current_optimizer_step": current,
        "target_steps": target,
        "steps_per_epoch": steps_per_epoch,
        "epochs": cfg["epochs"],
        "step_progress_fraction": min(1, current / target),
        "remaining_optimizer_steps": max(0, target - current),
        "completed_eval_cycles": len(complete),
        "total_scheduled_eval_cycles": len(schedule),
        "completion_marker_present": bool(run.get("training_complete")),
    }
    if run.get("training_complete") or (
        current >= target
        and (
            not evaluation["enabled"]
            or any(item["step"] == target for item in complete)
        )
    ):
        marker = run.get("training_complete") or {}
        completed_at = marker.get("mtime_unix") or (
            complete[-1]["unix"] if complete else checkpoint["mtime_unix"]
        )
        base.update(
            state="complete"
            if marker
            else "training_budget_and_final_evaluation_complete",
            eta={
                "point": _dates(completed_at),
                "range_lower": _dates(completed_at),
                "range_upper": _dates(completed_at),
                "remaining_minutes_point": 0,
                "remaining_minutes_range": [0, 0],
            },
            handoff_note="Without a completion marker, final cleanup or the paused original supervisor's bookkeeping can still be pending.",
        )
        return base
    if len(complete) < 2:
        base.update(
            state="insufficient_data",
            reason="Need at least two complete three-offset evaluation endpoints",
        )
        return base
    intervals = []
    for left, right in zip(complete, complete[1:]):
        if left["unix"] >= concurrent_start:
            intervals.append(
                {
                    "from_step": left["step"],
                    "to_step": right["step"],
                    "from_time": _dates(left["unix"]),
                    "to_time": _dates(right["unix"]),
                    "delta_steps": right["step"] - left["step"],
                    "delta_wall_seconds": right["unix"] - left["unix"],
                }
            )
    intervals = intervals[-12:]
    disruptive = sorted(
        set(schedule)
        | set(range(cfg["validate_every"], target + 1, cfg["validate_every"]))
    )
    pure = []
    for left, right in zip(train_rows, train_rows[1:]):
        if (
            right["step"] - left["step"] != cfg["log_every"]
            or right["step"] <= max(current - 3000, 0)
            or any(left["step"] <= step < right["step"] for step in disruptive)
        ):
            continue
        throughput = right["system/samples_per_second"]
        if throughput > 0 and math.isfinite(throughput):
            pure.append(cfg["global_batch_size"] / throughput)
    recent = complete[-10:]
    raw_eval_seconds = statistics.mean(
        sum(
            item["offsets"][offset]["data"]["metrics"]["evaluation_seconds"]
            for offset in offsets
        )
        for item in recent
    )
    launch_gaps = [
        item["offsets"][nxt]["mtime_unix"] - item["offsets"][prev]["mtime_unix"]
        for item in recent
        for prev, nxt in zip(offsets, offsets[1:])
    ]
    full_cycle_wall = (
        len(offsets) * statistics.mean(launch_gaps) if launch_gaps else raw_eval_seconds
    )
    xs, ys = (
        [item["delta_steps"] for item in intervals],
        [item["delta_wall_seconds"] for item in intervals],
    )
    variance = sum((x - statistics.mean(xs)) ** 2 for x in xs) if xs else 0
    if len(intervals) >= 3 and variance > 0:
        xm, ym = statistics.mean(xs), statistics.mean(ys)
        step_seconds = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / variance
        cycle_seconds = ym - step_seconds * xm
        fit_method = (
            "OLS consecutive complete-cycle intervals during concurrent execution"
        )
    elif pure:
        step_seconds, cycle_seconds = statistics.mean(pure), full_cycle_wall
        fit_method = "Fallback: clean training windows plus observed within-evaluation launch gaps"
    else:
        base.update(
            state="insufficient_data",
            reason="No identifiable cycle fit or clean training timing",
        )
        return base
    if step_seconds <= 0 or cycle_seconds <= 0:
        base.update(
            state="insufficient_data",
            reason="Timing fit has nonpositive coefficients; inspect stalls/restarts",
        )
        return base
    for item in intervals:
        item["residual_seconds"] = (
            item["delta_wall_seconds"]
            - step_seconds * item["delta_steps"]
            - cycle_seconds
        )
    anchor = complete[-1]
    remaining_from_anchor = [step for step in schedule if step > anchor["step"]]
    remaining_from_current = [step for step in schedule if step > current]
    modeled_seconds = step_seconds * (target - anchor["step"]) + cycle_seconds * len(
        remaining_from_anchor
    )
    point = anchor["unix"] + modeled_seconds + 15
    low = anchor["unix"] + 0.85 * modeled_seconds + 5
    high = anchor["unix"] + 1.15 * modeled_seconds + 60
    current_offsets = groups.get(current, {})
    partial = (
        current == checkpoint_step
        and current in schedule
        and len(current_offsets) < len(offsets)
    )
    partial_remaining = (
        max(0, full_cycle_wall - (observed - checkpoint["mtime_unix"]))
        if partial
        else 0
    )
    training_remaining = (
        max(0, target - current) * statistics.mean(pure) if pure else None
    )
    eval_remaining = partial_remaining + len(remaining_from_current) * full_cycle_wall
    overdue = point < observed
    base.update(
        state="cadence_projection_overdue" if overdue else "projected",
        last_complete_eval={"step": anchor["step"], "time": _dates(anchor["unix"])},
        current_partial_evaluation={
            "in_progress": partial,
            "offsets_already_saved_when_read": sorted(current_offsets),
            "estimated_seconds_remaining": partial_remaining,
            "note": "File observations may differ by seconds during live collection; use checkpoint-time phase estimate, not each missing JSON as a whole unfinished offset.",
        },
        remaining_eval_steps_after_current=remaining_from_current,
        remaining_eval_steps_after_last_complete=remaining_from_anchor,
        fit={
            "method": fit_method,
            "interval_count": len(intervals),
            "seconds_per_optimizer_step": step_seconds,
            "seconds_per_three_offset_eval_cycle": cycle_seconds,
            "residual_rmse_seconds": math.sqrt(
                statistics.mean(item["residual_seconds"] ** 2 for item in intervals)
            )
            if intervals
            else None,
            "intervals": intervals,
        },
        clean_training_rate={
            "log_window_count": len(pure),
            "mean_seconds_per_step": statistics.mean(pure),
            "median_seconds_per_step": statistics.median(pure),
            "p10_seconds_per_step": _quantile(pure, 0.1),
            "p90_seconds_per_step": _quantile(pure, 0.9),
        }
        if pure
        else None,
        evaluation_rate={
            "recent_complete_cycles": len(recent),
            "mean_reported_evaluator_seconds_sum_three_offsets": raw_eval_seconds,
            "mean_three_offset_wall_seconds_including_subprocess_startup": full_cycle_wall,
            "mean_unlogged_startup_io_seconds_per_three_offsets": full_cycle_wall
            - raw_eval_seconds,
        },
        remaining_cost_separation={
            "pure_training_seconds": training_remaining,
            "estimated_environment_evaluation_seconds_including_launches": eval_remaining,
            "remaining_other_overhead_seconds_by_difference": point
            - observed
            - training_remaining
            - eval_remaining
            if training_remaining is not None
            else None,
            "remaining_seconds_point": max(0, point - observed),
        },
        eta={
            "point": _dates(point),
            "range_lower": _dates(low),
            "range_upper": _dates(high),
            "remaining_minutes_point": max(0, point - observed) / 60,
            "remaining_minutes_range": [
                max(0, low - observed) / 60,
                max(0, high - observed) / 60,
            ],
        },
    )
    if overdue:
        base["warning"] = (
            "Recent-cadence projection is already past, while completion is unconfirmed. Investigate active phase/stall; zero projected remaining minutes is not completion."
        )
    return base


def estimate_eta(snapshot):
    """Return auditable ETA data without touching runtime files or processes."""
    observed = _timestamp(snapshot["collection_finished_utc"])
    concurrent = (
        (snapshot.get("controller_status", {}).get("concurrency-status.json") or {})
        .get("data", {})
        .get("launched_at")
    )
    concurrent_start = _timestamp(concurrent) if concurrent else 0
    results = {}
    for name, run in snapshot.get("runs", {}).items():
        try:
            results[name] = _run_eta(run, observed, concurrent_start)
        except (KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
            results[name] = {
                "state": "insufficient_data",
                "reason": f"{type(exc).__name__}: {exc}",
            }
    report = {
        "format_version": 1,
        "snapshot_time": _dates(observed),
        "methodology": {
            "point_formula": "last_complete_eval_time + fitted_seconds_per_step * (target_steps - last_complete_eval_step) + fitted_seconds_per_three_offset_cycle * remaining_scheduled_cycles + 15 seconds finalization",
            "schedule": "Union of every-1000-step evaluations and configured epoch ends; final at_end evaluation is deduplicated when final epoch already evaluates.",
            "phase_control": "Fit complete evaluation endpoint intervals; exclude pre-concurrent intervals and use the latest 12. Subtract elapsed time since the last complete endpoint, so an in-progress evaluation is not counted afresh.",
            "range": "Sensitivity envelope: 0.85/1.15 multipliers on fitted training and evaluation rates, plus 5/60 seconds finalization. Planning range, not a statistical confidence interval.",
            "separation": "Pure training timing excludes log windows crossing local validation or environment evaluation; subprocess startup/I/O is inferred from consecutive offset file mtimes and compared with evaluator-reported durations.",
            "assumptions": "No failure, restart, or major host-load change. Remaining wall time can be dominated by evaluation rather than optimizer steps.",
        },
        "runs": results,
    }
    if results and all("eta" in result for result in results.values()):
        report["both_runs_complete_eta"] = {
            key: _dates(
                max(
                    _timestamp(result["eta"][key]["utc"]) for result in results.values()
                )
            )
            for key in ("point", "range_lower", "range_upper")
        }
        report["both_runs_complete_eta"].update(
            remaining_minutes_point=max(
                result["eta"]["remaining_minutes_point"] for result in results.values()
            ),
            remaining_minutes_range=[
                max(
                    result["eta"]["remaining_minutes_range"][index]
                    for result in results.values()
                )
                for index in (0, 1)
            ],
            completion_unconfirmed=any(
                not result.get("completion_marker_present")
                for result in results.values()
            ),
            projection_overdue=any(
                result["state"] == "cadence_projection_overdue"
                for result in results.values()
            ),
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = estimate_eta(json.loads(args.snapshot.read_text()))
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
