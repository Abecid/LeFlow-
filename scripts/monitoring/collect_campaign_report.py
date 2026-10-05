#!/usr/bin/env python3
"""Read a live matched campaign and emit JSON; never modify training state.

Run with the existing runtime environment. Checkpoints are trusted artifacts
created by this campaign and are loaded on CPU. No network or W&B API is used.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time


OMIT = {
    "hostname",
    "boot_id",
    "uuid",
    "gpu_uuid",
    "gpu_uuids",
    "btm_gpu_uuids",
    "cuda_visible_devices",
}


def clean(value):
    if isinstance(value, dict):
        return {
            key: clean(item)
            for key, item in value.items()
            if key.lower() not in OMIT
            and not any(
                term in key.lower()
                for term in (
                    "password",
                    "api_key",
                    "access_token",
                    "secret",
                    "private_key",
                )
            )
        }
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def utc(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="seconds")


def metadata(path, stat=None):
    stat = stat or path.stat()
    return dict(
        file=str(path),
        bytes=stat.st_size,
        mtime_utc=utc(stat.st_mtime),
        mtime_unix=stat.st_mtime,
    )


def read_json(path):
    if not path.exists():
        return None
    try:
        with path.open() as stream:
            return {
                **metadata(path, os.fstat(stream.fileno())),
                "data": json.load(stream),
            }
    except Exception as exc:
        return dict(file=str(path), error=f"{type(exc).__name__}: {exc}")


def read_metrics(path):
    if not path.exists():
        return None
    with path.open() as stream:
        lines = stream.readlines()
        result = metadata(path, os.fstat(stream.fileno()))
    rows, errors = [], []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            errors.append(
                dict(
                    line=index + 1,
                    trailing_partial=index == len(lines) - 1
                    and not line.endswith("\n"),
                )
            )
    result.update(
        rows=rows,
        parse_errors=errors,
        latest_training_row=next(
            (row for row in reversed(rows) if "train/loss" in row), None
        ),
    )
    result["timestamp_note"] = (
        "Rows have optimizer steps but no individual wall timestamps; mtime is the latest observed metrics-file write."
    )
    return result


def read_checkpoint(path):
    if not path.exists():
        return None
    try:
        import torch

        with path.open("rb") as stream:
            info = metadata(path, os.fstat(stream.fileno()))
            payload = torch.load(stream, map_location="cpu", weights_only=False)
        info.update(
            {
                key: payload.get(key)
                for key in (
                    "step",
                    "epoch",
                    "next_batch",
                    "best_success",
                    "wandb_id",
                    "world_size",
                    "manifest_sha256",
                )
            }
        )
        info.update(
            method=payload.get("config", {}).get("method"),
            seed=payload.get("config", {}).get("seed"),
        )
        del payload
        return info
    except Exception as exc:
        return dict(file=str(path), error=f"{type(exc).__name__}: {exc}")


def wandb_sessions(run_dir, observed_at):
    sessions = []
    for path in sorted((run_dir / "wandb").glob("*run-*")):
        match = re.fullmatch(r"(?:offline-)?run-(\d{8}_\d{6})-(.+)", path.name)
        if not path.is_dir() or not match:
            continue
        # W&B's directory clock is host-local; convert it explicitly to UTC.
        started = datetime.strptime(match[1], "%Y%m%d_%H%M%S").astimezone(timezone.utc)
        sessions.append(
            dict(
                directory=str(path),
                run_id=match[2],
                started_utc=started.isoformat(timespec="seconds"),
                started_unix=started.timestamp(),
                elapsed_seconds=max(0, observed_at - started.timestamp()),
                directory_clock_timezone=list(time.tzname),
                offline=path.name.startswith("offline-"),
            )
        )
    return sessions


def nvidia_snapshot():
    fields = [
        "index",
        "name",
        "utilization.gpu",
        "memory.used",
        "memory.total",
        "power.draw",
    ]
    try:
        text = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=" + ",".join(fields),
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=20,
        )
        return dict(
            observed_utc=utc(time.time()),
            units={
                "utilization.gpu": "%",
                "memory.used": "MiB",
                "memory.total": "MiB",
                "power.draw": "W",
            },
            gpus=[
                dict(zip(fields, [item.strip() for item in row]))
                for row in csv.reader(io.StringIO(text))
                if row
            ],
        )
    except Exception as exc:
        return dict(error=f"{type(exc).__name__}: {exc}")


def relevant_processes(record, campaign, repo):
    """Keep only campaign processes, without copying their full command lines."""
    results = []
    try:
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        ticks = os.sysconf("SC_CLK_TCK")
    except OSError:
        return {"error": "Linux /proc is unavailable"}
    names = {
        "train_subgoals.py",
        "eval_subgoals.py",
        "run_comparison.py",
        "run_gpu_stages.sh",
        "launch_concurrent_flow.py",
        "launch_when_idle.py",
        "collect_campaign_report.py",
    }
    for root in Path("/proc").iterdir():
        if not root.name.isdigit():
            continue
        try:
            args = (root / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
            scripts = [Path(arg).name for arg in args if Path(arg).name in names]
            if not scripts:
                continue
            cwd = str((root / "cwd").resolve())
            if not any(
                str(record) in arg or str(campaign) in arg for arg in args
            ) and cwd != str(repo):
                continue
            stat = (root / "stat").read_text().rsplit(")", 1)[1].split()
            method = next(
                (
                    name
                    for name in ("btm", "flow")
                    if any(re.search(rf"(?:^|[/=_]){name}(?:_|$)", arg) for arg in args)
                ),
                None,
            )
            results.append(
                dict(
                    pid=int(root.name),
                    ppid=int(stat[1]),
                    state=stat[0],
                    elapsed_seconds=round(max(0, uptime - int(stat[19]) / ticks), 1),
                    scripts=scripts,
                    method=method,
                )
            )
        except (OSError, ValueError, IndexError):
            continue
    return sorted(results, key=lambda item: item["pid"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record-root",
        type=Path,
        default=Path("/home/mtxu/adam/LeFlow-experiments/20261004"),
    )
    parser.add_argument(
        "--stablewm-home",
        type=Path,
        default=Path(os.environ.get("STABLEWM_HOME", "/tmp/mtxu-leflow-20261004/data")),
    )
    parser.add_argument("--campaign", default="pusht_pair_20261004")
    args = parser.parse_args()
    began = time.time()
    record = args.record_root.resolve()
    campaign = args.stablewm_home.resolve() / "runs" / args.campaign
    repo = Path(os.environ.get("LEFLOW_REPO", str(record / "repo"))).resolve()
    plan_record = read_json(campaign / "comparison.json")
    plan = (plan_record or {}).get("data", {})
    manifest = read_json(Path(plan["manifest"])) if plan.get("manifest") else None
    manifest_data = (manifest or {}).get("data", {})
    runs = {}
    for job in plan.get("jobs", []):
        run_dir = Path(job["run_dir"])
        metrics = read_metrics(run_dir / "metrics.jsonl")
        evaluations = [
            read_json(path)
            for path in sorted((run_dir / "eval").glob("step_*_offset_*.json"))
        ]
        checkpoints = {
            name: read_checkpoint(run_dir / name) for name in ("last.pt", "best.pt")
        }
        events = []
        for name, checkpoint in checkpoints.items():
            if checkpoint and "error" not in checkpoint:
                events.append(
                    dict(
                        source=name,
                        step=checkpoint["step"],
                        observed_utc=checkpoint["mtime_utc"],
                        observed_unix=checkpoint["mtime_unix"],
                    )
                )
        for evaluation in evaluations:
            if evaluation and "error" not in evaluation:
                match = re.search(r"step_(\d+)_offset_(\d+)\.json$", evaluation["file"])
                events.append(
                    dict(
                        source="closed_loop_evaluation",
                        step=int(match[1]),
                        goal_offset=int(match[2]),
                        observed_utc=evaluation["mtime_utc"],
                        observed_unix=evaluation["mtime_unix"],
                    )
                )
        sessions = wandb_sessions(run_dir, began)
        runs[run_dir.name] = dict(
            method=job["method"],
            seed=job["seed"],
            run_dir=str(run_dir),
            config=read_json(run_dir / "config.json"),
            frozen_config=job,
            provenance=read_json(run_dir / "provenance.json"),
            metrics=metrics,
            evaluations=evaluations,
            checkpoints=checkpoints,
            training_complete=read_json(run_dir / "training_complete.json"),
            wandb_sessions=sessions,
            observed_progress_events=sorted(
                events, key=lambda event: event["observed_unix"]
            ),
            earliest_wandb_start_utc=min(
                (session["started_utc"] for session in sessions), default=None
            ),
        )
    status_paths = set(record.glob("*status.json"))
    for name in ("queue.json", "gpu-queue.json", "execution-allocation.json"):
        if (record / name).exists():
            status_paths.add(record / name)
    report = dict(
        format_version=1,
        collection_started_utc=utc(began),
        collection_finished_utc=utc(time.time()),
        campaign=args.campaign,
        frozen_plan={
            key: plan.get(key)
            for key in (
                "format_version",
                "name",
                "gpus",
                "seeds",
                "method_order",
                "git_commit",
                "manifest_sha256",
                "encoder_sha256",
                "primary_goal_offset",
            )
        },
        dataset=dict(
            manifest_sha256=plan.get("manifest_sha256"),
            encoder_sha256=plan.get("encoder_sha256"),
            split_counts={
                name: len(manifest_data.get("split", {}).get(name, []))
                for name in ("train", "val", "test")
            },
            episodes=manifest_data.get("source", {}).get("episodes"),
            frames=manifest_data.get("source", {}).get("frames"),
            test_fixture=manifest_data.get("test_fixture"),
        ),
        runs=runs,
        controller_status={path.name: read_json(path) for path in sorted(status_paths)},
        hardware=nvidia_snapshot(),
        processes=relevant_processes(record, campaign, repo),
        timing_note="Read-only live snapshot; files may advance during collection. Evaluation/checkpoint mtimes confirm when their saved optimizer steps were observed. Metrics rows do not contain per-row wall timestamps. W&B directory starts use the server's local clock converted to UTC.",
    )
    if not plan:
        report["plan_error"] = plan_record or "Frozen comparison.json not found"
    report["collection_finished_utc"] = utc(time.time())
    print(json.dumps(clean(report), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
