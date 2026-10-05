#!/usr/bin/env python3
"""Export auditable paired metrics and plots from a read-only campaign snapshot."""

import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.compare_evaluations import compare


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def table(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def publishable_report(report):
    result = deepcopy(report)
    # Keep the source identity, not a private absolute scratch-directory path.
    protocol = result.get("protocol", {})
    training_data = protocol.get("training_protocol", {}).get("data", {})
    if "manifest" in training_data:
        training_data["manifest"] = "sha256:" + protocol["manifest_sha256"]
    return result


def analyze(snapshot, out):
    out.mkdir(parents=True, exist_ok=True)
    s = json.loads(snapshot.read_text())
    evaluations, complete, train_rows, eval_rows, diagnostics = {}, {}, [], [], []
    summary = {
        "snapshot_started_utc": s["collection_started_utc"],
        "snapshot_finished_utc": s["collection_finished_utc"],
        "raw_snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "campaign": s["campaign"], "frozen_plan": s["frozen_plan"],
        "dataset": s["dataset"], "runs": {},
        "limits": [
            "One training seed; twenty fixed validation episodes per offset.",
            "Repeated checkpoints reuse cases and are not independent trials.",
            "Latency is descriptive under concurrent load on different physical GPUs.",
            "Generative objectives differ; do not compare their raw losses directly.",
            "Snapshot files can advance during collection; incomplete evaluations are excluded from complete-cycle comparisons.",
        ],
    }
    for method in ("btm", "flow"):
        r = s["runs"][method + "_3072"]
        evaluations[method] = {}
        errors = []
        for record in r["evaluations"]:
            if "data" not in record:
                errors.append(record.get("error", "missing evaluation data"))
                continue
            match = re.search(r"step_(\d+)_offset_(\d+)\.json$", record["file"])
            step, offset = map(int, match.groups())
            evaluations[method][step, offset] = record["data"]
            eval_rows.append(dict(method=method, step=step, offset=offset,
                                 saved_utc=record["mtime_utc"], **record["data"]["metrics"]))
        complete[method] = sorted(step for step, offset in evaluations[method]
                                  if offset == 100 and all((step, o) in evaluations[method] for o in (25, 50, 100)))
        rows = r["metrics"]["rows"]
        train_rows.extend(dict(method=method, **row) for row in rows)
        checkpoints = {name: {k: v for k, v in cp.items() if k != "file"}
                       for name, cp in r["checkpoints"].items() if cp}
        latest = complete[method][-1] if complete[method] else None
        best = checkpoints.get("best.pt", {}).get("step")

        def scores(step):
            if step is None or step not in complete[method]:
                return None
            values = {str(o): evaluations[method][step, o]["metrics"]["success_rate"]
                      for o in (25, 50, 100)}
            return dict(step=step, success_by_offset=values,
                        mean_success=float(np.mean(list(values.values()))))

        summary["runs"][method] = dict(
            latest_training_step=r["metrics"]["latest_training_row"]["step"],
            latest_complete_evaluation=scores(latest),
            selected_checkpoint_evaluation=scores(best), checkpoints=checkpoints,
            complete_evaluation_steps=complete[method], evaluation_read_errors=errors,
            metrics_parse_errors=r["metrics"]["parse_errors"],
            provenance=r["provenance"]["data"],
            started_utc=r["earliest_wandb_start_utc"],
            training_complete=bool(r["training_complete"]),
        )
        if latest is not None:
            first = complete[method][0]
            for offset in (25, 50, 100):
                a, b = evaluations[method][first, offset], evaluations[method][latest, offset]
                for field in ("episode_indices", "start_steps"):
                    if a["protocol"][field] != b["protocol"][field]:
                        raise ValueError(f"{method}: longitudinal pairs changed")
                for episode, start, old, new in zip(a["protocol"]["episode_indices"], a["protocol"]["start_steps"], a["episode_successes"], b["episode_successes"]):
                    behavior = {(False, False): "failed_both", (True, False): "lost_success",
                                (False, True): "gained_success", (True, True): "succeeded_both"}[bool(old), bool(new)]
                    diagnostics.append(dict(method=method, offset=offset, first_step=first,
                                            latest_step=latest, episode=episode, start_step=start,
                                            first_success=int(old), latest_success=int(new), transition=behavior))
    common = sorted(set(complete["btm"]) & set(complete["flow"]))
    paired, paired_rows = [], []
    for step in common:
        for offset in (25, 50, 100):
            result = compare(evaluations["flow"][step, offset], evaluations["btm"][step, offset])
            paired.append(dict(optimizer_step=step, **publishable_report(result)))
            ci = result["paired_episode_bootstrap_delta_ci95_pp"]
            paired_rows.append(dict(step=step, offset=offset, episodes=len(result["protocol"]["episode_indices"]),
                btm_success=result["btm_success_rate"], flow_success=result["flow_success_rate"],
                btm_minus_flow_pp=result["btm_minus_flow_success_pp"], ci95_low_pp=ci[0], ci95_high_pp=ci[1],
                btm_only_successes=result["btm_only_successes"], flow_only_successes=result["flow_only_successes"]))
    summary["latest_complete_matched_step"] = common[-1] if common else None
    summary["latest_matched_results"] = [row for row in paired_rows if row["step"] == summary["latest_complete_matched_step"]]
    dump(out / "summary.json", summary)
    dump(out / "paired_evaluations.json", paired)
    table(out / "training_metrics.csv", train_rows)
    table(out / "evaluation_metrics.csv", eval_rows)
    table(out / "paired_success.csv", paired_rows)
    table(out / "episode_transitions.csv", diagnostics)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    colors = {"btm": "#087f8c", "flow": "#d67a24"}
    for method in ("btm", "flow"):
        steps = complete[method]
        success = [evaluations[method][x, 100]["metrics"]["success_rate"] * 100 for x in steps]
        mean = [np.mean([evaluations[method][x, o]["metrics"]["success_rate"] for o in (25, 50, 100)]) * 100 for x in steps]
        for ax, y in ((axes[0, 0], success), (axes[0, 1], mean)):
            ax.plot(steps, y, marker="o", markersize=3, lw=1.7, color=colors[method], label=method.upper())
        rows = s["runs"][method + "_3072"]["metrics"]["rows"]
        for ax, key in ((axes[1, 0], "val/path_mse_single"), (axes[1, 1], "val/candidate_variance")):
            points = {r["step"]: r[key] for r in rows if key in r}
            ax.plot(list(points), list(points.values()), color=colors[method], lw=1.7, label=method.upper())
    titles = ["Long-distance task success (offset 100)", "Mean task success (offsets 25 / 50 / 100)",
              "Single-path latent error (32 clips)", "Absolute latent candidate variance (32 clips)"]
    labels = ["Success (%)", "Success (%)", "Latent MSE", "Variance (not behavioral mode coverage)"]
    for ax, title, label in zip(axes.flat, titles, labels):
        ax.set_title(title, loc="left", weight="bold", fontsize=11)
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel(label)
        ax.grid(alpha=0.18)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k"))
        ax.legend(frameon=False)
    for ax in axes[0]:
        ax.set_ylim(0, 60)
        if common:
            ax.axvline(common[-1], color="#999999", ls=":", lw=1)
    fig.suptitle("PushT control comparison: task success and latent proxies", weight="bold", fontsize=15)
    fig.supxlabel("One training seed • 20 fixed validation episodes per offset • dotted line: latest complete matched step\n"
                  + s["collection_finished_utc"] + " • Later BTM points have no matched flow result yet", fontsize=9)
    fig.savefig(out / "curves.png", dpi=170)
    fig.savefig(out / "curves.svg")
    svg = out / "curves.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.snapshot, args.out)
