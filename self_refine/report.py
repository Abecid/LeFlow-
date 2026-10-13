"""Aggregate only measured, completed, protocol-compatible native results."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from .protocol import paired_difference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_root")
    p.add_argument("--output",required=True)
    args=p.parse_args()
    runs=[json.loads(f.read_text()) for f in sorted(Path(args.run_root).glob("jobs/*/result.json"))]
    runs=[r for r in runs if r.get("status")=="completed"]
    if not runs:
        raise SystemExit("No completed benchmark evaluations. No scores or plots generated.")
    rows=[]
    for run in runs:
        references=[r for r in runs if r["method"]["name"]=="leflow" and
                    r["protocol_hash"]==run["protocol_hash"]]
        if len(references)!=1:
            raise ValueError("each protocol must have exactly one completed native LeFlow reference")
        diff=paired_difference(references[0],run)
        rows.append({"task":run["task"],"seed":run["seed"],"method":run["method"]["name"],
            "successes":run["successes"],"evaluations":run["evaluations"],
            "success_percent":run["success_rate_percent"],
            "mean_batch_controller_ms":run["timing"]["mean_batch_controller_ms"],
            "p95_batch_controller_ms":run["timing"]["p95_batch_controller_ms"],
            "flow_samples":run["timing"]["flow_samples"],
            "world_transitions":run["timing"]["world_transitions"],
            "backward_calls":run["timing"]["backward_calls"],**diff})
    dest=Path(args.output); dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open("x",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(f"Wrote {len(rows)} measured comparisons to {dest}")


if __name__=="__main__": main()
