"""Case identities, immutable run records, and paired native-benchmark summaries."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [jsonable(x) for x in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    return value


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(jsonable(value),sort_keys=True,allow_nan=False,
                                     separators=(",",":")).encode()).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as source:
        for data in iter(lambda: source.read(8*1024*1024),b""):
            digest.update(data)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: Any) -> None:
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(jsonable(payload),sort_keys=True,indent=2,allow_nan=False)+"\n"
    temp=path.with_name(path.name+f".{os.getpid()}.tmp")
    with temp.open("w",encoding="utf-8") as dest:
        dest.write(text); dest.flush(); os.fsync(dest.fileno())
    os.replace(temp,path)


def select_cases(episode_ids: np.ndarray, steps: np.ndarray, *,
                 goal_offset: int, num_eval: int, seed: int,
                 heldout: set[int] | None = None) -> list[dict]:
    """Same row sampling and sorting as native eval.py, no outcome filtering."""
    episode_ids=np.asarray(episode_ids)
    steps=np.asarray(steps)
    if episode_ids.ndim!=1 or steps.shape!=episode_ids.shape:
        raise ValueError("episode ids and steps must be aligned vectors")
    if num_eval<1 or goal_offset<1:
        raise ValueError("num_eval and goal_offset must be positive")
    unique=np.unique(episode_ids)
    max_start={ep:int(steps[episode_ids==ep].max())-goal_offset for ep in unique}
    valid=steps<=np.array([max_start[e] for e in episode_ids])
    if heldout is not None:
        if not heldout: raise ValueError("heldout episode set is empty")
        valid &= np.isin(episode_ids,list(heldout))
    indices=np.nonzero(valid)[0]
    if len(indices)<num_eval:
        raise ValueError(f"requested {num_eval} cases, found {len(indices)} valid starting rows")
    picked=np.sort(np.random.default_rng(seed).choice(indices,size=num_eval,replace=False))
    return [{"case_index":i,"row":int(row),"episode":int(episode_ids[row]),
             "start":int(steps[row]),"goal":int(steps[row])+goal_offset}
            for i,row in enumerate(picked)]


def validate_metrics(metrics: dict, cases: list[dict]) -> list[dict]:
    """Pinned stable-worldmodel reports success_rate in percent, not fraction."""
    values=np.asarray(metrics.get("episode_successes",[]))
    if values.shape!=(len(cases),) or not np.isin(values,[0,1]).all():
        raise ValueError("missing/invalid per-case native episode_successes")
    expected=100.0*float(values.mean())
    reported=float(metrics.get("success_rate",float('nan')))
    if not math.isfinite(reported) or not math.isclose(expected,reported,abs_tol=1e-6):
        raise ValueError("native aggregate success_rate does not match per-case outcomes")
    return [{**c,"success":bool(v)} for c,v in zip(cases,values)]


def summarize_telemetry(records: list[dict]) -> dict:
    if not records:
        raise ValueError("no controller telemetry; cannot report latency")
    batch_times=np.asarray([r["controller_ms"] for r in records],dtype=float)
    if not np.isfinite(batch_times).all() or (batch_times<0).any():
        raise ValueError("invalid controller timings")
    widths=[r["env_end"]-r["env_start"] for r in records]
    if min(widths)<1: raise ValueError("invalid telemetry batch widths")
    # Latency is per actual batched call. Do not divide by B and call it latency.
    return {
        "timed_batches":len(records), "batch_sizes":sorted(set(widths)),
        "mean_batch_controller_ms":float(batch_times.mean()),
        "p50_batch_controller_ms":float(np.percentile(batch_times,50)),
        "p95_batch_controller_ms":float(np.percentile(batch_times,95)),
        "p99_batch_controller_ms":float(np.percentile(batch_times,99)),
        "total_controller_seconds":float(batch_times.sum()/1000),
        **{key:sum(int(r[key]) for r in records) for key in
           ("flow_calls","flow_samples","world_calls","world_transitions",
            "inverse_calls","inverse_transitions","backward_calls","guided_candidates")},
    }


def paired_difference(baseline: dict, candidate: dict) -> dict:
    """Reject native/custom, data, checkpoint, horizon, or case-identity mixing."""
    for run in [baseline,candidate]:
        if run.get("status")!="completed":
            raise ValueError("cannot compare an incomplete run")
        if run.get("benchmark_family")!="native_leflow":
            raise ValueError("custom MetaWorld/crossbench results are a different protocol")
    if baseline["protocol_hash"]!=candidate["protocol_hash"]:
        raise ValueError("protocol/checkpoint/case identities differ")
    a,b=baseline["cases"],candidate["cases"]
    if len(a)!=len(b) or not a:
        raise ValueError("case counts differ or are empty")
    for x,y in zip(a,b):
        if {k:v for k,v in x.items() if k!="success"}!={k:v for k,v in y.items() if k!="success"}:
            raise ValueError("case identities/order differ")
    wins=sum(not x["success"] and y["success"] for x,y in zip(a,b))
    losses=sum(x["success"] and not y["success"] for x,y in zip(a,b))
    return {"cases":len(a),"wins":wins,"regressions":losses,
            "success_delta_pp":100.0*(wins-losses)/len(a)}
