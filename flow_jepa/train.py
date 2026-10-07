from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from .common import (
    barrier,
    checkpoint,
    config,
    digest,
    distributed,
    file_hash,
    gather,
    git_revision,
    save_json,
    seed_all,
)
from .data import Segments
from .evaluate import evaluate
from .models import System, prediction_loss, distance
from .vision import Encoder
from .budget import training_progress


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    )


def restore_rng(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None:
        torch.cuda.set_rng_state(s["cuda"])


def world_validation(model, c, root, device, rank, size):
    data = Segments(root, c, "world", 12345, 512, split="validation")
    loader = DataLoader(
        Subset(data, list(range(rank, len(data), size))),
        batch_size=c["training"]["micro_batch"],
    )
    total = torch.zeros(4, device=device)
    model.eval()
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            loss, _ = model(b)
            n, k, ad = b["a"].shape
            persistence = prediction_loss(
                b["z"][:, :1].expand_as(b["z"][:, 1:]), b["z"][:, 1:]
            )
            choices = b["a"][:, None].expand(n, 16, k, ad).clone()
            choices[:, 1:] = torch.rand_like(choices[:, 1:]) * 2 - 1
            z0 = b["z"][:, 0, None].expand(n, 16, *b["z"].shape[2:]).flatten(0, 1)
            future = model.world.rollout(z0, choices.flatten(0, 1))[:, -1]
            targets = b["z"][:, -1, None].expand(n, 16, *b["z"].shape[2:]).flatten(0, 1)
            ranked = distance(future, targets).reshape(n, 16)
            # Ties get fractional credit; an action-ignoring model scores chance.
            minimum = ranked.min(1, keepdim=True).values
            tied = torch.isclose(ranked, minimum, atol=1e-7, rtol=1e-5)
            identified = (tied[:, 0].float() / tied.sum(1)).sum()
            total += total.new_tensor(
                [float(loss) * n, n, float(persistence) * n, float(identified)]
            )
    if size > 1:
        dist.all_reduce(total)
    model.train()
    return dict(
        dynamics_loss=float(total[0] / total[1]),
        persistence_loss=float(total[2] / total[1]),
        action_identification_at_16=float(total[3] / total[1]),
        action_identification_chance=1 / 16,
    )


def train(c, root, run_dir, method, seed, world_path=None, *, allow_fixture=False):
    rank, size, device = distributed()
    if device.type != "cuda" and not allow_fixture:
        raise RuntimeError(
            "Real campaign requires an allocated CUDA GPU; CPU fixture tests are separate"
        )
    root, run_dir = Path(root), Path(run_dir)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("fixture") and not allow_fixture:
        raise ValueError("Fixture data cannot enter the real training campaign")
    if allow_fixture and not manifest.get("fixture"):
        raise ValueError(
            "CPU/offline allowance is restricted to explicitly marked fixtures"
        )
    manifest_hash = file_hash(manifest_path)
    tc = c["training"]
    micro, global_batch = tc["micro_batch"], tc["global_batch"]
    if global_batch % (micro * size):
        raise ValueError("Global batch must divide micro_batch * GPU count")
    accumulation = global_batch // (micro * size)
    steps = tc["world_steps"] if method == "world" else tc["planner_steps"]
    seed_all(seed)
    model = System(c, method).to(device)
    world, world_hash = None, None
    if method != "world":
        if world_path is None:
            raise ValueError("A frozen trained world checkpoint is required")
        w = torch.load(world_path, map_location="cpu", weights_only=False)
        if (
            w["protocol"] != digest(c)
            or w["manifest"] != manifest_hash
            or w["seed"] != seed
            or w["method"] != "world"
            or w["code"] != git_revision()
        ):
            raise ValueError("Wrong data/protocol/seed in frozen world checkpoint")
        ws = System(c, "world")
        ws.load_state_dict(w["model"])
        world = ws.world.to(device).eval().requires_grad_(False)
        world_hash = file_hash(world_path)
    lr = tc["world_learning_rate"] if method == "world" else tc["learning_rate"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=tc["weight_decay"]
    )
    start, best, run_id = 0, float("inf") if method == "world" else -1.0, None
    saved = None
    last = run_dir / "last.pt"
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        expected = dict(
            protocol=digest(c),
            manifest=manifest_hash,
            seed=seed,
            method=method,
            world_size=size,
            world_hash=world_hash,
            code=git_revision(),
        )
        for k, v in expected.items():
            if saved[k] != v:
                raise ValueError(f"Resume mismatch: {k}; start a separate campaign")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start, best, run_id = saved["step"], saved["best"], saved["wandb_id"]
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(
            "Nonempty run directory without checkpoint; do not overwrite it"
        )
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    barrier()
    module = (
        DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
        if size > 1
        else model
    )
    seed_all(seed + rank * 997)
    data = Segments(root, c, method, seed, steps * global_batch)
    data = Subset(data, range(start * global_batch, len(data)))
    sampler = (
        DistributedSampler(data, size, rank, shuffle=False, drop_last=True)
        if size > 1
        else None
    )
    loader = DataLoader(
        data,
        batch_size=micro,
        sampler=sampler,
        shuffle=False,
        drop_last=True,
        num_workers=tc["workers"],
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
        persistent_workers=tc["workers"] > 0,
    )
    if saved:
        restore_rng(saved["rng"][rank])
    iterator = iter(loader)
    run = None
    if rank == 0:
        import wandb

        run = wandb.init(
            project=c["wandb"]["project"],
            mode="offline" if allow_fixture else "online",
            name=f"{method}_{seed}",
            group=c["name"],
            dir=str(run_dir),
            id=run_id,
            resume="must" if run_id and not allow_fixture else None,
            config={
                **c,
                "method": method,
                "seed": seed,
                "code": git_revision(),
                "manifest": manifest_hash,
                "world_hash": world_hash,
                "parameters": sum(p.numel() for p in model.parameters()),
                "fixture": allow_fixture,
            },
        )
        run_id = run.id
        save_json(
            run_dir / "run.json",
            dict(
                wandb_url=run.url if not allow_fixture else None,
                id=run_id,
                method=method,
                seed=seed,
                protocol=digest(c),
                fixture=allow_fixture,
            ),
        )
    encoder, began = None, time.perf_counter()
    limit = tc.get("budget_seconds")
    ledger_path = run_dir / "compute_usage.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    used = max(float((saved or {}).get("training_seconds", 0.0)),
               float(ledger.get("training_seconds", 0.0)))
    validation_used = max(float((saved or {}).get("validation_seconds", 0.0)),
                          float(ledger.get("validation_seconds", 0.0)))
    validation_round = int((saved or {}).get("validation_round", 0))
    rounds = tc.get("validation_rounds", 4)
    step = start

    def max_rank_seconds(seconds):
        elapsed = torch.tensor(seconds, device=device, dtype=torch.float64)
        if size > 1:
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        return float(elapsed)

    def save_usage():
        if rank == 0:
            save_json(ledger_path, dict(
                step=step, training_seconds=used, validation_seconds=validation_used,
                training_gpu_hours=used * size / 3600,
                validation_gpu_hours=validation_used * size / 3600,
                training_budget_seconds=limit, gpu_count=size,
                budget_check="optimizer boundaries; any last-update overrun is reported",
            ))
    try:
        while step < steps:
            updated = not (limit and used >= limit)
            if updated:
                step += 1
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                update_started = time.perf_counter()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                progress = training_progress(step - 1, steps, used, limit)
                warmup = max(1, int(0.05 * steps))
                warmup_factor = max(progress, step / steps) / 0.05 if limit else step / warmup
                factor = min(1.0, warmup_factor) * (
                    0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr * factor
                metrics = {}
                for acc in range(accumulation):
                    batch = {
                        k: v.to(device, non_blocking=True)
                        for k, v in next(iterator).items()
                    }
                    sync = (
                        module.no_sync()
                        if size > 1 and acc + 1 < accumulation
                        else contextlib.nullcontext()
                    )
                    weight = 0.0
                    if method.endswith("_consistent"):
                        ramp = ((progress - 0.1) / 0.1) if limit else (
                            (step - tc["consistency_warmup"]) / max(tc["consistency_warmup"], 1)
                        )
                        weight = tc["consistency_weight"] * min(1.0, max(0.0, ramp))
                    with sync:
                        loss, parts = module(
                            batch,
                            world,
                            consistency_weight=weight,
                            consistency_batch=tc["consistency_batch"],
                            consistency_steps=tc["consistency_steps"],
                        )
                        if not torch.isfinite(loss):
                            raise FloatingPointError(
                                f"Nonfinite training loss at step {step}"
                            )
                        (loss / accumulation).backward()
                    for k, v in {"loss": loss.detach(), **parts}.items():
                        metrics[k] = metrics.get(k, 0.0) + float(v) / accumulation
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), tc["gradient_clip"], error_if_nonfinite=True
                )
                optimizer.step()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                used += max_rank_seconds(time.perf_counter() - update_started)
                save_usage()
            progress = training_progress(step, steps, used, limit)
            stop_now = progress >= 1.0
            if updated and (step % tc["log_every"] == 0 or step == 1 or stop_now):
                keys = sorted(metrics)
                vals = torch.tensor([metrics[k] for k in keys], device=device)
                if size > 1:
                    dist.all_reduce(vals)
                    vals /= size
                log = {f"train/{k}": float(v) for k, v in zip(keys, vals)}
                log.update(
                    step=step,
                    **{
                        "train/lr": lr * factor,
                        "train/grad_norm": float(norm),
                        "budget/training_seconds": used,
                        "budget/training_gpu_hours": used * size / 3600,
                        "budget/validation_gpu_hours": validation_used * size / 3600,
                        "budget/fraction": progress,
                        "budget/last_update_overrun_seconds": max(0.0, used - limit) if limit else 0.0,
                        "train/consistency_weight": weight,
                        "system/samples_per_second": (step - start)
                        * global_batch
                        / (time.perf_counter() - began),
                    },
                )
                if device.type == "cuda":
                    log["system/gpu_peak_memory_gib"] = (
                        torch.cuda.max_memory_allocated(device) / 2**30
                    )
                if rank == 0:
                    run.log(log, step=step)
                    with (run_dir / "metrics.jsonl").open("a") as f:
                        f.write(json.dumps(log, allow_nan=False) + "\n")
                    print(json.dumps(log), flush=True)
            evaluate_now = (progress >= (validation_round + 1) / rounds or stop_now) if limit else (
                step % tc["eval_every"] == 0 or step == steps
            )
            improved = False
            if evaluate_now:
                evaluation_started = time.perf_counter()
                rng = rng_state()
                validation_metrics = None
                if method == "world":
                    validation = world_validation(model, c, root, device, rank, size)
                    value = validation["dynamics_loss"]
                    improved = value < best
                    best = min(best, value)
                    if rank == 0:
                        validation_metrics = {
                            f"validation/{k}": v for k, v in validation.items()
                        }
                    if not allow_fixture and "cem_long" in c["methods"]:
                        if encoder is None:
                            encoder = Encoder(c["encoder"], root, device)
                        records, summary = evaluate(
                            None, model.world, c, root, "cem_long", seed, "validation",
                            encoder, device, tc["validation_episodes_per_task"],
                        )
                        if rank == 0:
                            save_json(run_dir / "validation" / f"cem_step_{step:07d}.json",
                                      dict(step=step, method="cem_long", metrics=summary, records=records))
                            validation_metrics.update({f"cem_validation/{k}": v for k, v in summary.items()})
                elif not allow_fixture:
                    if encoder is None:
                        encoder = Encoder(c["encoder"], root, device)
                    model.eval()
                    records, summary = evaluate(
                        model,
                        world,
                        c,
                        root,
                        method,
                        seed,
                        "validation",
                        encoder,
                        device,
                        tc["validation_episodes_per_task"],
                    )
                    value = summary["success_macro"]
                    improved = value > best
                    best = max(best, value)
                    if rank == 0:
                        save_json(
                            run_dir / "validation" / f"step_{step:07d}.json",
                            dict(step=step, metrics=summary, records=records),
                        )
                        validation_metrics = {
                            f"validation/{k}": v for k, v in summary.items()
                        }
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                validation_used += max_rank_seconds(time.perf_counter() - evaluation_started)
                validation_round += 1
                save_usage()
                if rank == 0 and validation_metrics is not None:
                    validation_metrics.update({
                        "budget/validation_round": validation_round,
                        "budget/training_gpu_hours": used * size / 3600,
                        "budget/validation_gpu_hours": validation_used * size / 3600,
                    })
                    run.log(validation_metrics, step=step)
                    log = dict(step=step, **validation_metrics)
                    with (run_dir / "metrics.jsonl").open("a") as f:
                        f.write(json.dumps(log, allow_nan=False) + "\n")
                    print(json.dumps(log), flush=True)
                restore_rng(rng)
            if step == 1 or step % tc["checkpoint_every"] == 0 or evaluate_now:
                rngs = gather(rng_state())
                if rank == 0:
                    saved = dict(
                        model=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        method=method,
                        seed=seed,
                        step=step,
                        best=best,
                        rng=rngs,
                        world_size=size,
                        protocol=digest(c),
                        manifest=manifest_hash,
                        world_hash=world_hash,
                        code=git_revision(),
                        wandb_id=run_id,
                        fixture=allow_fixture,
                        training_seconds=used,
                        validation_seconds=validation_used,
                        validation_round=validation_round,
                    )
                    checkpoint(last, saved)
                    if improved:
                        checkpoint(run_dir / "best.pt", saved)
                barrier()
            if stop_now:
                break
        if rank == 0:
            save_json(
                run_dir / "complete.json",
                dict(step=step, best=best, fixture=allow_fixture,
                     stop_reason="compute_budget" if limit and used >= limit else "update_limit",
                     training_seconds=used, validation_seconds=validation_used,
                     training_gpu_hours=used * size / 3600,
                     training_budget_seconds=limit, validation_rounds=validation_round),
            )
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/flow_metaworld.json")
    p.add_argument("--root", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--world")
    p.add_argument("--allow-fixture", action="store_true")
    a = p.parse_args()
    train(
        config(a.config),
        a.root,
        a.run_dir,
        a.method,
        a.seed,
        a.world,
        allow_fixture=a.allow_fixture,
    )
