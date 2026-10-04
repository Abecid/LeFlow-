"""torchrun-compatible FM/BTM planner training on frozen, cached JEPA latents."""
from __future__ import annotations

import contextlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from btm_jepa.data import LatentSegments, atomic_json, file_sha256
from btm_jepa.distributed import EvalShard, initialize
from btm_jepa.models import PathModel, PlannerTrainingModel, sample_paths
from btm_jepa.runtime import atomic_checkpoint


def main_process_call(fn, rank):
    """Propagate rank-zero I/O failures instead of leaving peers at a barrier."""
    error, result = None, None
    if rank == 0:
        try:
            result = fn()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    if dist.is_initialized():
        obj = [error]
        dist.broadcast_object_list(obj, src=0)
        error = obj[0]
    if error:
        raise RuntimeError(error)
    return result


def reduce_metrics(sums, count, device):
    keys = sorted(sums)
    values = torch.tensor([sums[k] for k in keys] + [count], device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(values)
    if values[-1] == 0:
        raise ValueError("Metric accumulator has no samples")
    return {k: float(values[i]/values[-1]) for i,k in enumerate(keys)}


def random_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def restore_random(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}


def loss_for(model, batch, cfg, dynamics):
    out = model(batch, boundary_weight=cfg.loss.boundary_weight, inverse_weight=cfg.loss.inverse_weight)
    if dynamics is not None:
        from latent_planner import lewm_consistency_loss
        consistency = lewm_consistency_loss(dynamics, batch["local_z"], out["pred_actions"], history_size=3)
        out["loss"] = out["loss"] + cfg.loss.consistency_weight * consistency
        out["consistency_loss"] = consistency.detach()
    return {k: v for k,v in out.items() if k != "pred_actions"}


def validation(model, loader, cfg, device, dynamics, rank):
    model.eval()
    keys = ["loss", "generative_loss", "inverse_loss"]
    if model.path.method == "btm":
        keys += ["transport_loss", "boundary_loss", "jvp_rms"]
    if dynamics is not None:
        keys += ["consistency_loss"]
    sums, count = dict.fromkeys(keys, 0.0), 0
    # Validation cannot perturb training noise, and repeated validation is comparable.
    with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
        torch.manual_seed(cfg.seed + 10000 + rank)
        for batch in loader:
            batch = move_batch(batch, device)
            # no_grad still allows forward-mode AD for the BTM validation loss.
            with torch.no_grad():
                out = loss_for(model, batch, cfg, dynamics)
            n = len(batch["z_path"])
            count += n
            for k,v in out.items():
                sums[k] += float(v) * n
    result = reduce_metrics(sums, count, device)
    model.train()
    return {"val/"+k: v for k,v in result.items()}


@torch.no_grad()
def generation_validation(model, dataset, cfg, device):
    """Single-sample accuracy + diversity, not oracle best-of-N success."""
    model.eval()
    results = dict(path_mse_single=0.0, candidate_variance=0.0, boundary_mse=0.0)
    n = min(len(dataset), cfg.val_generation_samples)
    gen = torch.Generator(device=device).manual_seed(cfg.seed+20000)
    for i in range(n):
        b = dataset[i]
        z = b["z_path"].to(device)[None]
        samples = sample_paths(model.path, z[:, 0], z[:, -1], horizon=cfg.data.horizon,
                               spacing=float(b["spacing"]), num_samples=cfg.val_candidates,
                               flow_steps=cfg.evaluation.flow_steps, generator=gen)
        results["path_mse_single"] += float((samples[:, 0, 1:-1] - z[:, 1:-1]).square().mean())
        results["candidate_variance"] += float(samples[:, :, 1:-1].var(1, unbiased=False).mean())
        # Endpoint clamping is an invariant check, explicitly not a prediction metric.
        results["boundary_mse"] += float((samples[:, :, -1]-z[:, None, -1]).square().mean())
    model.train()
    return {"val/"+k: v/max(n,1) for k,v in results.items()}


def closed_loop(cfg, checkpoint, step, run_dir):
    metrics = {}
    env = os.environ.copy()
    for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(k, None)
    for offset in cfg.evaluation.goal_offsets:
        output = run_dir / "eval" / f"step_{step:08d}_offset_{offset}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        args = [sys.executable, str(Path(__file__).parent / "eval_subgoals.py"),
                "--checkpoint", str(checkpoint), "--output", str(output),
                "--episodes", str(cfg.evaluation.episodes), "--split", cfg.evaluation.split,
                "--goal-offset", str(offset), "--budget", str(cfg.evaluation.budget),
                "--mode", cfg.evaluation.mode, "--spacing", str(cfg.evaluation.spacing),
                "--candidates", str(cfg.evaluation.candidates), "--flow-steps", str(cfg.evaluation.flow_steps),
                "--cem-candidates", str(cfg.evaluation.cem_candidates),
                "--cem-iterations", str(cfg.evaluation.cem_iterations), "--cem-elites", str(cfg.evaluation.cem_elites),
                "--device", cfg.device]
        with output.with_suffix(".log").open("w") as log:
            subprocess.run(args, check=True, stdout=log, stderr=subprocess.STDOUT, env=env,
                           timeout=cfg.evaluation.timeout_seconds)
        result = json.loads(output.read_text())
        for key, value in result["metrics"].items():
            if isinstance(value, (float, int)):
                metrics[f"eval/offset_{offset}/{key}"] = value
    metrics["eval/mean_success_rate"] = float(np.mean([metrics[f"eval/offset_{o}/success_rate"] for o in cfg.evaluation.goal_offsets]))
    return metrics


@hydra.main(version_base=None, config_path="config/train", config_name="subgoals")
def run(cfg):
    rank, world, device = initialize(cfg.device)
    if cfg.precision != "fp32":
        raise ValueError("Use precision=fp32: the BTM JVP and stopped-target update are validated in float32")
    if cfg.global_batch_size % (cfg.micro_batch_size*world):
        raise ValueError("global_batch_size must be divisible by micro_batch_size * WORLD_SIZE")
    accumulation = cfg.global_batch_size // (cfg.micro_batch_size*world)
    if accumulation < 1:
        raise ValueError("global batch is smaller than one distributed microbatch")
    if cfg.evaluation.split != "val":
        raise ValueError("Intermediate evaluation must use val; reserve test for final evaluation")
    random.seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)
    torch.manual_seed(cfg.seed + rank)
    # Each DDP replica will be synchronized by DDP after construction.
    data_args = OmegaConf.to_container(cfg.data, resolve=True)
    train_set = LatentSegments(split="train", **data_args)
    val_set = LatentSegments(split="val", **data_args)
    meta = train_set.meta
    if file_sha256(meta["lewm_checkpoint"]) != meta["checkpoint_sha256"]:
        raise ValueError("LeWM checkpoint no longer matches the cached latent encoder")
    manifest_hash = file_sha256(cfg.data.manifest)
    sampler = DistributedSampler(train_set, num_replicas=world, rank=rank, seed=cfg.seed, drop_last=True)
    loader_gen = torch.Generator().manual_seed(cfg.seed+rank)
    loader_args = dict(batch_size=cfg.micro_batch_size, num_workers=cfg.num_workers,
                       pin_memory=device.type=="cuda", persistent_workers=cfg.num_workers > 0)
    loader = DataLoader(train_set, sampler=sampler, drop_last=True, generator=loader_gen, **loader_args)
    val_subset = Subset(val_set, range(min(len(val_set), cfg.val_max_samples)))
    val_loader = DataLoader(val_subset, sampler=EvalShard(len(val_subset), rank, world), **loader_args)
    usable_batches = len(loader)//accumulation * accumulation
    if usable_batches == 0:
        raise ValueError("Not enough training clips for one global batch; reduce global_batch_size")
    model = PlannerTrainingModel(PathModel(latent_dim=meta["latent_dim"], method=cfg.method,
                                          **{k: v for k,v in cfg.model.items() if k != "inverse_hidden"}),
                                 action_dim=meta["source"]["action_dim"]*cfg.data.action_block,
                                 inverse_hidden=cfg.model.inverse_hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    dynamics = None
    if cfg.loss.consistency_weight:
        from latent_planner import load_lewm
        dynamics = load_lewm(meta["lewm_checkpoint"]).to(device).eval().requires_grad_(False)
    run_dir = Path(cfg.run_dir).expanduser().resolve()
    start_epoch, next_batch, step, best = 0, 0, 0, -float("inf")
    payload = None
    if cfg.resume:
        payload = torch.load(cfg.resume, map_location=device, weights_only=False)
        if payload["manifest_sha256"] != manifest_hash:
            raise ValueError("Resume cache/split differs from the original run")
        if payload["world_size"] != world or payload["config"]["global_batch_size"] != cfg.global_batch_size:
            raise ValueError("Exact resume requires the same world size and global batch")
        if payload["arch"]["path"] != model.path.arch:
            raise ValueError("Resume architecture/method differs")
        for field in ("data", "loss", "micro_batch_size"):
            current = OmegaConf.to_container(cfg[field], resolve=True) if OmegaConf.is_config(cfg[field]) else cfg[field]
            if payload["config"][field] != current:
                raise ValueError(f"Resume changed {field}")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch, next_batch, step = payload["epoch"], payload["next_batch"], payload["step"]
        best = payload.get("best_success", best)
    wrapped = DDP(model, device_ids=[device.index] if device.type == "cuda" else None) if world>1 else model
    if payload is not None:
        restore_random(payload["rng_states"][rank], device)
    config = OmegaConf.to_container(cfg, resolve=True)
    wandb_run = None

    def prepare_run():
        nonlocal wandb_run
        if (run_dir / "last.pt").exists() and not cfg.resume:
            raise FileExistsError(f"Run already exists: {run_dir}; set resume=.../last.pt or a new run_dir")
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(run_dir / "config.json", config)
        provenance = dict(manifest_sha256=manifest_hash, world_size=world, global_batch_size=cfg.global_batch_size,
                          micro_batch_size=cfg.micro_batch_size, accumulation=accumulation,
                          train_clips=len(train_set), val_clips=len(val_set),
                          parameters=sum(p.numel() for p in model.parameters()))
        try:
            provenance["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
            provenance["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=Path(__file__).parent, text=True).strip())
        except subprocess.CalledProcessError:
            provenance["git_commit"] = "unavailable"
        atomic_json(run_dir / "provenance.json", provenance)
        if cfg.wandb.enabled:
            import wandb
            run_id = payload.get("wandb_id") if payload else cfg.wandb.id
            wandb_run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity,
                                   name=cfg.wandb.name, group=cfg.wandb.group, mode=cfg.wandb.mode,
                                   id=run_id, resume="must" if run_id and cfg.resume else "allow",
                                   dir=str(run_dir), config={**config, "runtime": provenance})
            if cfg.wandb.mode == "online" and (wandb_run is None or wandb_run.offline):
                raise RuntimeError("Online W&B requested but not established; run wandb login on the training machine")
            wandb.define_metric("step")
            wandb.define_metric("*", step_metric="step")
        print(json.dumps(provenance), flush=True)
    main_process_call(prepare_run, rank)

    def log(metrics):
        if rank == 0:
            row = {"step": step, **metrics}
            with (run_dir / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False)+"\n")
            if wandb_run is not None:
                wandb_run.log(row)
            print(json.dumps(row), flush=True)

    def save(epoch, batch):
        rng = random_state(device)
        rngs = [None]*world
        if world > 1:
            dist.all_gather_object(rngs, rng)
        else:
            rngs[0] = rng
        ckpt = dict(format_version=1, arch={"path":model.path.arch, "inverse":model.inverse_arch},
                    model=model.state_dict(), optimizer=optimizer.state_dict(), config=config,
                    data=meta, manifest_sha256=manifest_hash, epoch=epoch, next_batch=batch, step=step,
                    world_size=world, rng_states=rngs, best_success=best,
                    wandb_id=wandb_run.id if wandb_run else None)
        main_process_call(lambda: atomic_checkpoint(run_dir / "last.pt", ckpt), rank)

    def evaluate():
        nonlocal best
        metrics = main_process_call(lambda: closed_loop(cfg, run_dir / "last.pt", step, run_dir), rank)
        if rank == 0:
            log(metrics)
            if metrics["eval/mean_success_rate"] > best:
                best = metrics["eval/mean_success_rate"]
                ckpt = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
                ckpt["best_success"] = best
                atomic_checkpoint(run_dir / "best.pt", ckpt)
        if world > 1:
            obj = [best]
            dist.broadcast_object_list(obj, src=0)
            best = obj[0]

    last_eval_step = -1
    started, last_log_step = time.perf_counter(), step
    try:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        final_epoch, final_batch = start_epoch, next_batch
        stop = bool(cfg.max_steps is not None and step >= cfg.max_steps)
        for epoch in range(start_epoch, cfg.epochs):
            if stop:
                break
            sampler.set_epoch(epoch)
            loader_gen.manual_seed(cfg.seed + rank + epoch*1000)
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= usable_batches:
                    break
                if epoch == start_epoch and batch_idx < next_batch:
                    continue
                batch = move_batch(batch, device)
                sync = (batch_idx+1) % accumulation == 0
                ctx = wrapped.no_sync() if world>1 and not sync else contextlib.nullcontext()
                with ctx:
                    out = loss_for(wrapped, batch, cfg, dynamics)
                    finite = torch.isfinite(out["loss"]).to(torch.int32)
                    if world>1:
                        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    if not finite:
                        raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, batch {batch_idx}")
                    (out["loss"]/accumulation).backward()
                if not sync:
                    continue
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                final_epoch, final_batch = epoch, batch_idx+1
                if final_batch == usable_batches:
                    final_epoch, final_batch = epoch+1, 0
                if step % cfg.log_every == 0 or step == 1:
                    metrics = reduce_metrics({k:float(v.detach())*len(batch["z_path"]) for k,v in out.items()}, len(batch["z_path"]), device)
                    elapsed = time.perf_counter()-started
                    log({"train/"+k:v for k,v in metrics.items()} | {
                        "train/grad_norm":float(grad), "train/lr":optimizer.param_groups[0]["lr"],
                        "train/epoch":epoch+1, "system/samples_per_second":(step-last_log_step)*cfg.global_batch_size/max(elapsed,1e-6),
                        "system/gpu_peak_memory_gib":torch.cuda.max_memory_allocated(device)/1024**3 if device.type=="cuda" else 0})
                    started, last_log_step = time.perf_counter(), step
                if step % cfg.validate_every == 0:
                    vm = validation(model, val_loader, cfg, device, dynamics, rank)
                    gm = main_process_call(lambda: generation_validation(model, val_set, cfg, device), rank)
                    log(vm | (gm or {}))
                do_eval = cfg.evaluation.enabled and step % cfg.evaluation.every_steps == 0
                if step % cfg.checkpoint_every == 0 or do_eval:
                    save(final_epoch, final_batch)
                if do_eval:
                    evaluate()
                    last_eval_step = step
                if cfg.max_steps is not None and step >= cfg.max_steps:
                    stop = True
                    break
            save(final_epoch, final_batch)
            if stop:
                break
        vm = validation(model, val_loader, cfg, device, dynamics, rank)
        gm = main_process_call(lambda: generation_validation(model, val_set, cfg, device), rank)
        log(vm | (gm or {}))
        save(final_epoch, final_batch)
        if cfg.evaluation.enabled and cfg.evaluation.at_end and last_eval_step != step:
            evaluate()
        save(final_epoch, final_batch)
        if rank == 0 and wandb_run is not None and cfg.wandb.upload_checkpoints:
            import wandb
            artifact = wandb.Artifact(f"{cfg.task}-{cfg.method}-planner", type="model")
            artifact.add_file(str(run_dir / "last.pt"))
            wandb_run.log_artifact(artifact)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    run()
