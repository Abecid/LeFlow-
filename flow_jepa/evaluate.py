from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from .common import config, digest, distributed, file_hash, gather, save_json, seed_all
from .environment import make_env
from .models import System, distance
from .vision import Encoder, history_clip


def cem(
    dynamics,
    start,
    goal,
    horizon,
    action_dim,
    candidates,
    elites,
    iterations,
    mean=None,
):
    if not 1 <= elites <= candidates or min(horizon, iterations) < 1:
        raise ValueError("Invalid CEM settings")
    mean = start.new_zeros(horizon, action_dim) if mean is None else mean.clone()
    std, best, best_cost = torch.ones_like(mean), mean, float("inf")
    for _ in range(iterations):
        a = (
            mean[None]
            + std[None]
            * torch.randn(candidates, horizon, action_dim, device=start.device)
        ).clamp(-1, 1)
        a[0] = mean.clamp(-1, 1)
        pred = dynamics.rollout(start.expand(candidates, -1, -1), a)
        costs = distance(pred, goal[:, None]).min(1).values
        idx = costs.topk(elites, largest=False).indices
        if float(costs[idx[0]]) < best_cost:
            best, best_cost = a[idx[0]].clone(), float(costs[idx[0]])
        mean, std = a[idx].mean(0), a[idx].std(0, unbiased=False).clamp_min(0.05)
    return best, best_cost


class Controller:
    def __init__(self, model, world, c, method):
        self.model, self.world, self.c, self.method = model, world, c, method

    @torch.no_grad()
    def plan(self, start, goal):
        c, e = self.c["model"], self.c["evaluation"]
        k, m = c["chunk_steps"], c["segments"]
        ad = 4 * self.c["data"]["action_repeat"]
        if self.method.startswith("cem_"):
            h = e["short_horizon"] if self.method == "cem_short" else e["long_horizon"]
            a, score = cem(
                self.world,
                start,
                goal,
                h,
                ad,
                e["cem_candidates"],
                e["cem_elites"],
                e["cem_iterations"],
            )
            return a[0], goal[0], score, h
        if self.method == "hwm_adapted":
            macro, score = cem(
                self.model.coarse,
                start,
                goal,
                m,
                c["macro_dim"],
                e["cem_candidates"],
                e["cem_elites"],
                e["cem_iterations"],
            )
            subgoal = self.model.coarse(start, macro[:1])
            warm = None
        else:
            # Identical deterministic proposals are not counted as diverse candidates.
            det = "deterministic" in self.method
            n = 1 if det else e["candidates"]
            astart, agoal = start.expand(n, -1, -1), goal.expand(n, -1, -1)
            path_only = self.method == "leflow_adapted"
            interior, acts = self.model.planner.sample(
                astart,
                agoal,
                steps=e["flow_steps"],
                deterministic=det,
                path_only=path_only,
            )
            paths = torch.cat([astart[:, None], interior, agoal[:, None]], 1)
            if path_only:
                acts = self.model.inverse(
                    paths[:, :-1].flatten(0, 1), paths[:, 1:].flatten(0, 1)
                ).reshape(n, m, k, ad)
                # LeFlow family: rank actual decoded actions by final rollout distance.
                endpoint = self.world.rollout(
                    astart, acts.clamp(-1, 1).reshape(n, m * k, ad)
                )[:, -1]
                scores = distance(endpoint, agoal)
            else:
                endpoints = self.world.rollout(
                    paths[:, :-1].flatten(0, 1), acts.clamp(-1, 1).reshape(n * m, k, ad)
                )[:, -1]
                scores = (
                    distance(endpoints, paths[:, 1:].flatten(0, 1))
                    .reshape(n, m)
                    .mean(1)
                )
            index = int(scores.argmin())
            subgoal, warm, score = (
                paths[index : index + 1, 1],
                acts[index, 0].clamp(-1, 1),
                float(scores[index]),
            )
        a, _ = cem(
            self.world,
            start,
            subgoal,
            k,
            ad,
            e["refine_candidates"],
            e["refine_elites"],
            e["refine_iterations"],
            mean=warm,
        )
        return a[0], subgoal[0], score, k


def load_models(c, world_path, method, checkpoint_path, device):
    base = torch.load(world_path, map_location="cpu", weights_only=False)
    if base["protocol"] != digest(c) or base["method"] != "world":
        raise ValueError("World checkpoint/protocol mismatch")
    system = System(c, "world")
    system.load_state_dict(base["model"])
    world = system.world.eval().requires_grad_(False).to(device)
    if method.startswith("cem_"):
        return None, world, base
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        saved["protocol"] != digest(c)
        or saved["method"] != method
        or saved["manifest"] != base["manifest"]
    ):
        raise ValueError("Planner checkpoint/protocol/data mismatch")
    if saved["world_hash"] != file_hash(world_path):
        raise ValueError("Planner was trained against a different world checkpoint")
    model = System(c, method).to(device)
    model.load_state_dict(saved["model"])
    return model.eval(), world, saved


def summarize(records, c):
    tasks = sorted({r["task"] for r in records})
    result = {}
    for task in tasks:
        rows = [r for r in records if r["task"] == task]
        s = np.array([r["success"] for r in rows], dtype=float)
        n, p = len(s), float(s.mean())
        denom = 1 + 1.96**2 / n
        center = (p + 1.96**2 / (2 * n)) / denom
        half = 1.96 * np.sqrt(p * (1 - p) / n + 1.96**2 / (4 * n * n)) / denom
        result[f"task/{task}/success"] = p
        result[f"task/{task}/ci95_low"] = float(center - half)
        result[f"task/{task}/ci95_high"] = float(center + half)
    result["success_macro"] = float(
        np.mean([result[f"task/{t}/success"] for t in tasks])
    )
    for label, group in [
        ("seen", c["training_tasks"]),
        ("heldout", c["heldout_tasks"]),
    ]:
        values = [result[f"task/{t}/success"] for t in tasks if t in group]
        if values:
            result[f"{label}_success_macro"] = float(np.mean(values))
    result["episodes"] = len(records)
    result["expert_goal_failure_rate"] = float(
        np.mean([not r["expert_goal_success"] for r in records])
    )
    result["latency_ms_mean"] = float(np.mean([r["latency_ms_mean"] for r in records]))
    timings = [x for r in records for x in r["controller_latency_ms"]]
    result["controller_step_latency_ms_mean"] = float(np.mean(timings))
    result["controller_step_latency_ms_p95"] = float(np.quantile(timings, 0.95))
    result["return_mean"] = float(np.mean([r["return"] for r in records]))
    result["world_predictions_mean"] = float(
        np.mean([r["world_predictions"] for r in records])
    )
    attained = [x for r in records for x in r["observed_subgoal_cosine"]]
    if attained:
        result["observed_subgoal_cosine"] = float(np.mean(attained))
    return result


def evaluate(model, world, c, root, method, seed, split, encoder, device, count=None):
    rank, size, _ = distributed()
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["protocol"] != digest(c) or manifest["encoder"] != encoder.fingerprint:
        raise ValueError("Evaluation encoder/data/protocol mismatch")
    records, controller = [], Controller(model, world, c, method)
    mean, std = (
        torch.tensor(manifest["mean"], device=device),
        torch.tensor(manifest["std"], device=device),
    )
    rows = [
        r
        for r in manifest["entries"]
        if r["split"] == split and (count is None or r["index"] < count)
    ]
    if not rows:
        raise ValueError("No evaluation episodes")
    expected_tasks = c["training_tasks"] + (
        c["heldout_tasks"] if split == "test" else []
    )
    expected_count = count or c["data"][split + "_episodes_per_task"]
    if len(rows) != len(expected_tasks) * expected_count:
        raise ValueError("Incomplete evaluation suite")
    for i in range(rank, len(rows), size):
        row = rows[i]
        # Per-episode randomness pairs evaluation independently of scheduling/order.
        seed_all((seed + row["seed"]) % 2**32)
        with h5py.File(root / row["path"], "r") as f:
            goal = (
                torch.tensor(f["goal"][:], device=device, dtype=torch.float32) - mean
            ) / std
            initial = f["initial_rgb"][:]
        env = make_env(row["task"], row["seed"], c["data"])
        try:
            env.reset()
            frames = [env.render().copy()]
            if not np.array_equal(frames[0], initial):
                raise ValueError(
                    "Reset RGB differs from registered episode; environment/rendering changed"
                )
            success, first, total_reward = False, None, 0.0
            latency, scores, observed, pending = [], [], [], []
            initial_calls = world.calls
            initial_coarse = model.coarse.calls if method == "hwm_adapted" else 0
            repeat, budget = (
                c["data"]["action_repeat"],
                c["evaluation"]["budget_primitive"],
            )
            for t in range(budget // repeat):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                clip = history_clip(
                    frames, len(frames) - 1, c["data"]["history_frames"]
                )
                z = (encoder(clip[None]).float() - mean) / std
                for due, target in pending:
                    if due == t:
                        observed.append(float(distance(z, target[None])))
                pending = [(due, target) for due, target in pending if due > t]
                action, subgoal, score, h = controller.plan(z, goal[None])
                pending.append((t + h, subgoal.detach()))
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                latency.append(1000 * (time.perf_counter() - started))
                scores.append(score)
                stop = False
                for j, a in enumerate(action.cpu().numpy().reshape(repeat, 4)):
                    _, reward, terminated, truncated, info = env.step(np.clip(a, -1, 1))
                    total_reward += float(reward)
                    frames.append(env.render().copy())
                    if info["success"] and first is None:
                        first, success = t * repeat + j + 1, True
                    if success or terminated or truncated:
                        stop = True
                        break
                if stop:
                    break
            records.append(
                dict(
                    id=row["id"],
                    task=row["task"],
                    reset_seed=row["seed"],
                    model_seed=seed,
                    success=success,
                    first_success_primitive=first,
                    steps=len(frames) - 1,
                    expert_goal_success=row["expert_success"],
                    episode_sha256=row["sha256"],
                    return_=total_reward,
                    latency_ms_mean=float(np.mean(latency)),
                    latency_ms_p95=float(np.quantile(latency, 0.95)),
                    controller_latency_ms=latency,
                    world_predictions=world.calls - initial_calls,
                    coarse_predictions=(model.coarse.calls - initial_coarse)
                    if method == "hwm_adapted"
                    else 0,
                    predicted_plan_cost=float(np.mean(scores)),
                    observed_subgoal_cosine=observed,
                )
            )
            records[-1]["return"] = records[-1].pop("return_")
            print(
                json.dumps(
                    dict(
                        event="evaluation_episode",
                        method=method,
                        seed=seed,
                        split=split,
                        episode=row["id"],
                        success=success,
                    )
                ),
                flush=True,
            )
        finally:
            env.close()
    records = sorted(
        [r for part in gather(records) for r in part], key=lambda r: r["id"]
    )
    return records, summarize(records, c)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/flow_metaworld.json")
    p.add_argument("--root", required=True)
    p.add_argument("--world", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--method", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    c = config(a.config)
    rank, _, device = distributed()
    model, world, saved = load_models(c, a.world, a.method, a.checkpoint, device)
    if saved["seed"] != a.seed:
        raise ValueError("Evaluation seed differs from checkpoint training seed")
    if saved.get("fixture", False):
        raise ValueError("Fixture checkpoints cannot enter benchmark evaluation")
    encoder = Encoder(c["encoder"], a.root, device)
    records, metrics = evaluate(
        model, world, c, a.root, a.method, a.seed, a.split, encoder, device
    )
    if rank == 0:
        import wandb

        run = wandb.init(
            project=c["wandb"]["project"],
            mode="online",
            group=c["name"],
            job_type="test" if a.split == "test" else "validation",
            name=f"{a.method}_{a.seed}_{a.split}",
            config=c,
        )
        run.log({f"{a.split}/{k}": v for k, v in metrics.items()})
        report = dict(
            protocol=digest(c),
            manifest=file_hash(Path(a.root) / "manifest.json"),
            world_hash=file_hash(a.world),
            checkpoint_hash=file_hash(a.checkpoint) if a.checkpoint else None,
            method=a.method,
            seed=a.seed,
            split=a.split,
            fixture=False,
            code=saved["code"],
            metrics=metrics,
            records=records,
            wandb_url=run.url,
            hardware=torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "CPU",
        )
        save_json(a.output, report)
        columns = [
            "id",
            "task",
            "success",
            "first_success_primitive",
            "latency_ms_mean",
            "return",
        ]
        run.log(
            {
                "episodes": wandb.Table(
                    columns=columns, data=[[r[k] for k in columns] for r in records]
                )
            }
        )
        run.finish()


if __name__ == "__main__":
    main()
