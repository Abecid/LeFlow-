"""Closed-loop task evaluation on validation or final held-out test episodes."""

import argparse
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch
import stable_worldmodel as swm
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler

from btm_jepa.data import atomic_json, image_transform, file_sha256
from btm_jepa.runtime import SubgoalSolver
from btm_jepa.env import World


def wilson(successes, n):
    p, z = successes / n, 1.96
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0, center - half), min(1, center + half)


def run(args):
    checkpoint = Path(args.checkpoint).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = payload["data"]
    # Allow moving data/checkpoints between machines without rewriting model state.
    source = Path(args.source or meta["source"]["path"])
    lewm = args.lewm_checkpoint or meta["lewm_checkpoint"]
    if file_sha256(lewm) != meta["checkpoint_sha256"]:
        raise ValueError(
            "Evaluation encoder checkpoint differs from the training cache"
        )
    if args.split not in {"val", "test"}:
        raise ValueError("Evaluation must use val or test")
    eligible = [
        ep
        for ep in meta["split"][args.split]
        if meta["source"]["lengths"][ep] > args.goal_offset
    ]
    if len(eligible) < args.episodes:
        raise ValueError(
            f"Need {args.episodes} distinct {args.split} episodes longer than {args.goal_offset}; found {len(eligible)}"
        )
    rng = np.random.default_rng(args.seed)
    episodes = rng.choice(eligible, args.episodes, replace=False).tolist()
    starts = [
        int(rng.integers(meta["source"]["lengths"][ep] - args.goal_offset))
        for ep in episodes
    ]
    cfg = OmegaConf.load(
        Path(__file__).parent / "config/eval" / f"{payload['config']['task']}.yaml"
    )
    cfg.world.num_envs = args.episodes
    cfg.world.max_episode_steps = max(2 * args.budget, args.goal_offset + 1)
    world = World(**cfg.world, image_shape=(224, 224))
    horizon = (
        payload["config"]["data"]["horizon"] if args.mode == "flat" else args.spacing
    )
    block = payload["config"]["data"]["action_block"]
    solver = SubgoalSolver(
        checkpoint,
        device=args.device,
        mode=args.mode,
        num_samples=args.candidates,
        flow_steps=args.flow_steps,
        spacing=args.spacing,
        horizon=payload["config"]["data"]["horizon"],
        cem_candidates=args.cem_candidates,
        cem_iterations=args.cem_iterations,
        cem_elites=args.cem_elites,
        seed=args.seed,
        lewm_checkpoint=lewm,
    )
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_ = (
        np.asarray(meta["source"]["action_mean"]),
        np.asarray(meta["source"]["action_std"]),
    )
    scaler.var_, scaler.n_features_in_ = scaler.scale_**2, len(scaler.mean_)
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(
            horizon=horizon, receding_horizon=args.execute_blocks, action_block=block
        ),
        process={"action": scaler},
        transform={"pixels": image_transform(), "goal": image_transform()},
    )
    dataset = swm.data.HDF5Dataset(
        name=str(source.resolve().with_suffix("")), frameskip=1, num_steps=1
    )
    if dataset.lengths.tolist() != meta["source"]["lengths"]:
        raise ValueError(
            "Evaluation dataset episode lengths differ from the training source"
        )
    callables = OmegaConf.to_container(cfg.eval.callables, resolve=True)
    for spec in callables:
        if not hasattr(world.envs.unwrapped.envs[0].unwrapped, spec["method"]):
            raise ValueError(
                f"Environment lacks required reset method {spec['method']}"
            )
        for item in spec["args"].values():
            if item.get("in_dataset", True):
                key = item["value"].removeprefix("goal_")
                if key not in dataset.column_names:
                    raise ValueError(
                        f"Evaluation dataset lacks required reset column {key}"
                    )
    world.set_policy(policy)
    started = time.perf_counter()
    try:
        result = world.evaluate_from_dataset(
            dataset,
            episodes_idx=episodes,
            start_steps=starts,
            goal_offset_steps=args.goal_offset,
            eval_budget=args.budget,
            callables=callables,
            save_video=args.save_video,
            video_path=Path(args.output).parent / (Path(args.output).stem + "_videos"),
        )
    finally:
        world.envs.close()
    successes = int(np.asarray(result["episode_successes"]).sum())
    lower, upper = wilson(successes, args.episodes)
    metrics = dict(
        success_rate=successes / args.episodes,
        successes=successes,
        episodes=args.episodes,
        success_ci95_low=lower,
        success_ci95_high=upper,
        evaluation_seconds=time.perf_counter() - started,
        **solver.diagnostics(),
    )
    out = dict(
        metrics=metrics,
        protocol=dict(
            task=payload["config"]["task"],
            split=args.split,
            seed=args.seed,
            goal_offset=args.goal_offset,
            budget=args.budget,
            mode=args.mode,
            spacing=args.spacing,
            candidates=args.candidates,
            flow_steps=args.flow_steps,
            execute_blocks=args.execute_blocks,
            episode_indices=episodes,
            start_steps=starts,
            checkpoint_sha256=file_sha256(checkpoint),
            manifest_sha256=payload["manifest_sha256"],
        ),
        test_fixture=meta.get("test_fixture", False),
        episode_successes=np.asarray(result["episode_successes"]).astype(bool).tolist(),
    )
    atomic_json(args.output, out)
    print(json.dumps(out, indent=2), flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source")
    p.add_argument("--lewm-checkpoint")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--goal-offset", type=int, default=50)
    p.add_argument("--budget", type=int, default=200)
    p.add_argument("--mode", choices=["hierarchical", "flat"], default="hierarchical")
    p.add_argument("--spacing", type=int, default=2)
    p.add_argument("--candidates", type=int, default=64)
    p.add_argument("--flow-steps", type=int, default=16)
    p.add_argument("--cem-candidates", type=int, default=32)
    p.add_argument("--cem-iterations", type=int, default=3)
    p.add_argument("--cem-elites", type=int, default=8)
    p.add_argument("--execute-blocks", type=int, default=1)
    p.add_argument("--save-video", action="store_true")
    args = p.parse_args()
    if min(args.episodes, args.goal_offset, args.budget, args.execute_blocks) < 1:
        p.error("episodes, offsets, budget and execution length must be positive")
    run(args)


if __name__ == "__main__":
    main()
