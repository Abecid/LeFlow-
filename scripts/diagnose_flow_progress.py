#!/usr/bin/env python3
"""Bounded validation-only candidate calibration using frozen legacy checkpoints.

No optimizer, checkpoint conversion, training, test outcomes, or hyperparameter
search. Run from the frozen execution checkout; load proposed scoring explicitly
from the reporting revision. Outputs describe diagnostics, never benchmark scores.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch


def ranks(x):
    x = np.asarray(x)
    order = np.argsort(x, kind="stable")
    out = np.empty(len(x), float)
    for value in np.unique(x):
        locations = np.flatnonzero(x[order] == value)
        out[order[locations]] = float(locations.mean())
    return out


def spearman(x, y):
    x, y = ranks(x), ranks(y)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--execution-repo", required=True)
    p.add_argument("--scoring-file", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    sys.path.insert(0, args.execution_repo)
    from flow_jepa.common import distributed, file_hash, save_json, seed_all
    from flow_jepa.models import distance
    from flow_jepa.evaluate import load_models, cem
    from flow_jepa.environment import make_env
    from flow_jepa.vision import Encoder, history_clip
    from flow_jepa.budget import PlanningBudget
    spec = importlib.util.spec_from_file_location("flow_jepa.diagnostic_scoring", args.scoring_file)
    scoring = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scoring)

    rank, size, device = distributed()
    assert size <= 4 and device.type == "cuda"
    torch.set_num_threads(2)
    root, output = Path(args.root), Path(args.output)
    c = json.loads((Path(args.execution_repo) / "config/flow_metaworld.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    rows = sorted([r for r in manifest["entries"] if r["split"] == "validation" and r["index"] == 0],
                  key=lambda r: r["task"])
    assert len(rows) == 13 and {r['task'] for r in rows} == set(c['training_tasks'])
    assert all((root / "runs" / f"{method}_3072" / "complete.json").exists()
               for method in ("world", "joint_flow_consistent", "leflow_adapted", "hwm_adapted"))
    assert (root / "post-training-hold.json").exists()
    started = time.perf_counter()
    deadline = PlanningBudget(600)
    encoder = Encoder(c["encoder"], root, device)
    mean = torch.tensor(manifest["mean"], device=device)
    std = torch.tensor(manifest["std"], device=device)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    methods = ["joint_flow_consistent", "leflow_adapted"]
    world_path = root / "runs/world_3072/best.pt"
    checkpoint_hashes = {"world": file_hash(world_path)}
    with torch.inference_mode():
        for method in methods:
            checkpoint = root / "runs" / f"{method}_3072/best.pt"
            checkpoint_hashes[method] = file_hash(checkpoint)
            model, world, saved = load_models(c, world_path, method, checkpoint, device)
            for row in rows[rank::size]:
                deadline.check()
                assert file_hash(root / row["path"]) == row["sha256"]
                seed_all((3072 + row["seed"]) % 2**32)
                with h5py.File(root / row["path"], "r") as f:
                    goal = (torch.tensor(f["goal"][:], device=device).float() - mean) / std
                    initial = f["initial_rgb"][:]
                clip = np.repeat(initial[None], c["data"]["history_frames"], axis=0)
                start = (encoder(clip[None]).float() - mean) / std
                n, m, k, ad = 8, c["model"]["segments"], c["model"]["chunk_steps"], 8
                s, g = start.expand(n, -1, -1), goal[None].expand(n, -1, -1)
                interior, actions = model.planner.sample(s, g, steps=8,
                        path_only=method == "leflow_adapted", budget=deadline)
                paths = torch.cat([s[:, None], interior, g[:, None]], 1)
                if method == "leflow_adapted":
                    actions = model.inverse(paths[:, :-1].flatten(0, 1), paths[:, 1:].flatten(0, 1)).reshape(n, m, k, ad)
                    end = world.rollout(s, actions.clamp(-1, 1).reshape(n, m*k, ad), budget=deadline)[:, -1]
                    legacy = distance(end, g)
                else:
                    end = world.rollout(paths[:, :-1].flatten(0, 1),
                            actions.clamp(-1, 1).reshape(n*m, k, ad), budget=deadline)[:, -1]
                    legacy = distance(end, paths[:, 1:].flatten(0, 1)).reshape(n, m).mean(1)
                continuous = scoring.continuous_plan_score(world, start, goal[None], paths, actions, budget=deadline)
                refined, clips, stills, simulator = [], [], [], []
                for i in range(n):
                    deadline.check()
                    chunk, _ = cem(world, start, paths[i:i+1, 1], k, ad, 32, 8, 3,
                                   mean=actions[i, 0].clamp(-1, 1), budget=deadline)
                    refined.append(chunk)
                    env = make_env(row["task"], row["seed"], c["data"])
                    frames, reward_sum, success = [], 0., False
                    try:
                        env.reset()
                        frames.append(env.render().copy())
                        assert np.array_equal(frames[0], initial)
                        for primitive in chunk.cpu().numpy().reshape(-1, 4):
                            _, reward, terminated, truncated, info = env.step(np.clip(primitive, -1, 1))
                            reward_sum += float(reward)
                            success |= bool(info["success"])
                            frames.append(env.render().copy())
                            if terminated or truncated:
                                break
                    finally:
                        env.close()
                    clips.append(history_clip(frames, len(frames)-1, c["data"]["history_frames"]))
                    stills.append(np.repeat(frames[-1][None], c["data"]["history_frames"], axis=0))
                    simulator.append(dict(reward=reward_sum, success=success, primitive_actions=len(frames)-1))
                # Single batched encoder pass; no change to weights or precision.
                actual = (encoder(np.stack(clips + stills)).float() - mean) / std
                real_history, real_static = actual[:n], actual[n:]
                refined = torch.stack(refined)
                pred = world.rollout(s, refined, budget=deadline)[:, -1]
                pred_subgoal = distance(pred, paths[:, 1])
                real_subgoal = distance(real_history, paths[:, 1])
                initial_distance = distance(s, g)
                pred_progress = initial_distance - distance(pred, g)
                real_progress = initial_distance - distance(real_static, g)
                candidates = []
                for i in range(n):
                    candidates.append(dict(index=i, legacy_cost=float(legacy[i]),
                        continuous_cost=float(continuous[i]),
                        predicted_subgoal_distance=float(pred_subgoal[i]),
                        observed_subgoal_distance=float(real_subgoal[i]),
                        predicted_goal_progress=float(pred_progress[i]),
                        observed_static_goal_progress=float(real_progress[i]),
                        refined_actions=refined[i].cpu().tolist(), **simulator[i]))
                legacy_i, continuous_i = int(legacy.argmin()), int(continuous.argmin())
                record = dict(method=method, checkpoint_step=saved["step"], episode_id=row["id"],
                    reset_seed=row["seed"], task=row["task"], split="validation", candidates=candidates,
                    legacy_selected=legacy_i, continuous_selected=continuous_i,
                    legacy_progress_regret=float(real_progress.max()-real_progress[legacy_i]),
                    continuous_progress_regret=float(real_progress.max()-real_progress[continuous_i]),
                    random_selection_expected_progress=float(real_progress.mean()),
                    rho_legacy_goal_progress=spearman(-legacy.cpu().numpy(),real_progress.cpu().numpy()),
                    rho_continuous_goal_progress=spearman(-continuous.cpu().numpy(),real_progress.cpu().numpy()),
                    rho_predicted_actual_goal_progress=spearman(pred_progress.cpu().numpy(),real_progress.cpu().numpy()),
                    rho_predicted_actual_subgoal_distance=spearman(pred_subgoal.cpu().numpy(),real_subgoal.cpu().numpy()))
                records.append(record)
                save_json(output / f"rank-{rank}.json", dict(records=records,
                    elapsed_seconds=time.perf_counter()-started, complete=False))
                print(json.dumps(dict(event="diagnostic_case",rank=rank,method=method,task=row["task"],
                    rho_legacy=record["rho_legacy_goal_progress"],rho_continuous=record["rho_continuous_goal_progress"])),flush=True)
            del model, world
            torch.cuda.empty_cache()
    save_json(output / f"rank-{rank}.json", dict(records=records,complete=True,
        elapsed_seconds=time.perf_counter()-started,checkpoint_hashes=checkpoint_hashes,
        manifest_sha256=file_hash(root/'manifest.json'),scoring_sha256=file_hash(args.scoring_file),
        gpu=str(device),size=size,optimization_updates=0,benchmark=False,
        protocol='13 existing validation resets, 8 candidates, 2 methods, refined 5-control-step chunks'))


if __name__ == "__main__":
    main()
