"""Execute generated subgoals using a frozen action-conditioned world model."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from btm_jepa.models import PathModel, PlannerTrainingModel, sample_paths


def build_model(arch):
    return PlannerTrainingModel(
        PathModel(**arch["path"]),
        action_dim=arch["inverse"]["action_dim"],
        inverse_hidden=arch["inverse"]["hidden_dim"],
    )


def atomic_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, tmp)
    tmp.replace(path)


class SubgoalRuntime(nn.Module):
    def __init__(self, planner, dynamics, action_block=5, history_size=3):
        super().__init__()
        self.planner, self.dynamics = (
            planner.eval(),
            dynamics.eval().requires_grad_(False),
        )
        self.action_block, self.history_size = action_block, history_size
        self.dynamics_calls = 0

    @property
    def device(self):
        return next(self.planner.parameters()).device

    def step(self, z_history, action_history):
        self.dynamics_calls += len(z_history)
        return self.dynamics.predict(
            z_history[:, -self.history_size :],
            self.dynamics.action_encoder(action_history[:, -self.history_size :]),
        )[:, -1]

    @torch.no_grad()
    def rollout(self, start, actions):
        z = start[:, None]
        for j in range(actions.shape[1]):
            nxt = self.step(z, actions[:, : j + 1])
            z = torch.cat([z, nxt[:, None]], 1)
        return z

    @torch.no_grad()
    def decode_and_verify(self, paths, spacing, low, high):
        """Predict actions, then score their consequences, never clamped endpoints.

        For coarse paths, interpolate the *next local target* toward each waypoint
        over `spacing` blocks. This supplies proposals; world-model rollout tests
        them and CEM can refine the selected first segment. No pixel decoding.
        """
        b, s, h1, d = paths.shape
        p = paths.reshape(b * s, h1, d)
        z = p[:, :1]
        actions, misses = [], []
        for j in range(1, h1):
            for k in range(spacing):
                target = z[:, -1] + (p[:, j] - z[:, -1]) / (spacing - k)
                a = self.planner.inverse_actions(z[:, -1], target).clamp(low, high)
                actions.append(a)
                nxt = self.step(z, torch.stack(actions, 1))
                z = torch.cat([z, nxt[:, None]], 1)
            misses.append((z[:, -1] - p[:, j]).square().mean(-1))
        action = torch.stack(actions, 1).reshape(b, s, (h1 - 1) * spacing, -1)
        goal_error = (z[:, -1] - p[:, -1]).square().mean(-1).reshape(b, s)
        waypoint_error = torch.stack(misses, -1).mean(-1).reshape(b, s)
        return action, goal_error, waypoint_error

    @torch.no_grad()
    def refine(
        self,
        start,
        goal,
        initial,
        low,
        high,
        *,
        candidates,
        iterations,
        elites,
        generator,
    ):
        if iterations == 0:
            return initial
        if not 1 < elites <= candidates:
            raise ValueError("CEM requires 1 < elites <= candidates")
        b, k, a = initial.shape
        mean, std = initial.clone(), torch.ones_like(initial)
        best, best_cost = initial, torch.full((b,), float("inf"), device=start.device)
        for _ in range(iterations):
            noise = torch.randn(
                b, candidates, k, a, device=start.device, generator=generator
            )
            samples = (mean[:, None] + noise * std[:, None]).clamp(low, high)
            samples[:, 0] = best
            z = self.rollout(
                start[:, None].expand(b, candidates, -1).reshape(b * candidates, -1),
                samples.reshape(b * candidates, k, a),
            )
            cost = (
                (z[:, -1].reshape(b, candidates, -1) - goal[:, None]).square().mean(-1)
            )
            idx = cost.topk(elites, largest=False).indices
            elite = samples[torch.arange(b, device=start.device)[:, None], idx]
            mean, std = elite.mean(1), elite.std(1, unbiased=False).clamp_min(0.05)
            current_cost = cost.gather(1, idx[:, :1])[:, 0]
            better = current_cost < best_cost
            best = torch.where(better[:, None, None], elite[:, 0], best)
            best_cost = torch.minimum(best_cost, current_cost)
        return best


class SubgoalSolver:
    """stable-worldmodel solver for both flat and hierarchical comparisons."""

    def __init__(
        self,
        checkpoint,
        *,
        device="cuda",
        mode="hierarchical",
        num_samples=64,
        flow_steps=16,
        spacing=2,
        horizon=5,
        reachability_weight=1.0,
        cem_candidates=32,
        cem_iterations=3,
        cem_elites=8,
        seed=42,
        lewm_checkpoint=None,
    ):
        from latent_planner import load_lewm

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.payload = payload
        planner = build_model(payload["arch"])
        planner.load_state_dict(payload["model"])
        dynamics = load_lewm(lewm_checkpoint or payload["data"]["lewm_checkpoint"])
        self.model = (
            SubgoalRuntime(planner, dynamics, payload["config"]["data"]["action_block"])
            .to(device)
            .eval()
        )
        self.mode, self.spacing, self.path_horizon = mode, spacing, horizon
        if mode not in {"flat", "hierarchical"} or spacing < 1:
            raise ValueError("Choose flat or hierarchical mode and positive spacing")
        if mode == "flat" and spacing != 1:
            raise ValueError("Flat mode requires spacing=1")
        trained = payload["config"]["data"]["spacing_blocks"]
        if spacing not in trained:
            raise ValueError(
                f"Spacing {spacing} was not trained; trained spacings: {trained}"
            )
        self.samples, self.flow_steps = num_samples, flow_steps
        self.reachability_weight = reachability_weight
        self.cem = dict(
            candidates=cem_candidates, iterations=cem_iterations, elites=cem_elites
        )
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.latencies, self.goal_errors, self.waypoint_errors = [], [], []
        self.observed_subgoal_errors = []
        self.previous_subgoals = None

    def configure(self, *, action_space, n_envs, config):
        self.config, self.n_envs = config, n_envs
        expected = self.path_horizon if self.mode == "flat" else self.spacing
        if config.horizon != expected or config.action_block != self.model.action_block:
            raise ValueError(
                f"Controller horizon must be {expected}, action_block={self.model.action_block}"
            )
        source = self.payload["data"]["source"]
        mean, std = np.asarray(source["action_mean"]), np.asarray(source["action_std"])
        low, high = (
            np.asarray(action_space.low).reshape(-1, len(mean))[0],
            np.asarray(action_space.high).reshape(-1, len(mean))[0],
        )
        self.low = torch.tensor(
            np.tile((low - mean) / std, self.model.action_block),
            dtype=torch.float32,
            device=self.model.device,
        )
        self.high = torch.tensor(
            np.tile((high - mean) / std, self.model.action_block),
            dtype=torch.float32,
            device=self.model.device,
        )
        self.action_dim = len(self.low)
        self.horizon = config.horizon

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    @torch.no_grad()
    def solve(self, info_dict, init_action=None):
        del init_action
        device = self.model.device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        z0 = self.model.dynamics.encode({"pixels": info_dict["pixels"].to(device)})[
            "emb"
        ][:, -1]
        zg = self.model.dynamics.encode({"pixels": info_dict["goal"].to(device)})[
            "emb"
        ][:, -1]
        if self.previous_subgoals is not None and len(z0) == len(
            self.previous_subgoals
        ):
            self.observed_subgoal_errors.extend(
                (z0 - self.previous_subgoals).square().mean(-1).cpu().tolist()
            )
        all_actions, all_costs, all_subgoals = [], [], []
        # Bound peak memory independently of vectorized environment count.
        for i in range(len(z0)):
            paths = sample_paths(
                self.model.planner.path,
                z0[i : i + 1],
                zg[i : i + 1],
                horizon=self.path_horizon,
                spacing=self.spacing,
                num_samples=self.samples,
                flow_steps=self.flow_steps,
                generator=self.generator,
            )
            actions, goal_error, reach_error = self.model.decode_and_verify(
                paths, self.spacing, self.low, self.high
            )
            cost = goal_error + (
                self.reachability_weight * reach_error
                if self.mode == "hierarchical"
                else 0
            )
            best = cost.argmin(1).item()
            chosen = actions[:, best, : self.config.horizon]
            subgoal = paths[:, best, 1]
            if self.mode == "hierarchical":
                chosen = self.model.refine(
                    z0[i : i + 1],
                    subgoal,
                    chosen,
                    self.low,
                    self.high,
                    **self.cem,
                    generator=self.generator,
                )
            all_actions.append(chosen.cpu())
            all_costs.append(float(cost[0, best]))
            all_subgoals.append(subgoal)
            self.goal_errors.append(float(goal_error[0, best]))
            self.waypoint_errors.append(float(reach_error[0, best]))
        self.previous_subgoals = torch.cat(all_subgoals)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.latencies.append(time.perf_counter() - started)
        return dict(
            actions=torch.cat(all_actions),
            costs=all_costs,
            rollout_count=self.model.dynamics_calls,
        )

    def diagnostics(self):
        def mean(xs):
            return float(np.mean(xs)) if xs else 0.0

        return {
            "planning_batch_latency_ms_mean": 1000 * mean(self.latencies),
            "planning_batch_latency_ms_p95": float(
                np.percentile(self.latencies, 95) * 1000
            )
            if self.latencies
            else 0,
            "selected_rollout_goal_mse": mean(self.goal_errors),
            "selected_rollout_waypoint_mse": mean(self.waypoint_errors),
            "observed_subgoal_mse_after_chunk": mean(self.observed_subgoal_errors),
            "world_model_state_predictions": self.model.dynamics_calls,
            "generator_nfe_per_candidate": self.flow_steps
            if self.model.planner.path.method == "flow"
            else 1,
            "planner_calls": len(self.latencies),
        }
