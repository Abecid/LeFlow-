"""Differentiable native-LeWM adapter and frozen planning engine.

This module deliberately bypasses only the inference-only runtime rollout
wrappers, not JEPA's action encoder, predictor, history, or action normalization.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import asdict

import torch
from torch import Tensor, nn

from .core import RefinementConfig, Work, _finite, sample_interiors


def state_fingerprint(model: nn.Module) -> str:
    """Hash weights AND buffers; call at run boundaries, never per decision."""
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class NativeAdapter:
    def __init__(self, runtime: nn.Module, history_size: int = 3):
        if history_size < 1:
            raise ValueError("history_size must be positive")
        self.runtime = runtime.eval().requires_grad_(False)
        self.history_size = history_size
        for name in ("flow", "inverse_dynamics", "lewm"):
            if not hasattr(runtime, name):
                raise TypeError(f"native runtime is missing {name}")

    @staticmethod
    def join(clean: Tensor, start: Tensor, goal: Tensor) -> Tensor:
        return torch.cat([start[:, None], clean, goal[:, None]], dim=1)

    def decode(self, paths: Tensor, work: Work) -> Tensor:
        q, nodes, d = paths.shape
        if nodes < 2:
            raise ValueError("a path requires both endpoints")
        actions = self.runtime.inverse_dynamics(
            paths[:, :-1].reshape(-1, d), paths[:, 1:].reshape(-1, d)
        ).reshape(q, nodes - 1, -1)
        _finite(actions, "decoded actions")
        work.inverse_calls += 1
        work.inverse_transitions += q * (nodes - 1)
        return actions

    def rollout(self, start: Tensor, actions: Tensor, work: Work) -> Tensor:
        """[Q,D] + [Q,H,A] -> [Q,H+1,D], preserving path gradients.

Matches LatentPlannerRuntime.rollout_paths at upstream blob
91bf238994765d67e9b941e138d5da482feafb2a. No clipping is introduced here:
WorldModelPolicy remains responsible for the native physical action interface.
"""
        if actions.ndim != 3 or len(actions) != len(start):
            raise ValueError("actions must have shape [Q,H,A]")
        z = start[:, None]
        for t in range(actions.shape[1]):
            act_hist = actions[:, max(0, t - self.history_size + 1):t + 1]
            emb_hist = z[:, -act_hist.size(1):]
            act_emb = self.runtime.lewm.action_encoder(act_hist)
            pred = self.runtime.lewm.predict(emb_hist, act_emb)[:, -1:]
            if pred.shape != z[:, -1:].shape:
                raise ValueError("LeWM predictor must preserve [Q,1,D] state shape")
            z = torch.cat([z, pred], dim=1)
            work.world_calls += 1
            work.world_transitions += len(start)
        _finite(z, "world rollout")
        return z

    def energy(self, clean: Tensor, start: Tensor, goal: Tensor,
               weight: float, work: Work) -> Tensor:
        paths = self.join(clean, start, goal)
        actions = self.decode(paths, work)
        predicted = self.rollout(start, actions, work)
        goal_cost = (predicted[:, -1] - goal).square().mean(-1)
        if clean.shape[1] and weight:
            consistency = (predicted[:, 1:-1] - clean).square().mean((-1, -2))
            goal_cost = goal_cost + weight * consistency
        return goal_cost


class PlannerEngine:
    def __init__(self, runtime: nn.Module, config: RefinementConfig, history_size: int = 3):
        self.adapter = NativeAdapter(runtime, history_size)
        self.runtime = runtime
        self.config = config
        self.rollout_count = 0
        self.last_work = Work()

    @property
    def device(self) -> torch.device:
        return next(self.runtime.parameters()).device

    def plan(self, info_dict: dict, *, horizon: int, num_samples: int,
             flow_steps: int, generator: torch.Generator,
             perturb_generator: torch.Generator) -> dict:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        before = time.perf_counter()
        work = Work()
        with torch.inference_mode(False), torch.no_grad():
            start, goal = self.runtime.encode_current_and_goal(info_dict)
            start, goal = start.detach().clone(), goal.detach().clone()
            clean, work = sample_interiors(
                self.runtime.flow, start, goal, horizon=horizon,
                num_samples=num_samples, flow_steps=flow_steps, config=self.config,
                generator=generator, perturb_generator=perturb_generator,
                energy=self.adapter.energy, work=work,
            )
            b, n, _, d = clean.shape
            zs = start[:, None].expand(b, n, d).reshape(b * n, d)
            zg = goal[:, None].expand(b, n, d).reshape(b * n, d)
            paths = self.adapter.join(clean.reshape(b * n, horizon - 1, d), zs, zg)
            actions = self.adapter.decode(paths, work)
            predicted = self.adapter.rollout(zs, actions, work)
            # Keep native terminal-only reranking IDENTICAL for every method.
            costs = (predicted[:, -1] - zg).square().mean(-1).reshape(b, n)
            _finite(costs, "selection costs")
            choice = costs.argmin(dim=1)
            rows = torch.arange(b, device=choice.device)
            actions = actions.reshape(b, n, horizon, -1)
            out = {
                "actions": actions[rows, choice].detach().cpu(),
                "costs": costs[rows, choice].detach().cpu(),
                "goal_costs": costs[rows, choice].detach().cpu(),
                "all_costs": costs.detach().cpu(),
                "selected_candidates": choice.detach().cpu(),
            }
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        work.controller_ms = (time.perf_counter() - before) * 1000.0
        self.last_work = work
        # Compatibility field counts candidate trajectories, not primitive calls.
        self.rollout_count += work.world_transitions // horizon
        out["work"] = asdict(work)
        return out
