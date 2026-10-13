"""Frozen rectified-flow inference: P&P and execution-consistency guidance.

Noise time is 0 and clean time is 1. No optimizer or weight adaptation is used.
P&P: https://arxiv.org/abs/2601.18577 . Dynamics guidance is our experimental
extension, not part of that paper. Call counts are not FLOPs or latency estimates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable

import torch
from torch import Tensor


@dataclass(frozen=True)
class RefinementConfig:
    mode: str = "world_pnp"
    samples: int = 64
    steps: int = 16
    horizon: int = 5
    refine_at: tuple[int, ...] = (6, 8, 10, 12)
    repeats: int = 2
    radius: float = 0.05
    consistency_weight: float = 0.1

    def __post_init__(self):
        if self.mode not in {"euler", "pnp", "gradient", "world_pnp"}:
            raise ValueError(f"Unknown refinement mode: {self.mode}")
        for name in ("samples", "steps", "horizon", "repeats"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if min(self.samples, self.steps) < 1 or self.horizon < 2 or self.repeats < 0:
            raise ValueError("Require samples/steps >= 1, horizon >= 2, repeats >= 0")
        if len(set(self.refine_at)) != len(self.refine_at):
            raise ValueError("Duplicate refinement indices")
        if self.mode != "euler" and any(
            not isinstance(i, int) or i <= 0 or i >= self.steps for i in self.refine_at
        ):
            raise ValueError("Refinement indices must satisfy 0 < index < steps")
        for name in ("radius", "consistency_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass
class Compute:
    flow_batches: int = 0
    flow_candidate_evaluations: int = 0
    inverse_batches: int = 0
    inverse_candidate_transitions: int = 0
    world_batches: int = 0
    world_candidate_transitions: int = 0
    backward_batches: int = 0
    backward_candidate_transitions: int = 0
    correction_updates: int = 0
    gradient_norm_sum: float = 0.0
    max_node_displacement: float = 0.0

    def as_dict(self):
        return asdict(self)


def finite(x: Tensor, label: str) -> Tensor:
    if not torch.isfinite(x).all():
        raise FloatingPointError(f"Nonfinite {label}")
    return x


def assemble(start: Tensor, interior: Tensor, goal: Tensor) -> Tensor:
    """[M,d], [M,H-1,d], [M,d] -> [M,H+1,d], fixed endpoints."""
    return torch.cat((start[:, None], interior, goal[:, None]), dim=1)


class PlannerOps:
    """Count real module calls; rollout must be differentiable and include z0.

    velocity: (x[M,H-1,d], t[M], start[M,d], goal[M,d]) -> same shape as x.
    decode: paths[M,H+1,d] -> bounded/executable actions[M,H,block_dim].
    rollout: (start[M,d], actions[M,H,block_dim]) -> states[M,H+1,d].
    Action units and clipping must match the deployment adapter.
    """
    def __init__(self, velocity: Callable, decode: Callable, rollout: Callable,
                 check: Callable[[], None] | None = None):
        self.velocity_fn, self.decode_fn, self.rollout_fn = velocity, decode, rollout
        self.check = check or (lambda: None)
        self.compute = Compute()

    def velocity(self, x, t, start, goal):
        self.check()
        value = self.velocity_fn(x, t, start, goal)
        self.compute.flow_batches += 1
        self.compute.flow_candidate_evaluations += len(x)
        self.check()
        if value.shape != x.shape:
            raise ValueError("Velocity shape differs from noisy path shape")
        return finite(value, "flow velocity")

    def consequences(self, start, paths):
        self.check()
        actions = finite(self.decode_fn(paths), "decoded actions")
        m, h = paths.shape[0], paths.shape[1] - 1
        if actions.ndim != 3 or actions.shape[:2] != (m, h):
            raise ValueError("Decoder must return [candidates,horizon,block_dim]")
        self.compute.inverse_batches += 1
        self.compute.inverse_candidate_transitions += m * h
        self.check()
        pred = self.rollout_fn(start, actions)
        self.compute.world_batches += h
        self.compute.world_candidate_transitions += m * h
        self.check()
        if pred.shape != paths.shape:
            raise ValueError("Rollout must include start and have the full path shape")
        return actions, finite(pred, "world rollout")


def correct(clean: Tensor, start: Tensor, goal: Tensor, ops: PlannerOps,
            cfg: RefinementConfig) -> Tensor:
    """Input-gradient update only; supports callers inside inference_mode."""
    with torch.inference_mode(False), torch.enable_grad():
        y = clean.detach().clone().requires_grad_(True)
        s, g = start.detach().clone(), goal.detach().clone()
        paths = assemble(s, y, g)
        _, pred = ops.consequences(s, paths)
        terminal = (pred[:, -1] - g).square().mean(-1)
        consistency = (pred[:, 1:-1] - y).square().mean(dim=(-1, -2))
        # Sum per-candidate objectives: gradient scale must not depend on pool size.
        objective = (terminal + cfg.consistency_weight * consistency).sum()
        gradient = finite(torch.autograd.grad(objective, y)[0], "path gradient")
        norms = gradient.norm(dim=-1, keepdim=True)
        displacement = cfg.radius * gradient / (norms + 1e-8)
        result = y - displacement
        c = ops.compute
        c.backward_batches += 1
        c.backward_candidate_transitions += len(y) * cfg.horizon
        c.correction_updates += 1
        c.gradient_norm_sum += float(norms.detach().mean())
        c.max_node_displacement = max(c.max_node_displacement,
                                     float(displacement.detach().norm(dim=-1).max()))
        ops.check()
        return finite(result.detach(), "corrected path")


def sample(start: Tensor, goal: Tensor, ops: PlannerOps, cfg: RefinementConfig,
           *, initial_noise: Tensor | None = None,
           generator: torch.Generator | None = None,
           perturb_generator: torch.Generator | None = None) -> Tensor:
    """Return [B,N,H+1,d]. No weights change; fixed endpoints never get noise.

    Pass initial_noise[B,N,H-1,d] for exact paired candidate pools. Separate
    perturb_generator prevents refinement draws from changing future base pools.
    """
    if start.ndim != 2 or goal.shape != start.shape or start.device != goal.device:
        raise ValueError("start and goal must share [B,d] shape and device")
    if not start.is_floating_point() or goal.dtype != start.dtype or len(start) < 1:
        raise ValueError("Nonempty floating-point endpoints with equal dtype required")
    b, d = start.shape
    with torch.inference_mode(False), torch.no_grad():
        start, goal = finite(start.clone(), "start"), finite(goal.clone(), "goal")
        s = start[:, None].expand(b, cfg.samples, d).reshape(-1, d)
        g = goal[:, None].expand(b, cfg.samples, d).reshape(-1, d)
        shape = (b, cfg.samples, cfg.horizon - 1, d)
        if initial_noise is None:
            x = torch.randn(shape, device=s.device, dtype=s.dtype, generator=generator)
        else:
            if initial_noise.shape != shape or initial_noise.device != s.device:
                raise ValueError(f"initial_noise must have shape {shape} on {s.device}")
            x = initial_noise.to(dtype=s.dtype).clone()
        x = finite(x.reshape(-1, cfg.horizon - 1, d), "initial noise")
        for index in range(cfg.steps):
            t = x.new_full((len(x),), index / cfg.steps)
            velocity = ops.velocity(x, t, s, g)
            if cfg.mode != "euler" and index in cfg.refine_at:
                for _ in range(cfg.repeats):
                    clean = x + (1 - index / cfg.steps) * velocity
                    revised = (correct(clean, s, g, ops, cfg)
                               if cfg.mode in {"gradient", "world_pnp"} else clean)
                    if cfg.mode in {"pnp", "world_pnp"}:
                        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype,
                                            generator=perturb_generator)
                        x = (index / cfg.steps) * revised + (1 - index / cfg.steps) * noise
                    else:
                        # Same-noise guidance control: change clean component only.
                        x = x + (index / cfg.steps) * (revised - clean)
                    velocity = ops.velocity(x, t, s, g)
            x = finite(x + velocity / cfg.steps, "integrated path")
        return assemble(s, x, g).reshape(b, cfg.samples, cfg.horizon + 1, d)


def select(paths: Tensor, ops: PlannerOps) -> dict[str, Tensor]:
    """Identical terminal-world-distance ranking for every sampler variant."""
    with torch.inference_mode(False), torch.no_grad():
        b, n, hp, d = paths.shape
        flat = paths.detach().clone().reshape(b * n, hp, d)
        actions, pred = ops.consequences(flat[:, 0], flat)
        costs = (pred[:, -1] - flat[:, -1]).square().mean(-1).reshape(b, n)
        best = costs.argmin(-1)
        actions = actions.reshape(b, n, hp - 1, -1)
        return {"actions": actions[torch.arange(b, device=paths.device), best],
                "costs": costs.gather(1, best[:, None]).squeeze(1),
                "all_costs": costs, "selected_indices": best}
