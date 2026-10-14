"""Native linear-flow P&P and dynamics-guided P&P.

Time convention: t=0 Gaussian noise; t=1 clean path interior. Start and goal
are not sampled. D(x,t)=x+(1-t)*v(x,t); P&P re-noises the clean prediction at
the *same* t, not the next physical or diffusion timestep. Inspired by
https://arxiv.org/abs/2601.18577; this is a latent-planning adaptation, not a
reproduction of that paper's video masking, CFG, or noise schedule.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Callable, Protocol

import torch
from torch import Tensor


@dataclass(frozen=True)
class RefinementConfig:
    mode: str = "native"
    indices: tuple[int, ...] = (4, 8, 12, 14)
    rounds: int = 2
    radius: float = 0.05
    consistency_weight: float = 0.1
    guidance_batch_size: int = 8

    def validate(self, flow_steps: int) -> None:
        if isinstance(flow_steps, bool) or not isinstance(flow_steps, int) or flow_steps < 1:
            raise ValueError("flow_steps must be a positive integer")
        if self.mode not in {"native", "pnp", "guidance", "guided_pnp"}:
            raise ValueError(f"unknown refinement mode: {self.mode}")
        if isinstance(self.rounds, bool) or not isinstance(self.rounds, int) or self.rounds < 0:
            raise ValueError("rounds must be a nonnegative integer")
        if not isinstance(self.guidance_batch_size, int) or self.guidance_batch_size < 1:
            raise ValueError("guidance_batch_size must be positive")
        for name, value in [("radius", self.radius), ("consistency_weight", self.consistency_weight)]:
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.mode != "native" and self.rounds > 0:
            if not self.indices or len(set(self.indices)) != len(self.indices):
                raise ValueError("active indices must be nonempty and unique")
            if any(isinstance(i, bool) or not isinstance(i, int) or not 0 < i < flow_steps for i in self.indices):
                raise ValueError("active indices must satisfy 0 < index < flow_steps")

    @classmethod
    def from_dict(cls, values: dict) -> "RefinementConfig":
        values = dict(values)
        if "indices" in values:
            values["indices"] = tuple(values["indices"])
        return cls(**values)


@dataclass
class Work:
    """Actual work, not flow-only NFE relabelled as end-to-end compute."""
    flow_calls: int = 0
    flow_samples: int = 0
    inverse_calls: int = 0
    inverse_transitions: int = 0
    world_calls: int = 0
    world_transitions: int = 0
    backward_calls: int = 0
    guided_candidates: int = 0
    refinement_rounds: int = 0
    gradient_norm_sum: float = 0.0
    gradient_tokens: int = 0
    zero_gradient_tokens: int = 0
    max_correction_l2: float = 0.0
    controller_ms: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


class Energy(Protocol):
    def __call__(self, clean: Tensor, start: Tensor, goal: Tensor,
                 weight: float, work: Work) -> Tensor:
        """Return one cost per candidate, shape [Q]; must preserve clean's graph."""
        ...


def _finite(x: Tensor, label: str) -> None:
    if not bool(torch.isfinite(x).all()):
        raise FloatingPointError(f"non-finite {label}; refusing to report a completed plan")


def correct_clean(clean: Tensor, start: Tensor, goal: Tensor, *,
                  energy: Energy, config: RefinementConfig, work: Work,
                  check: Callable[[], None] | None = None) -> Tensor:
    """Bound each token's correction in L2; never train model parameters.

Explicitly disable inference_mode as stable-worldmodel callers may enable it.
Fresh normal (non-inference) tensors are required for autograd to save inputs.
Microbatching bounds gradient memory. Energy must be candidate-independent.
"""
    if config.radius == 0:
        return clean
    results = []
    for lo in range(0, len(clean), config.guidance_batch_size):
        if check is not None: check()
        hi = lo + config.guidance_batch_size
        with torch.inference_mode(False), torch.enable_grad():
            y = clean[lo:hi].detach().clone().requires_grad_(True)
            zs = start[lo:hi].detach().clone()
            zg = goal[lo:hi].detach().clone()
            costs = energy(y, zs, zg, config.consistency_weight, work)
            if costs.shape != (len(y),):
                raise ValueError("energy must return one cost per candidate [Q]")
            _finite(costs, "guidance energy")
            gradient, = torch.autograd.grad(costs.sum(), y, create_graph=False)
            _finite(gradient, "guidance gradient")
            norm = gradient.norm(dim=-1, keepdim=True)
            updated = y - config.radius * gradient / (norm + 1e-8)
            results.append(updated.detach())
            work.backward_calls += 1
            work.guided_candidates += len(y)
            work.gradient_norm_sum += float(norm.detach().sum())
            work.gradient_tokens += norm.numel()
            work.zero_gradient_tokens += int((norm.detach() == 0).sum())
            work.max_correction_l2 = max(work.max_correction_l2,
                                        float((updated-y).detach().norm(dim=-1).max()))
        if check is not None: check()
    return torch.cat(results, dim=0)


def sample_interiors(
    flow: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    start: Tensor,
    goal: Tensor,
    *,
    horizon: int,
    num_samples: int,
    flow_steps: int,
    config: RefinementConfig,
    generator: torch.Generator,
    perturb_generator: torch.Generator | None = None,
    energy: Energy | None = None,
    work: Work | None = None,
    initial_noise: Tensor | None = None,
    check: Callable[[], None] | None = None,
) -> tuple[Tensor, Work]:
    """Return [B,N,H-1,D] interiors; zero refinement matches native Euler.

The initial candidate RNG and perturbation RNG are separate, so adding P&P
calls does not change the next decision's initial candidate draws. Only the
noise sampler/gradient corrector changes; fixed endpoints stay outside x.
"""
    config.validate(flow_steps)
    for name, value in [("horizon", horizon), ("num_samples", num_samples)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if start.ndim != 2 or goal.shape != start.shape or start.shape[0] < 1 or start.shape[1] < 1:
        raise ValueError("start and goal must have equal nonempty shape [B,D]")
    if goal.device != start.device or goal.dtype != start.dtype or not start.is_floating_point():
        raise ValueError("start/goal must share floating dtype and device")
    if start.dtype not in {torch.float32, torch.float64}:
        raise ValueError("use float32/float64 for comparable refinement and gradient numerics")
    _finite(start, "start")
    _finite(goal, "goal")
    active = config.mode != "native" and config.rounds > 0
    stochastic = active and config.mode in {"pnp", "guided_pnp"}
    guided = active and config.mode in {"guidance", "guided_pnp"} and config.radius > 0
    if stochastic and (perturb_generator is None or perturb_generator is generator):
        raise ValueError("P&P requires a separate explicit perturb_generator")
    if guided and energy is None:
        raise ValueError("guidance requires a differentiable energy")
    work = Work() if work is None else work
    b, d = start.shape
    q = b * num_samples
    # Gradient-requiring substeps must also work under an outer inference_mode.
    with torch.inference_mode(False), torch.no_grad():
        zs = start.detach().clone()[:, None].expand(b, num_samples, d).reshape(q, d)
        zg = goal.detach().clone()[:, None].expand(b, num_samples, d).reshape(q, d)
        if horizon == 1:
            return start.new_empty(b, num_samples, 0, d), work
        shape = (q, horizon - 1, d)
        if initial_noise is None:
            x = torch.randn(shape, device=start.device, dtype=start.dtype, generator=generator)
        else:
            if initial_noise.shape != shape or initial_noise.device != start.device or initial_noise.dtype != start.dtype:
                raise ValueError(f"initial_noise must match {shape}, device and dtype")
            x = initial_noise.detach().clone()
        _finite(x, "initial noise")
        dt = 1.0 / flow_steps

        def velocity(state: Tensor, times: Tensor) -> Tensor:
            if check is not None: check()
            value = flow(state, times, zs, zg)
            if value.shape != state.shape:
                raise ValueError("flow velocity shape must equal path-interior shape")
            _finite(value, "flow velocity")
            work.flow_calls += 1
            work.flow_samples += q
            if check is not None: check()
            return value

        for i in range(flow_steps):
            t = i * dt
            times = torch.full((q,), t, device=x.device, dtype=x.dtype)
            v = velocity(x, times)
            if active and i in config.indices:
                for _ in range(config.rounds):
                    clean = x + (1.0 - t) * v
                    if guided:
                        clean = correct_clean(clean, zs, zg, energy=energy, config=config, work=work, check=check)
                    if stochastic:
                        noise = torch.randn(shape, device=x.device, dtype=x.dtype, generator=perturb_generator)
                        x = t * clean + (1.0 - t) * noise
                        v = velocity(x, times)
                    else:
                        # Guidance-only control: correct the denoised endpoint;
                        # retain current x and derive the corresponding velocity.
                        v = (clean - x) / (1.0 - t)
                    work.refinement_rounds += 1
            x = x + v * dt
            _finite(x, "integrated interior")
    return x.reshape(b, num_samples, horizon - 1, d), work
