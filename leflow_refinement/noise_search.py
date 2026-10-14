"""JEPA-scored CEM over a frozen LeFlow sampler's initial noise.

No model training, extra critic, latent-gradient edits or action averaging.
Elite moments fit RAW Gaussian proposals; optional whole-vector shell projection
is part of the objective mapping before the existing Euler sampler. The final
score is exactly core.select's predicted terminal latent MSE.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .core import PlannerOps, RefinementConfig, finite, sample, select


@dataclass(frozen=True)
class NoiseSearchConfig:
    mode: str = 'noise_cem'
    samples: int = 64
    steps: int = 16
    horizon: int = 5
    rounds: int = 3
    elites: int = 8
    smoothing: float = 0.25  # Weight on OLD mean/variance, not on new moments.
    min_std: float = 0.1    # Variance floor is min_std**2.
    project_shell: bool = False

    def __post_init__(self):
        if self.mode not in {'noise_cem', 'noise_random'}:
            raise ValueError('mode must be noise_cem or noise_random')
        for key in ('samples', 'steps', 'horizon', 'rounds', 'elites'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        if self.horizon < 2 or not 1 <= self.elites <= self.samples:
            raise ValueError('Require horizon >= 2 and 1 <= elites <= samples')
        for key in ('smoothing', 'min_std'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{key} must be a finite number')
        if not 0 <= self.smoothing < 1 or not 0 < self.min_std <= 1:
            raise ValueError('Require 0 <= smoothing < 1 and 0 < min_std <= 1')
        if not isinstance(self.project_shell, bool):
            raise ValueError('project_shell must be boolean')


def noise_variants() -> dict[str, NoiseSearchConfig]:
    """Explicit opt-in additions; never extend a registered campaign implicitly."""
    return {
        'noise_cem': NoiseSearchConfig(),
        'noise_cem_shell': NoiseSearchConfig(project_shell=True),
        'noise_random': NoiseSearchConfig(mode='noise_random'),
        'noise_random_shell': NoiseSearchConfig(mode='noise_random', project_shell=True),
        'noise_bon192': NoiseSearchConfig(mode='noise_random', samples=192, rounds=1),
        'noise_bon192_shell': NoiseSearchConfig(mode='noise_random', samples=192,
                                              rounds=1, project_shell=True),
    }


def shell_project(raw: Tensor, *, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor, int]:
    """Return (projected, raw_after_zero_resampling, count) for [B,N,H-1,D].

    Radius sqrt((H-1)*D) applies to the WHOLE interior, not each node. A zero
    vector has no direction and is redrawn from N(0,I), not divided by epsilon.
    These exceptional replacement raw vectors are also used in the elite fit.
    Scaling by max-abs avoids overflow when computing finite vectors' norms.
    """
    if raw.ndim != 4 or min(raw.shape) < 1 or not raw.is_floating_point():
        raise ValueError('Expected nonempty floating [B,N,H-1,D] noise')
    finite(raw, 'shell input')
    x = raw.detach().clone()
    flat = x.flatten(2)
    replaced = 0
    for _ in range(8):
        zeros = flat.abs().amax(-1) == 0
        count = int(zeros.sum())
        if not count:
            break
        replaced += count
        flat[zeros] = torch.randn((count, flat.shape[-1]), dtype=x.dtype,
                                  device=x.device, generator=generator)
    else:
        if (flat.abs().amax(-1) == 0).any():
            raise FloatingPointError('Could not resample a nonzero noise direction')
    finite(x, 'resampled shell input')
    scaled = flat / flat.abs().amax(-1, keepdim=True)
    unit = scaled / scaled.norm(dim=-1, keepdim=True)
    projected = unit * math.sqrt(flat.shape[-1])
    return finite(projected.reshape_as(x), 'shell projection'), x, replaced


def fit_elites(raw: Tensor, costs: Tensor, mean: Tensor, std: Tensor,
               cfg: NoiseSearchConfig) -> tuple[Tensor, Tensor]:
    """Independent per-environment diagonal-Gaussian fit, no batch pooling.

    raw[B,N,H-1,D], costs[B,N], mean/std[B,1,H-1,D]. Stable ties select the
    earlier candidate. Population variance uses divisor E, including E=1.
    """
    if (raw.ndim != 4 or costs.shape != raw.shape[:2] or
            mean.shape != (raw.shape[0], 1, *raw.shape[2:]) or std.shape != mean.shape or
            cfg.elites > raw.shape[1]):
        raise ValueError('Invalid elite-fit shapes')
    finite(raw, 'elite proposals'); finite(costs, 'elite costs')
    finite(mean, 'search mean'); finite(std, 'search std')
    if (std <= 0).any():
        raise ValueError('Search standard deviations must be positive')
    order = torch.argsort(costs, dim=1, stable=True)[:, :cfg.elites]
    rows = torch.arange(len(raw), device=raw.device)[:, None]
    elite = raw[rows, order]
    estimate = elite.mean(1, keepdim=True)
    variance = (elite - estimate).square().mean(1, keepdim=True)
    alpha = cfg.smoothing
    new_mean = alpha * mean + (1 - alpha) * estimate
    new_std = (alpha * std.square() + (1 - alpha) * variance).sqrt().clamp_min(cfg.min_std)
    return finite(new_mean, 'fitted mean'), finite(new_std, 'fitted std')


def search(start: Tensor, goal: Tensor, ops: PlannerOps, cfg: NoiseSearchConfig,
           *, generator: torch.Generator | None = None,
           initial_noise: Tensor | None = None) -> dict:
    """Evaluate R complete populations and return a cached best candidate.

    Models are frozen by the existing adapters. This routine uses no autograd.
    initial_noise is the RAW first-round pool [B,N,H-1,D]; later rounds use the
    supplied RNG. Each call resets its distribution. No warm start, own-model
    labels or environment rollouts are used. Deadline exceptions propagate;
    an incumbent is never used to execute after a caller's deadline.
    """
    if (start.ndim != 2 or goal.shape != start.shape or min(start.shape) < 1 or
            goal.device != start.device or goal.dtype != start.dtype or
            start.dtype not in (torch.float32, torch.float64)):
        raise ValueError('Require matching nonempty FP32/FP64 endpoints [B,D]')
    finite(start, 'start'); finite(goal, 'goal')
    b, d = start.shape
    shape = (b, cfg.samples, cfg.horizon - 1, d)
    if initial_noise is not None:
        if (initial_noise.shape != shape or initial_noise.device != start.device or
                initial_noise.dtype != start.dtype):
            raise ValueError(f'initial_noise must have shape {shape} and endpoint dtype/device')
        finite(initial_noise, 'initial noise')
    sampler_cfg = RefinementConfig(mode='euler', samples=cfg.samples,
                                   steps=cfg.steps, horizon=cfg.horizon)
    with torch.inference_mode(False), torch.no_grad():
        mean = start.new_zeros(b, 1, cfg.horizon - 1, d)
        std = torch.ones_like(mean)
        best_cost = start.new_full((b,), float('inf'))
        best_actions = best_noise = best_raw = best_indices = None
        cost_history, round_minima, incumbent_history = [], [], []
        noise_norms, fit_stds = [], []
        resampled = 0
        rows = torch.arange(b, device=start.device)
        for r in range(cfg.rounds):
            ops.check()
            if r == 0 and initial_noise is not None:
                raw = initial_noise.detach().clone()
            else:
                eps = torch.randn(shape, dtype=start.dtype, device=start.device, generator=generator)
                raw = mean + std * eps if cfg.mode == 'noise_cem' else eps
            finite(raw, 'noise proposals')
            if cfg.project_shell:
                noise, raw, count = shell_project(raw, generator=generator)
                resampled += count
            else:
                noise = raw
            paths = sample(start, goal, ops, sampler_cfg, initial_noise=noise)
            out = select(paths, ops)  # One decode/rollout per candidate, no rescoring.
            costs = finite(out['all_costs'], 'candidate costs')
            index = out['selected_indices']
            improve = out['costs'] < best_cost  # Earliest round wins exact ties.
            if best_actions is None:
                best_actions = out['actions'].clone()
                best_noise = noise[rows, index].clone()
                best_raw = raw[rows, index].clone()
                best_indices = index.clone()
            else:
                best_actions[improve] = out['actions'][improve]
                best_noise[improve] = noise[rows, index][improve]
                best_raw[improve] = raw[rows, index][improve]
                best_indices[improve] = (r * cfg.samples + index)[improve]
            best_cost = torch.minimum(best_cost, out['costs'])
            cost_history.append(costs.clone())
            round_minima.append(out['costs'].clone())
            incumbent_history.append(best_cost.clone())
            noise_norms.append(noise.flatten(2).norm(dim=-1).mean(1))
            if cfg.mode == 'noise_cem' and r + 1 < cfg.rounds:
                mean, std = fit_elites(raw, costs, mean, std, cfg)
            fit_stds.append(std.flatten(1).min(-1).values)
            ops.check()
        diagnostics = {
            'rounds_completed': cfg.rounds,
            'candidates_per_environment': cfg.rounds * cfg.samples,
            'elite_fit_space': 'raw_gaussian_proposal',
            'project_shell': cfg.project_shell,
            'zero_norm_resamples': resampled,
            'round_best_costs': torch.stack(round_minima, 1).cpu().tolist(),
            'incumbent_costs': torch.stack(incumbent_history, 1).cpu().tolist(),
            'mean_input_noise_norm': torch.stack(noise_norms, 1).cpu().tolist(),
            'min_search_std': torch.stack(fit_stds, 1).cpu().tolist(),
        }
        ops.check()
        return {'actions': best_actions, 'costs': best_cost,
                'all_costs': torch.cat(cost_history, 1), 'selected_indices': best_indices,
                'selected_noise': best_noise, 'selected_raw_noise': best_raw,
                'noise_search': diagnostics}


def plan(start: Tensor, goal: Tensor, ops: PlannerOps,
         cfg: RefinementConfig | NoiseSearchConfig, *, initial_noise: Tensor | None = None,
         generator: torch.Generator | None = None,
         perturb_generator: torch.Generator | None = None) -> dict:
    """Single dispatch point used by native and shared-execution adapters."""
    if isinstance(cfg, NoiseSearchConfig):
        return search(start, goal, ops, cfg, generator=generator, initial_noise=initial_noise)
    if not isinstance(cfg, RefinementConfig):
        raise TypeError('Expected RefinementConfig or NoiseSearchConfig')
    return select(sample(start, goal, ops, cfg, initial_noise=initial_noise,
                         generator=generator, perturb_generator=perturb_generator), ops)
