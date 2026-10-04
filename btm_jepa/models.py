"""Matched FM, BTM terminal-map, and deterministic waypoint generators.

Input: path interiors [B, H-1, D], endpoints [B, D], physical spacing [B].
Only FM receives the artificial interpolation time. All models receive physical
spacing (in action blocks), which must not be confused with diffusion time.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class AttentionBlock(nn.Module):
    """Explicit attention supports forward-mode AD on both CPU and CUDA.

    Flash/efficient SDPA kernels in the pinned torch version do not all implement
    forward AD. This small waypoint transformer deliberately uses ordinary ops.
    """

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        weights = (q @ k.transpose(-1, -2) / math.sqrt(d // self.heads)).softmax(-1)
        x = x + self.proj((weights @ v).transpose(1, 2).reshape(b, n, d))
        return x + self.mlp(self.norm2(x))


class PathModel(nn.Module):
    def __init__(self, latent_dim: int, method: str = "btm", hidden_dim: int = 512,
                 depth: int = 4, heads: int = 8, max_horizon: int = 20):
        super().__init__()
        if method not in {"btm", "flow", "deterministic"}:
            raise ValueError(f"Unknown method: {method}")
        if hidden_dim % heads or max_horizon < 2:
            raise ValueError("hidden_dim must divide heads; max_horizon must be >=2")
        self.arch = dict(latent_dim=latent_dim, method=method, hidden_dim=hidden_dim,
                         depth=depth, heads=heads, max_horizon=max_horizon)
        self.method = method
        self.token = nn.Linear(latent_dim, hidden_dim)
        self.start, self.goal = nn.Linear(latent_dim, hidden_dim), nn.Linear(latent_dim, hidden_dim)
        self.spacing = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.time = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)) if method == "flow" else None
        self.position = nn.Parameter(torch.randn(1, max_horizon - 1, hidden_dim) * 0.02)
        self.blocks = nn.Sequential(*[AttentionBlock(hidden_dim, heads) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, latent_dim))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, start, goal, spacing, time=None):
        if x.shape[1] > self.position.shape[1] or x.shape[1] < 1:
            raise ValueError("Path must have 1..max_horizon-1 interior states")
        c = self.start(start) + self.goal(goal) + self.spacing(torch.log1p(spacing[:, None]))
        if self.method == "flow":
            if time is None:
                raise ValueError("FM requires interpolation time")
            c = c + self.time(time[:, None])
        elif time is not None:
            raise ValueError("BTM/deterministic predictors do not accept diffusion time")
        h = self.token(x) + self.position[:, :x.shape[1]] + c[:, None]
        y = self.out(self.blocks(h))
        return x + y if self.method == "btm" else y


def btm_loss(model, target, start, goal, spacing, *, noise=None, time=None, boundary_weight=1.0):
    """Eq. (20), arXiv:2608.01692v3, conditional linear-interpolant version.

    Ltr = MSE(T(x), stop_gradient[T(x) + J_T(x) (target-noise)]).
    Squaring J_T d *without* this stopped target gives a DIFFERENT gradient.
    JVP is detached; backward needs no Hessian or second-order parameter graph.
    """
    noise = torch.randn_like(target) if noise is None else noise
    time = torch.rand(target.shape[0], device=target.device) if time is None else time
    x = torch.lerp(noise, target, time[:, None, None])
    direction = target - noise
    with torch.no_grad():
        value, tangent = torch.func.jvp(lambda u: model(u, start, goal, spacing), (x,), (direction,))
        stopped_target = (value + tangent).detach()
    pred = model(x, start, goal, spacing)
    transport = F.mse_loss(pred, stopped_target)
    boundary = F.mse_loss(model(target, start, goal, spacing), target)
    return transport + boundary_weight * boundary, {
        "transport_loss": transport.detach(), "boundary_loss": boundary.detach(),
        "jvp_rms": tangent.detach().square().mean().sqrt(),
    }


class PlannerTrainingModel(nn.Module):
    def __init__(self, path: PathModel, action_dim: int, inverse_hidden: int = 512):
        super().__init__()
        self.path = path
        d = path.arch["latent_dim"]
        self.inverse_arch = dict(action_dim=action_dim, hidden_dim=inverse_hidden)
        self.inverse = nn.Sequential(nn.Linear(3*d, inverse_hidden), nn.LayerNorm(inverse_hidden), nn.GELU(),
                                     nn.Linear(inverse_hidden, inverse_hidden), nn.LayerNorm(inverse_hidden), nn.GELU(),
                                     nn.Linear(inverse_hidden, action_dim))

    def inverse_actions(self, z, nxt):
        return self.inverse(torch.cat([z, nxt, nxt-z], -1))

    def forward(self, batch, boundary_weight=1.0, inverse_weight=1.0):
        z, spacing = batch["z_path"], batch["spacing"]
        target, start, goal = z[:, 1:-1], z[:, 0], z[:, -1]
        extras = {}
        if self.path.method == "btm":
            generative, extras = btm_loss(self.path, target, start, goal, spacing, boundary_weight=boundary_weight)
        elif self.path.method == "flow":
            noise, t = torch.randn_like(target), torch.rand(len(z), device=z.device)
            x = torch.lerp(noise, target, t[:, None, None])
            generative = F.mse_loss(self.path(x, start, goal, spacing, t), target-noise)
        else:
            generative = F.mse_loss(self.path(torch.zeros_like(target), start, goal, spacing), target)
        local = batch["local_z"]
        pred_actions = self.inverse_actions(local[:, :-1], local[:, 1:])
        inverse = F.mse_loss(pred_actions, batch["actions"])
        return {"loss": generative + inverse_weight * inverse, "generative_loss": generative.detach(),
                "inverse_loss": inverse.detach(), "pred_actions": pred_actions, **extras}


@torch.no_grad()
def sample_paths(model: PathModel, start, goal, *, horizon: int, spacing: float,
                 num_samples: int, flow_steps: int = 16, generator=None):
    if horizon < 2 or num_samples < 1 or spacing < 1 or flow_steps < 1:
        raise ValueError("Require horizon>=2, samples/spacing/flow_steps>=1")
    b, d = start.shape
    z0 = start[:, None].expand(b, num_samples, d).reshape(-1, d)
    zg = goal[:, None].expand(b, num_samples, d).reshape(-1, d)
    dt = start.new_full((len(z0),), spacing)
    shape = (len(z0), horizon-1, d)
    x = torch.randn(shape, device=start.device, generator=generator)
    if model.method == "flow":
        for i in range(flow_steps):
            t = start.new_full((len(z0),), i/flow_steps)
            x = x + model(x, z0, zg, dt, t) / flow_steps
    elif model.method == "btm":
        x = model(x, z0, zg, dt)
    else:
        x = model(torch.zeros_like(x), z0, zg, dt)
    return torch.cat([z0[:, None], x, zg[:, None]], 1).reshape(b, num_samples, horizon+1, d)
