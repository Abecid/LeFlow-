"""Tensor-explicit JEPA dynamics, joint plans, and two literature adaptations."""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def positions(n, width, device, dtype):
    p = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    w = torch.exp(torch.arange(0, width, 2, device=device) * (-math.log(10000) / width))
    out = torch.zeros(n, width, device=device)
    out[:, 0::2], out[:, 1::2] = (p * w).sin(), (p * w).cos()
    return out.to(dtype)


class Trunk(nn.Module):
    def __init__(self, width, depth, heads):
        super().__init__()
        self.net = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                width,
                heads,
                4 * width,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            depth,
            enable_nested_tensor=False,
        )
        # TransformerEncoder clones identical initial weights. Reinitialize each.
        for layer in self.net.layers:
            for name, p in layer.named_parameters():
                if p.ndim > 1:
                    nn.init.xavier_uniform_(p)
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        return self.norm(
            self.net(x + positions(x.shape[1], x.shape[2], x.device, x.dtype))
        )


def distance(x, y):
    # [B,...,P,D] -> [B,...]; cosine is computed per spatial/temporal token.
    return (1 - F.cosine_similarity(x.float(), y.float(), dim=-1, eps=1e-8)).mean(-1)


def prediction_loss(x, y):
    return (x.float() - y.float()).square().mean() + distance(x, y).mean()


class Dynamics(nn.Module):
    def __init__(self, dim, action_dim, width=256, depth=4, heads=4):
        super().__init__()
        self.z = nn.Linear(dim, width)
        self.a = nn.Linear(action_dim, width)
        self.trunk = Trunk(width, depth, heads)
        self.out = nn.Linear(width, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.calls = 0

    def forward(self, z, a):
        self.calls += len(z)
        return z + self.out(self.trunk(self.z(z) + self.a(a)[:, None]))

    def rollout(self, z, actions):
        result = []
        for i in range(actions.shape[1]):
            z = self(z, actions[:, i])
            result.append(z)
        return torch.stack(result, 1)


class JointPlanner(nn.Module):
    def __init__(
        self, dim, action_dim, segments, chunk_steps, width=256, depth=4, heads=4
    ):
        super().__init__()
        self.dim, self.action_dim = dim, action_dim
        self.segments, self.chunk_steps = segments, chunk_steps
        self.zin = nn.Linear(dim, width)
        self.ain = nn.Linear(action_dim * chunk_steps, width)
        self.time = nn.Sequential(
            nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.types = nn.Parameter(torch.randn(4, width) * 0.02)
        self.trunk = Trunk(width, depth, heads)
        self.zout, self.aout = (
            nn.Linear(width, dim),
            nn.Linear(width, action_dim * chunk_steps),
        )
        for out in (self.zout, self.aout):
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)

    def forward(self, z, actions, start, goal, time):
        b, m, p, _ = z.shape
        ztokens = self.zin(z).flatten(1, 2) + self.types[0]
        atokens = self.ain(actions.flatten(2)) + self.types[1]
        context = torch.cat(
            (self.zin(start) + self.types[2], self.zin(goal) + self.types[3]), 1
        )
        tokens = torch.cat((context, ztokens, atokens), 1)
        h = self.trunk(tokens + self.time(time[:, None])[:, None])[:, 2 * p :]
        return self.zout(h[:, : m * p]).reshape(b, m, p, self.dim), self.aout(
            h[:, m * p :]
        ).reshape_as(actions)

    def sample(
        self, start, goal, *, steps=8, deterministic=False, path_only=False, noise=None
    ):
        b, p, d = start.shape
        m, k, a = self.segments, self.chunk_steps, self.action_dim
        if noise is None:
            z = torch.randn(b, m - 1, p, d, device=start.device)
            u = torch.randn(b, m, k, a, device=start.device)
        else:
            z, u = noise
        if deterministic:
            return self(
                torch.zeros_like(z),
                torch.zeros_like(u),
                start,
                goal,
                start.new_zeros(b),
            )
        for i in range(steps):
            if path_only:
                u = torch.zeros_like(u)
            vz, vu = self(z, u, start, goal, start.new_full((b,), i / steps))
            z, u = z + vz / steps, u + vu / steps
        return z, u


class Inverse(nn.Module):
    def __init__(self, dim, action_dim, chunk_steps, width=256, heads=4):
        super().__init__()
        self.inp = nn.Linear(3 * dim, width)
        self.trunk = Trunk(width, 2, heads)
        self.out = nn.Linear(width, action_dim * chunk_steps)
        self.chunk_steps, self.action_dim = chunk_steps, action_dim

    def forward(self, start, goal):
        h = self.trunk(self.inp(torch.cat([start, goal, goal - start], -1)))
        return self.out(h.mean(1)).reshape(
            len(start), self.chunk_steps, self.action_dim
        )


class System(nn.Module):
    def __init__(self, c, method):
        super().__init__()
        self.method = method
        d, m = c["encoder"]["dim"], c["model"]
        a = 4 * c["data"]["action_repeat"]
        kwargs = {k: m[k] for k in ("width", "depth", "heads")}
        if method == "world":
            self.world = Dynamics(d, a, **kwargs)
        elif method == "hwm_adapted":
            self.macro_encoder = nn.Sequential(
                nn.Linear(a * m["chunk_steps"], m["width"]),
                nn.GELU(),
                nn.Linear(m["width"], m["macro_dim"]),
                nn.Tanh(),
            )
            self.coarse = Dynamics(d, m["macro_dim"], **kwargs)
        else:
            self.planner = JointPlanner(d, a, m["segments"], m["chunk_steps"], **kwargs)
            if method == "leflow_adapted":
                self.inverse = Inverse(d, a, m["chunk_steps"], m["width"], m["heads"])

    def forward(
        self,
        batch,
        world=None,
        *,
        consistency_weight=0.0,
        consistency_batch=2,
        consistency_steps=4,
    ):
        z, a = batch["z"], batch["a"]
        if self.method == "world":
            loss = prediction_loss(self.world.rollout(z[:, 0], a), z[:, 1:])
            return loss, {"dynamics": loss.detach()}
        if self.method == "hwm_adapted":
            pred = self.coarse(z[:, 0], self.macro_encoder(a.flatten(1)))
            loss = prediction_loss(pred, z[:, -1])
            return loss, {"coarse_dynamics": loss.detach()}
        start, goal, target = z[:, 0], z[:, -1], z[:, 1:-1]
        det, path_only = "deterministic" in self.method, self.method == "leflow_adapted"
        t = torch.rand(len(z), device=z.device)
        nz, na = torch.randn_like(target), torch.randn_like(a)
        xz = torch.lerp(nz, target, t[:, None, None, None])
        xa = torch.lerp(na, a, t[:, None, None, None])
        if det:
            xz, xa, t = torch.zeros_like(xz), torch.zeros_like(xa), torch.zeros_like(t)
        if path_only:
            xa = torch.zeros_like(xa)
        vz, va = self.planner(xz, xa, start, goal, t)
        zloss = F.mse_loss(vz, target if det else target - nz)
        aloss = F.mse_loss(va, a if det else a - na) if not path_only else va.sum() * 0
        loss = zloss + aloss
        metrics = dict(state_objective=zloss.detach(), action_objective=aloss.detach())
        if path_only:
            pred_a = self.inverse(batch["local"][:, 0], batch["local"][:, 1])
            inverse_loss = F.mse_loss(pred_a, a[:, 0])
            # Published family uses observed-transition consistency, shared frozen world.
            observed = prediction_loss(
                world.rollout(batch["local"][:, 0], pred_a.clamp(-1, 1))[:, -1],
                batch["local"][:, 1],
            )
            loss = loss + inverse_loss + 0.1 * observed
            metrics.update(
                inverse=inverse_loss.detach(), observed_consistency=observed.detach()
            )
        if consistency_weight:
            n = min(consistency_batch, len(z))
            gz, ga = self.planner.sample(
                start[:n], goal[:n], steps=consistency_steps, deterministic=det
            )
            path = torch.cat([start[:n, None], gz, goal[:n, None]], 1)
            # Each local bridge starts at its generated state; never teacher-force
            # observed interiors here. Frozen dynamics retains input derivatives.
            b, m, k, ad = ga.shape
            predicted = world.rollout(
                path[:, :-1].flatten(0, 1), ga.clamp(-1, 1).reshape(b * m, k, ad)
            )[:, -1]
            reach = prediction_loss(predicted, path[:, 1:].flatten(0, 1))
            loss = loss + consistency_weight * reach
            metrics["generated_consistency"] = reach.detach()
        return loss, metrics
