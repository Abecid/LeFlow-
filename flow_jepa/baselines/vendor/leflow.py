# Extracted unchanged from hsiangwei0903/LeFlow, commit
# f1fe192e41ec6f20de25cd1054ede40a8cbdcfa5. See leflow-LICENSE.
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0, device=t.device))
            * torch.arange(half, device=t.device)
            / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class LatentPathFlow(nn.Module):
    """Rectified-flow velocity model for latent path interiors."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 512,
        depth: int = 4,
        max_horizon: int = 20,
        time_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_horizon = max_horizon
        self.depth = depth
        self.time_dim = time_dim
        self.dropout = dropout
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_horizon - 1, hidden_dim) * 0.02
        )
        self.token_proj = nn.Linear(latent_dim, hidden_dim)
        self.start_proj = nn.Linear(latent_dim, hidden_dim)
        self.goal_proj = nn.Linear(latent_dim, hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.net = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, latent_dim))

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        z_start: torch.Tensor,
        z_goal: torch.Tensor,
    ) -> torch.Tensor:
        n_tokens = x_t.size(1)
        if n_tokens > self.max_horizon - 1:
            raise ValueError(
                f"Requested {n_tokens + 1} horizon, but max_horizon={self.max_horizon}"
            )
        cond = (
            self.start_proj(z_start)
            + self.goal_proj(z_goal)
            + self.time_embed(t.float())
        )
        x = self.token_proj(x_t)
        x = x + self.pos_embedding[:, :n_tokens] + cond[:, None]
        return self.out(self.net(x))


class InverseDynamics(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        depth: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout = dropout
        layers: list[nn.Module] = []
        in_dim = latent_dim * 3
        for i in range(depth):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_t, z_next, z_next - z_t], dim=-1)
        return self.net(x)


def flow_matching_loss(
    flow: LatentPathFlow,
    z_path: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    z_start = z_path[:, 0]
    z_goal = z_path[:, -1]
    target = z_path[:, 1:-1]
    noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
    t = torch.rand(z_path.size(0), device=z_path.device, dtype=z_path.dtype, generator=generator)
    x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * target
    pred_v = flow(x_t, t, z_start, z_goal)
    return F.mse_loss(pred_v, target - noise)


def inverse_dynamics_loss(
    inverse_dynamics: InverseDynamics,
    z_path: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = inverse_dynamics(z_path[:, :-1].reshape(-1, z_path.size(-1)), z_path[:, 1:].reshape(-1, z_path.size(-1)))
    pred = pred.reshape(z_path.size(0), z_path.size(1) - 1, -1)
    return F.mse_loss(pred, actions), pred
