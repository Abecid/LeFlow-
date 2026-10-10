"""LeFlow release modules with the paper's Appendix-E spatial adapter.

The MetaWorld adapter changes input/action dimensions only. It does not use
retrieval, action CEM refinement, execution-error corrections or our workspace.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .vendor.leflow import LatentPathFlow, InverseDynamics
from .vendor.leflow import flow_matching_loss, inverse_dynamics_loss


class SpatialAutoencoder(nn.Module):
    """Appendix E: positioned patches + summary token, two-layer encoder.

    The paper does not release its adapter decoder. The training-only decoder
    here is a positioned, two-layer Transformer; its exact choice is recorded
    as an implementation detail, not claimed to be the authors' source.
    """
    def __init__(self, dim=1024, patches=32, width=512):
        super().__init__()
        self.input = nn.Linear(dim, width)
        self.summary = nn.Parameter(torch.randn(1, 1, width) * .02)
        self.position = nn.Parameter(torch.randn(1, patches + 1, width) * .02)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            width, 8, 4 * width, dropout=0., activation='gelu', batch_first=True,
            norm_first=True), 2, norm=nn.LayerNorm(width), enable_nested_tensor=False)
        self.decoder_position = nn.Parameter(torch.randn(1, patches, width) * .02)
        self.decoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            width, 8, 4 * width, dropout=0., activation='gelu', batch_first=True,
            norm_first=True), 2, norm=nn.LayerNorm(width), enable_nested_tensor=False)
        self.output = nn.Linear(width, dim)

    def encode(self, z):
        x = torch.cat((self.summary.expand(len(z), -1, -1), self.input(z)), 1)
        return self.encoder(x + self.position)[:, 0]

    def forward(self, batch, world=None):
        z = batch['z'][:, 0]
        compact = self.encode(z)
        reconstructed = self.output(self.decoder(compact[:, None] + self.decoder_position))
        mse = F.mse_loss(reconstructed, z)
        cosine = (1 - F.cosine_similarity(reconstructed, z, dim=-1)).mean()
        loss = mse + cosine
        return loss, dict(reconstruction_mse=mse.detach(),
                          reconstruction_cosine=cosine.detach(),
                          compact_std=compact.std(0, unbiased=False).mean().detach())


class LeFlow(nn.Module):
    def __init__(self, c):
        super().__init__()
        cfg = c['baseline']['leflow']
        self.adapter = SpatialAutoencoder(c['encoder']['dim'], 32, 512)
        self.flow = LatentPathFlow(512, **cfg['flow'])
        self.inverse = InverseDynamics(512, 8, **cfg['inverse'])
        self.register_buffer('action_mean', torch.zeros(8))
        self.register_buffer('action_std', torch.ones(8))
        self.horizon = cfg['horizon']
        self.adapter_stage = True

    def set_stage(self, adapter):
        self.adapter_stage = adapter
        self.adapter.requires_grad_(adapter)
        self.flow.requires_grad_(not adapter)
        self.inverse.requires_grad_(not adapter)
        if not adapter:
            self.adapter.eval()

    def train(self, mode=True):
        super().train(mode)
        if not self.adapter_stage:
            self.adapter.eval()
        return self

    def forward(self, batch, world=None):
        if self.adapter_stage:
            return self.adapter(batch)
        z, actions = batch['z'], batch['a']
        with torch.no_grad():
            compact = self.adapter.encode(z.flatten(0, 1)).reshape(len(z), z.shape[1], 512)
        flow_loss = flow_matching_loss(self.flow, compact)
        inverse_loss, normalized = inverse_dynamics_loss(
            self.inverse, compact, (actions - self.action_mean) / self.action_std)
        # Match release step_batch: inverse actions on RECORDED paths, not
        # generated paths. The shared world's action units are physical units.
        predicted_actions = normalized * self.action_std + self.action_mean
        predicted = world(z[:, :-1].flatten(0, 1), predicted_actions.flatten(0, 1))
        predicted_compact = self.adapter.encode(predicted).reshape(len(z), self.horizon, 512)
        consistency = F.mse_loss(predicted_compact, compact[:, 1:])
        loss = flow_loss + inverse_loss + .1 * consistency
        return loss, dict(flow_loss=flow_loss.detach(), inverse_loss=inverse_loss.detach(),
                          consistency_loss=consistency.detach(),
                          action_outside_bounds=(predicted_actions.abs() > 1).float().mean().detach())

    @torch.no_grad()
    def sample_paths(self, start, goal, count, steps, budget=None):
        # Same Euler equations, independent Gaussian noise and endpoint clamping
        # as upstream LatentPlannerRuntime.sample_paths, with clock checks only.
        z0 = self.adapter.encode(start).expand(count, -1)
        zg = self.adapter.encode(goal).expand(count, -1)
        x = torch.randn(count, self.horizon - 1, 512, device=start.device, dtype=start.dtype)
        for i in range(steps):
            if budget is not None:
                budget.check()
            t = torch.full((count,), i / steps, device=start.device, dtype=start.dtype)
            x = x + self.flow(x, t, z0, zg) / steps
        return torch.cat((z0[:, None], x, zg[:, None]), 1)


class LeFlowController:
    def __init__(self, model, world, c):
        self.model, self.world, self.c = model, world, c
        self.execution_blocks = c['baseline']['leflow']['receding_horizon']

    @torch.no_grad()
    def plan(self, start, goal, *, budget=None):
        cfg = self.c['baseline']['leflow']
        paths = self.model.sample_paths(start, goal, cfg['candidates'], cfg['flow_steps'], budget)
        actions = self.model.inverse(paths[:, :-1].flatten(0, 1), paths[:, 1:].flatten(0, 1))
        actions = actions.reshape(len(paths), self.model.horizon, 8)
        actions = actions * self.model.action_std + self.model.action_mean
        end = self.world.rollout(start.expand(len(paths), -1, -1), actions, budget=budget)[:, -1]
        # Appendix E scores ORIGINAL spatial-world predictions, not clamped
        # generated endpoints or reconstruction-decoded compact plans.
        costs = (end - goal).square().mean((1, 2))
        index = int(costs.argmin())
        return actions[index, :self.execution_blocks].flatten(), goal[0], float(costs[index]), self.model.horizon
