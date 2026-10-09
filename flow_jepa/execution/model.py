"""Short multimodal inverse control with observed-prefix supervision."""
import math

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..data import Segments
from ..models import Trunk, prediction_loss


class ExecutionSegments(Segments):
    def __getitem__(self, index):
        # Identical task, episode and start draws to the previous 60-step head.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        rows = self.tasks[self.names[int(rng.integers(len(self.names)))]]
        row = rows[int(rng.integers(len(rows)))]
        max_start = row['steps'] - self.horizon
        if row.get('first_success_action') is not None:
            max_start = min(max_start, row['first_success_action'] // self.c['data']['action_repeat'])
        start = int(rng.integers(max_start + 1))
        goal_rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index), 1701]))
        delta = int(goal_rng.choice([5, 10, 20, 40, 60]))
        with h5py.File(self.root / row['path'], 'r') as f:
            states = np.stack([f[self.state_key][start + j] for j in (0, 1, 5, delta)])
            states = (states.astype(np.float32) - self.mean) / self.std
            actions = f['actions'][start:start + 5].astype(np.float32)
        return dict(z=torch.from_numpy(states), a=torch.from_numpy(actions))


class ChunkPolicy(nn.Module):
    def __init__(self, c):
        super().__init__()
        cfg = c['controller_grounded']
        self.k, self.ad, self.mixtures = 5, 8, cfg['mixtures']
        self.input = nn.Linear(c['encoder']['dim'], cfg['width'])
        self.types = nn.Parameter(torch.randn(3, cfg['width']) * 0.02)
        self.trunk = Trunk(cfg['width'], cfg['depth'], 4)
        self.output = nn.Linear(cfg['width'], self.mixtures * (1 + 2 * self.k * self.ad))

    def distribution(self, start, target, goal):
        x = torch.cat([self.input(v) + self.types[i] for i, v in enumerate((start, target, goal))], 1)
        h = self.trunk(x).mean(1)
        raw = self.output(h).reshape(-1, self.mixtures, 81)
        logits, means = raw[:, :, 0], raw[:, :, 1:41].tanh()
        scales = 0.05 + 0.45 * raw[:, :, 41:].sigmoid()
        return logits, means, scales

    def forward(self, batch, world=None, weight=0.0):
        z, actions = batch['z'], batch['a'].flatten(1)
        logits, means, scales = self.distribution(z[:, 0], z[:, 2], z[:, 3])
        logp = -0.5 * (((actions[:, None] - means) / scales).square()
                       + 2 * scales.log() + math.log(2 * math.pi)).sum(-1)
        posterior = F.log_softmax(logits, -1) + logp
        nll = -torch.logsumexp(posterior, -1).mean() / 40
        loss, metrics = nll, dict(action_nll=nll.detach())
        if weight:
            # A demonstrated action chooses the supervised mode; generated actions
            # seek OBSERVED next/endpoint states. This is not a true outcome label
            # for those generated actions, and the frozen world can still err.
            mode = posterior.detach().argmax(-1)
            proposed = means[torch.arange(len(z), device=z.device), mode].reshape(-1, 5, 8)
            predicted = world.rollout(z[:, 0], proposed)
            prefix = prediction_loss(predicted[:, 0], z[:, 1])
            endpoint = prediction_loss(predicted[:, -1], z[:, 2])
            execution = 0.5 * (prefix + endpoint)
            loss = loss + weight * execution
            metrics.update(prefix_loss=prefix.detach(), endpoint_loss=endpoint.detach())
        return loss, metrics

    @torch.no_grad()
    def propose(self, start, target, goal, count):
        logits, means, scales = self.distribution(start, target, goal)
        modes = torch.multinomial(logits.softmax(-1), count, replacement=True)
        index = modes[:, :, None].expand(-1, -1, 40)
        actions = means.gather(1, index) + scales.gather(1, index) * torch.randn(
            len(start), count, 40, device=start.device)
        best = logits.argmax(-1)
        actions[:, 0] = means[torch.arange(len(start), device=start.device), best]
        return actions.reshape(len(start), count, 5, 8).clamp(-1, 1)
