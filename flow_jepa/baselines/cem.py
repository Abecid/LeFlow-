"""Released stable-worldmodel CEM with only the shared-world interface adapter."""
from types import SimpleNamespace
import inspect
import json
from pathlib import Path
from importlib.metadata import version

from ..common import file_hash

import numpy as np
import torch
from gymnasium.spaces import Box
from stable_worldmodel.solver import CEMSolver


def verify_solver():
    fingerprint = file_hash(inspect.getfile(CEMSolver))
    expected = 'd88c86dcd1bd1e6d89221ac22079a3efe296cc7566532de0d36605e2f1536050'
    assert version('stable-worldmodel') == '0.0.6' and fingerprint == expected
    return dict(package='stable-worldmodel', version='0.0.6', cem_sha256=fingerprint)


class SpatialWorldCost:
    def __init__(self, world, mean, std):
        self.world, self.mean, self.std = world, mean, std
        self.budget = None

    def get_cost(self, info, actions):
        b, n, h, d = actions.shape
        physical = actions.reshape(b*n, h, d) * self.std + self.mean
        start = info['start'].flatten(0, 1)
        goal = info['goal'].flatten(0, 1)
        end = self.world.rollout(start, physical, budget=self.budget)[:, -1]
        return (end-goal).square().mean((1, 2)).reshape(b, n)


class ReleasedCEMController:
    def __init__(self, model, world, c):
        verify_solver()
        self.world, self.c = world, c
        cfg = c['baseline']['cem']
        self.execution_blocks = cfg['receding_horizon']
        device = next(world.parameters()).device
        meta = json.loads((Path(c['baseline']['packed_cache'])/'complete.json').read_text())
        self.mean = torch.tensor(meta['action_mean'], device=device)
        self.std = torch.tensor(meta['action_std'], device=device)
        self.cost = SpatialWorldCost(world, self.mean, self.std)
        self.solver = CEMSolver(self.cost, batch_size=1, num_samples=cfg['candidates'],
            var_scale=cfg['var_scale'], n_steps=cfg['iterations'], topk=cfg['elites'],
            device=device, seed=3072)
        self.solver.configure(action_space=Box(-1., 1., shape=(1, 4), dtype=np.float32),
            n_envs=1, config=SimpleNamespace(horizon=cfg['horizon'], action_block=2))

    def begin_episode(self):
        # Common benchmark's per-reset seed, independent of GPU scheduling.
        self.solver.torch_gen.manual_seed(torch.initial_seed())

    @torch.no_grad()
    def plan(self, start, goal, *, budget=None):
        self.cost.budget = budget
        outputs = self.solver(dict(start=start, goal=goal), init_action=None)
        actions = outputs['actions'][0].to(start.device) * self.std + self.mean
        return actions[:self.execution_blocks].flatten(), goal[0], float(outputs['costs'][0]), self.c['baseline']['cem']['horizon']
