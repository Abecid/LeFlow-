"""Tests of baseline isolation, released equations and production dimensions."""
import ast
import json
import hashlib
import tempfile
from unittest.mock import patch

import h5py
import numpy as np
from pathlib import Path
import unittest

import torch
from torch import nn

from flow_jepa.baselines.leflow import LeFlow, LeFlowController
from flow_jepa.baselines.vendor.leflow import LatentPathFlow, flow_matching_loss


class BaselineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(3072)
        self.c = json.loads(Path('config/flow_metaworld_leflow_release.json').read_text())

    def test_no_method_spillover(self):
        for path in Path('flow_jepa/baselines').glob('*.py'):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse('execution' in (node.module or '').split('.'), str(path))
        for key in ['controller_grounded', 'latent_revision', 'execution_revision']:
            self.assertNotIn(key, self.c)

    def test_flow_has_no_fixed_low_rank_output_restriction(self):
        flow = LatentPathFlow(512)
        output = flow.out[-1]
        self.assertEqual(tuple(output.weight.shape), (512, 512))
        augmented = torch.cat((output.weight.detach(), output.bias.detach()[:, None]), 1)
        self.assertEqual(int(torch.linalg.matrix_rank(augmented)), 512)

    def test_released_flow_objective_and_noise(self):
        flow = LatentPathFlow(16, hidden_dim=16, depth=1)
        z = torch.randn(2, 6, 16)
        g = torch.Generator().manual_seed(99)
        actual = flow_matching_loss(flow, z, generator=g)
        g.manual_seed(99)
        noise = torch.randn(2, 4, 16, generator=g)
        t = torch.rand(2, generator=g)
        interpolated = (1-t[:, None, None])*noise+t[:, None, None]*z[:, 1:-1]
        expected = (flow(interpolated, t, z[:, 0], z[:, -1])-(z[:, 1:-1]-noise)).square().mean()
        self.assertTrue(torch.equal(actual, expected))
        actual.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in flow.parameters() if p.grad is not None))

    def test_sampler_matches_released_euler_and_fixed_endpoints(self):
        model = LeFlow(self.c).eval()
        # Exercise the actual 512-dimensional sampler with a simple deterministic
        # velocity to make differences from release Euler integration observable.
        class Adapter(nn.Module):
            def encode(self, x): return x[:, 0, :512]
        class Velocity(nn.Module):
            def forward(self, x, t, a, b): return b[:, None]-x
        model.adapter, model.flow = Adapter(), Velocity()
        start, goal = torch.randn(1, 32, 1024), torch.randn(1, 32, 1024)
        torch.manual_seed(88)
        paths = model.sample_paths(start, goal, 3, 16)
        torch.manual_seed(88)
        expected = torch.randn(3, 4, 512)
        for _ in range(16): expected = expected+(goal[:, :1, :512]-expected)/16
        self.assertTrue(torch.equal(paths[:, 1:-1], expected))
        self.assertTrue(torch.equal(paths[:, 0], start[:, 0, :512].expand(3, -1)))
        self.assertTrue(torch.equal(paths[:, -1], goal[:, 0, :512].expand(3, -1)))

    def test_ranking_uses_continuous_world_outcome_and_no_refinement(self):
        class Model(nn.Module):
            horizon = 5
            action_mean = torch.zeros(8)
            action_std = torch.ones(8)
            def sample_paths(self, start, goal, count, steps, budget):
                return torch.zeros(2, 6, 512)
            def inverse(self, start, end):
                return torch.cat((torch.ones(5, 8), torch.zeros(5, 8)))
        class World:
            def __init__(self): self.calls = 0
            def rollout(self, start, actions, budget=None):
                self.calls += len(start)*actions.shape[1]
                return start[:, None]+actions.cumsum(1).mean(-1)[:, :, None, None]
        world = World(); ctl = LeFlowController(Model(), world, self.c)
        a, _, score, _ = ctl.plan(torch.zeros(1, 32, 1024), torch.zeros(1, 32, 1024))
        self.assertEqual(score, 0.)
        self.assertTrue(torch.equal(a, torch.zeros(40)))
        self.assertEqual(world.calls, 10)

    def test_adapter_freezes_during_planner_learning(self):
        model = LeFlow(self.c)
        model.set_stage(False); model.train()
        self.assertFalse(model.adapter.training)
        self.assertFalse(any(p.requires_grad for p in model.adapter.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.flow.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.inverse.parameters()))


    def test_vendored_symbols_are_verbatim_upstream(self):
        root = Path('flow_jepa/baselines/vendor')
        meta = json.loads((root / 'provenance.json').read_text())
        src = (root / 'leflow.py').read_text()
        hashes = {node.name: hashlib.sha256(ast.get_source_segment(src, node).encode()).hexdigest()
                  for node in ast.parse(src).body if hasattr(node, 'name')}
        self.assertEqual(hashes, meta['symbol_sha256'])

    def run_evaluation_fixture(self, blocks, success_at=None, timeout=False):
        from flow_jepa.evaluate import evaluate
        from flow_jepa.common import digest
        from flow_jepa.budget import PlanningBudgetExceeded
        c = json.loads(json.dumps(self.c))
        c['training_tasks'], c['heldout_tasks'] = ['fixture'], []
        c['evaluation']['budget_primitive'] = 22
        class World:
            calls = 0
        class Encoder:
            fingerprint = 'fixture-encoder'
            def __call__(self, x): return torch.ones(1, 2, 8)
        class Controller:
            execution_blocks = blocks
            calls = 0
            def plan(self, z, goal, budget=None):
                self.calls += 1
                if timeout: raise PlanningBudgetExceeded('fixture')
                return torch.zeros(blocks*8), goal[0], 0., blocks
        class Env:
            steps = 0
            def reset(self): pass
            def render(self): return np.zeros((4, 4, 3), dtype=np.uint8)
            def close(self): pass
            def step(self, a):
                self.steps += 1
                return None, 1., False, False, dict(success=self.steps == success_at)
        ctl, env = Controller(), Env()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with h5py.File(root / 'episode.h5', 'w') as f:
                f['goal'] = np.ones((2, 8), dtype=np.float32)
                f['initial_rgb'] = env.render()
            row = dict(id='validation/fixture/00000', task='fixture', seed=3072,
                       split='validation', index=0, path='episode.h5', sha256='fixture',
                       expert_success=True)
            (root / 'manifest.json').write_text(json.dumps(dict(protocol=digest(c),
                encoder=Encoder.fingerprint, mean=0., std=1., entries=[row])))
            with patch('flow_jepa.evaluate.make_env', return_value=env):
                records, _ = evaluate(None, World(), c, root, 'leflow_release',3072,
                    'validation', Encoder(), torch.device('cpu'), count=1, controller=ctl,
                    distributed_context=(0, 1, torch.device('cpu')))
        return records[0], ctl

    def test_released_execution_prefix_keeps_primitive_cap(self):
        for blocks, plans in [(1, 11), (5, 3)]:
            row, ctl = self.run_evaluation_fixture(blocks)
            self.assertEqual(row['steps'], 22)
            self.assertEqual(ctl.calls, plans)
            self.assertFalse(row['success'])

    def test_success_inside_multi_block_prefix(self):
        row, ctl = self.run_evaluation_fixture(5, success_at=13)
        self.assertEqual(row['first_success_primitive'], 13)
        self.assertEqual(row['steps'], 13)
        self.assertEqual(ctl.calls, 2)

    def test_timed_out_plan_executes_no_action(self):
        row, _ = self.run_evaluation_fixture(5, timeout=True)
        self.assertEqual(row['steps'], 0)
        self.assertTrue(row['controller_budget_exhausted'])


    def test_released_cem_matches_equations_and_returns_elite_mean(self):
        from types import SimpleNamespace
        from gymnasium.spaces import Box
        from flow_jepa.baselines.cem import CEMSolver, verify_solver, SpatialWorldCost
        verify_solver()
        class World(nn.Module):
            def rollout(self, z, a, budget=None):
                return z[:, None] + a.cumsum(1).mean(-1)[:, :, None, None]
        mean, std = torch.arange(8)*.01, torch.ones(8)*.5
        cost = SpatialWorldCost(World(), mean, std)
        solver = CEMSolver(cost, num_samples=7, topk=3,n_steps=3,seed=19)
        solver.configure(action_space=Box(-1.,1.,shape=(1,4)), n_envs=1,
                         config=SimpleNamespace(horizon=2,action_block=2))
        start, goal = torch.zeros(1,2,8), torch.ones(1,2,8)
        actual = solver(dict(start=start,goal=goal))['actions']
        g = torch.Generator().manual_seed(19)
        m, sd = torch.zeros(1,2,8), torch.ones(1,2,8)
        for _ in range(3):
            x = torch.randn(1,7,2,8,generator=g)*sd[:,None]+m[:,None]
            x[:,0] = m
            physical = x*std+mean
            distances = (physical.sum(2).mean(-1)-1).square()
            inds = distances.topk(3,largest=False).indices
            elites = x[torch.arange(1)[:,None],inds]
            m, sd = elites.mean(1), elites.std(1)
        self.assertTrue(torch.equal(actual,m))


if __name__ == '__main__': unittest.main()
