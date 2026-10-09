"""Behavioral checks for the new controller and offline supervision."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from flow_jepa.execution.bank import RouteBank
from flow_jepa.execution.controller import ExecutionController, local_cost
from flow_jepa.execution.model import ChunkPolicy
from flow_jepa.common import digest


class World(nn.Module):
    def __init__(self):
        super().__init__()
        self.map = nn.Linear(8, 8, bias=False).requires_grad_(False)
        self.calls = 0

    def rollout(self, z, a, budget=None):
        result = []
        for u in a.unbind(1):
            if budget: budget.check()
            self.calls += len(z)
            z = z + self.map(u)[:, None] * 0.1
            result.append(z)
        return torch.stack(result, 1)


class FakeBank:
    def query(self, start, goal, count):
        targets = start.expand(count, -1, -1).clone()
        targets[:, :, 0] += torch.linspace(0.1, 1.0, count)[:, None]
        return targets, torch.zeros(count, 5, 8), torch.linspace(0.05, 0.2, count), [{} for _ in range(count)]


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(3072)
        self.c = {'encoder':{'dim':8}, 'controller_grounded':{'width':16,'depth':1,'mixtures':4}}

    def test_observed_execution_loss_backpropagates_to_policy_only(self):
        model, world = ChunkPolicy(self.c), World()
        batch = dict(z=torch.randn(3,4,2,8), a=torch.rand(3,5,8)*2-1)
        initial = copy.deepcopy(world.state_dict())
        loss, metrics = model(batch, world, weight=0.1)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIn('prefix_loss', metrics)
        self.assertGreater(float(model.output.weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in world.parameters()))
        self.assertTrue(all(torch.equal(initial[k],v) for k,v in world.state_dict().items()))

    def test_exact_executed_prefix_and_fixed_prediction_budget(self):
        model, world = ChunkPolicy(self.c).eval(), World()
        controller = ExecutionController(model, world, self.c, FakeBank())
        start, goal = torch.randn(1,2,8), torch.randn(1,2,8)
        action, target, score, horizon = controller.plan(start, goal)
        self.assertEqual(world.calls, 3*8*4*5)
        self.assertEqual(horizon,5)
        self.assertTrue(torch.isfinite(action).all())
        self.assertLessEqual(float(action.abs().max()),1)
        predicted = world.rollout(start, action[None,None])[:,0]
        self.assertTrue(torch.allclose(predicted,controller.pending[0]))
        controller.observe(predicted)
        self.assertLess(abs(controller.trace[-1]['actual_prefix_prediction_error']),1e-6)
        self.assertFalse(controller.diagnostics()['full_chunk_executability_observed'])

    def test_bank_rejects_evaluation_examples(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bank.pt'
            torch.save(dict(protocol=digest(self.c),manifest='x',episode_ids=['validation/reach/0']),path)
            with self.assertRaisesRegex(ValueError,'Non-training'):
                RouteBank(path,self.c,'x',torch.device('cpu'))

    def test_terminal_window_does_not_reward_early_transient_match(self):
        goal=torch.tensor([[[1.,0.]]])
        predictions=torch.tensor([[[[1.,0.]],[[1.,0.]],[[1.,0.]],[[0.,1.]],[[0.,1.]]]])
        self.assertAlmostEqual(float(local_cost(predictions,goal)),1.)


if __name__ == '__main__': unittest.main()
