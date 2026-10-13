"""Pure PyTorch smoke checks; no simulator or training dataset required."""
import unittest
import torch
from flow_jepa.baselines.simple_action_flow import (
    VanillaActionFlowPolicy, ActionFlowSearchPolicy,
    VanillaActionFlowController, ActionFlowSearchController,
)


def cfg(method):
    return {
        "primary_method": method,
        "encoder": {"dim": 16},
        "additional_baselines": {
            "width": 32, "context_depth": 1, "flow_depth": 1,
            "flow_steps": 2, "search_rounds": 3, "candidates_per_target": 4,
        },
    }


class FakeBank:
    def query(self, start, goal, count=7):
        q = goal.expand(count, -1, -1).clone()
        a = start.new_zeros(count, 5, 8)
        return q, a, start.new_zeros(count), [{} for _ in range(count)]


class FakeWorld:
    def __init__(self):
        self.calls = 0

    def rollout(self, start, action, budget=None):
        assert action.shape[-2:] == (5, 8)
        self.calls += len(start)*5
        return start[:,None].expand(-1,5,-1,-1).clone()


class TestActionFlowBaselines(unittest.TestCase):
    def test_train_backward_and_policy(self):
        for method, clazz in [
            ("vanilla_action_flow", VanillaActionFlowPolicy),
            ("action_flow_search", ActionFlowSearchPolicy),
        ]:
            model = clazz(cfg(method))
            batch = {"z": torch.randn(2,4,4,16), "a":torch.randn(2,5,8)}
            loss, parts = model(batch)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(model.flow.out[-1].weight.grad).all())
            self.assertIn("flow_loss", parts)
            sample = model.sample(batch["z"][:,0],batch["z"][:,2],batch["z"][:,3],3)
            self.assertEqual(tuple(sample.shape),(2,3,5,8))
            self.assertLessEqual(sample.abs().max().item(),1.)

    def test_controller_world_budget_and_prefix(self):
        start = torch.randn(1,4,16)
        goal = torch.randn_like(start)
        world = FakeWorld()
        simple = VanillaActionFlowController(VanillaActionFlowPolicy(cfg("vanilla_action_flow")),
                                              world,cfg("vanilla_action_flow"),FakeBank())
        simple.begin_episode()
        self.assertEqual(tuple(simple.plan(start,goal)[0].shape),(8,))
        self.assertEqual(world.calls,0)
        search = ActionFlowSearchController(ActionFlowSearchPolicy(cfg("action_flow_search")),
                                            world,cfg("action_flow_search"),FakeBank())
        search.begin_episode()
        action,_,_,_ = search.plan(start,goal)
        self.assertEqual(tuple(action.shape),(8,))
        self.assertEqual(world.calls,480)
        self.assertLessEqual(action.abs().max().item(),1.)


if __name__ == "__main__":
    unittest.main()
