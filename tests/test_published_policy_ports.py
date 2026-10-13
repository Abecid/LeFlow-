"""Shape and gradient checks on synthetic data; no simulator/world weights."""
import unittest
import torch
from flow_jepa.baselines.sage_port import SagePolicy, SageController
from flow_jepa.baselines.diffusion_port import DiffusionActionPolicy, DiffusionPolicyController


def conf(method):
    return dict(primary_method=method, encoder={"dim":16},
       controller_grounded={"width":32,"depth":1,"mixtures":8},
       sage_port={"durations":[5,10,20,40,60],
           "deployment_durations":[5,5,10,10,20,20,40,60],
           "subgoal_weight":1,"candidates_per_subgoal":4,"cem_rounds":3},
       diffusion_policy_port={"width":32,"channels":32,"train_steps":20,"inference_steps":4})


class TestPublishedPorts(unittest.TestCase):
    def test_sage_loss_and_subgoal(self):
        c=conf("sage_port");model=SagePolicy(c)
        batch=dict(z=torch.randn(2,4,4,16),a=torch.randn(2,5,8),
                   sage_duration=torch.tensor([5.,20.]),
                   sage_subgoal=torch.randn(2,4,16))
        loss,values=model(batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(model.subgoal.out.weight.grad).all())
        self.assertTrue(torch.isfinite(model.actor.output.weight.grad).all())
        self.assertIn("subgoal_loss",values)

    def test_sage_world_search_and_execution_prefix(self):
        class World:
            def __init__(self):
                self.calls = 0

            def rollout(self, start, actions, budget=None):
                self.calls += len(start)*actions.shape[1]
                return start[:, None].expand(-1,5,-1,-1).clone()

        c=conf("sage_port")
        world=World()
        model=SagePolicy(c)
        ctrl=SageController(model,world,c,bank=None)
        start=torch.randn(1,4,16)
        goal=torch.randn_like(start)
        command,_,cost,_=ctrl.plan(start,goal)
        self.assertEqual(tuple(command.shape),(8,))
        self.assertLessEqual(float(command.abs().max()),1.)
        self.assertTrue(float("-inf")<cost<float("inf"))
        self.assertEqual(world.calls,480)

    def test_diffusion_training_and_sampling(self):
        c=conf("diffusion_policy_port");model=DiffusionActionPolicy(c)
        batch=dict(z=torch.randn(2,4,4,16),a=torch.randn(2,5,8))
        loss,values=model(batch);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.net.out[-1].weight.grad).all())
        actions=model.sample(batch["z"][:,0],batch["z"][:,3],count=3)
        self.assertEqual(tuple(actions.shape),(2,3,5,8))
        self.assertLessEqual(actions.abs().max().item(),1)
        self.assertIn("noise_mse",values)


if __name__=="__main__":
    unittest.main()
