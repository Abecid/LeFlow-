"""Feedback-specific tests using small differentiable fake world."""
import unittest
import torch
from flow_jepa.baselines.feedback_wm_port import FeedbackWorldPolicy, FeedbackWorldController


def cfg():
    return dict(primary_method="feedback_wm_port",encoder={"dim":16},
       diffusion_policy_port={"width":32,"channels":32,
                              "train_steps":20,"inference_steps":4},
       feedback_wm_port={"feedback_decay":.7,"feedback_gain":.5,
            "guidance_last_steps":2,"guidance_strength":.05,
            "max_gradient_rms":.25,"probes":4,"probe_sigma":.1,
            "controllability_power":.5})


class FakeWorld(torch.nn.Module):
    def forward(self,state,action):
        assert action.shape[1]==8
        # Differentiable with respect to actions. Simulates no real robot.
        return state+action[:,0,None,None]*.02


class TestFeedbackPort(unittest.TestCase):
    def test_factual_updates_only_and_gradients(self):
        model=FeedbackWorldPolicy(cfg())
        ctrl=FeedbackWorldController(model,FakeWorld(),cfg(),bank=None)
        start=torch.randn(1,4,16)
        goal=torch.randn_like(start)
        action,_,_,_=ctrl.plan(start,goal)
        self.assertEqual(tuple(action.shape),(8,))
        self.assertIsNone(ctrl._feedback)
        self.assertTrue(torch.isfinite(action).all())
        self.assertIsNotNone(ctrl._predicted_next)
        ctrl.observe(start+.2)
        self.assertEqual(ctrl.feedback_steps,1)
        self.assertEqual(tuple(ctrl._feedback.shape),(1,4,16))
        ctrl.plan(start+.2,goal)
        ctrl.begin_episode()
        self.assertEqual(ctrl.feedback_steps,0)
        self.assertIsNone(ctrl._feedback)

    def test_diffusion_training_still_works(self):
        m=FeedbackWorldPolicy(cfg())
        batch=dict(z=torch.randn(2,4,4,16),a=torch.randn(2,5,8))
        loss,parts=m(batch);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("noise_mse",parts)


if __name__=="__main__":
    unittest.main()
