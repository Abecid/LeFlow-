"""Full controller adapters with synthetic dynamics, not task-success evidence."""
from dataclasses import replace
import unittest
import torch
from torch import nn
from crossbench.spec import Spec
from crossbench.models import build
from crossbench.world import ActionUnits, LeWMWorld
from crossbench.controller import Controller
from leflow_refinement.crossbench import RefinementController
from leflow_refinement.runtime import variants, NativeRuntime
from flow_jepa.budget import PlanningBudget, PlanningBudgetExceeded


class TinyWorld(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(16))
    def action_encoder(self, a):
        return a.mean(-1, keepdim=True).expand(*a.shape[:-1], 16)
    def predict(self, z, a):
        return z + 0.03*a + self.bias


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setup_case(self, task='pusht'):
        pd = 2 if task == 'pusht' else 5
        spec = Spec(task, 'leflow', dim=16, primitive_dim=pd)
        world = LeWMWorld(TinyWorld(), ActionUnits([-1]*pd, [1]*pd, [0]*pd, [1]*pd))
        model = build(spec).eval()
        z, g = torch.randn(1,1,16), torch.randn(1,1,16)
        return spec, world, model, z, g

    def test_baseline_exact_actions_and_rng(self):
        s,w,m,z,g = self.setup_case()
        old = Controller(m,w,None,s)
        new = RefinementController(m,w,s,variants()['leflow'])
        torch.manual_seed(31)
        expected = old.plan(z,g)
        old_rng = torch.get_rng_state().clone()
        torch.manual_seed(31)
        actual = new.plan(z,g)
        new_rng = torch.get_rng_state().clone()
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        torch.testing.assert_close(new_rng,old_rng,atol=0,rtol=0)
        self.assertEqual(new.trace[0]['world_candidate_transitions'],320)

    def test_both_action_dimensions_and_frozen_weights(self):
        for task in ('pusht','cube'):
            s,w,m,z,g = self.setup_case(task)
            state = {k:v.clone() for k,v in m.state_dict().items()}
            cfg = replace(variants()['world_pnp'],samples=2,steps=4,refine_at=(1,2),repeats=1)
            controller=RefinementController(m,w,s,cfg)
            with torch.inference_mode():
                action=controller.plan(z,g)
            self.assertEqual(action.shape,(s.ad,))
            self.assertLessEqual(float(action.abs().max()),1)
            self.assertGreater(controller.trace[0]['gradient_norm_sum'],0)
            self.assertEqual(w.calls,2*5*3)
            self.assertTrue(all(p.grad is None for p in m.parameters()))
            self.assertTrue(all(p.grad is None for p in w.parameters()))
            for k,v in m.state_dict().items():
                torch.testing.assert_close(v,state[k],atol=0,rtol=0)
            controller.begin_episode()
            self.assertEqual(controller.trace,[])

    def test_pnp_does_not_consume_global_perturb_rng(self):
        s,w,m,z,g=self.setup_case()
        for name in ('leflow','pnp','world_gradient','world_pnp'):
            torch.manual_seed(234)
            ctrl=RefinementController(m,w,s,variants()[name])
            ctrl.plan(z,g)
            next_draw=torch.randn(10)
            torch.manual_seed(234);torch.randn(64,4,16)
            torch.testing.assert_close(next_draw,torch.randn(10),atol=0,rtol=0)

    def test_terminal_only_gradient_through_frozen_forward_and_inverse(self):
        s,w,m,z,g=self.setup_case()
        cfg=replace(variants()['world_gradient'],samples=2,steps=4,
                    refine_at=(1,),repeats=1,consistency_weight=0.)
        ctrl=RefinementController(m,w,s,cfg)
        ctrl.plan(z,g)
        self.assertGreater(ctrl.trace[0]['gradient_norm_sum'],0)
        self.assertTrue(all(p.grad is None for p in w.parameters()))

    def test_deadline_is_not_bypassed_by_backward(self):
        s,w,m,z,g=self.setup_case()
        ctrl=RefinementController(m,w,s,variants()['world_pnp'])
        with self.assertRaises(PlanningBudgetExceeded):
            ctrl.plan(z,g,budget=PlanningBudget(0))


if __name__=='__main__':unittest.main()
