import unittest
import torch
from torch import nn
from crossbench.spec import Spec, METHODS
from crossbench.models import build
from crossbench.world import ActionUnits, LeWMWorld
from crossbench.controller import Controller
from flow_jepa.budget import PlanningBudget, PlanningBudgetExceeded


class TinyLeWM(nn.Module):
    def __init__(self,dim=16):
        super().__init__();self.bias=nn.Parameter(torch.zeros(dim))
    def action_encoder(self,a):
        return a.mean(-1,keepdim=True).expand(*a.shape[:-1],len(self.bias))
    def predict(self,z,a):
        return z+.02*a+self.bias


class Bank:
    def __init__(self,s):self.s=s
    def query(self,z,g,count=7):
        return g.expand(count,-1,-1).clone(),z.new_zeros(count,5,self.s.ad),z.new_zeros(count)
    def training_query(self,z,q,e):
        return z.new_zeros(len(z),5,self.s.ad)


def fixture(s):
    return dict(z=torch.randn(2,6,1,s.dim),a=torch.rand(2,5,s.ad)*2-1,goal=torch.randn(2,1,s.dim),
        history_start=torch.randn(2,4,1,s.dim),history_next=torch.randn(2,4,1,s.dim),
        history_action=torch.rand(2,4,s.ad),history_mask=torch.tensor([[0,0,1,1],[0,1,1,1]],dtype=torch.bool),
        episode=torch.tensor([0,1]),index=torch.tensor([0,1]),duration=torch.tensor([5,5]),subgoal=torch.randn(2,1,s.dim))


class TestCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_action_conversion(self):
        u=ActionUnits([0,0],[512,512],[256,256],[80,80],5)
        a=torch.rand(3,5,10)*2-1
        self.assertTrue(torch.allclose(u.from_world(u.to_world(a)),a,atol=1.e-6))
        self.assertTrue(torch.allclose(u.unit(u.physical(a)),a,atol=1.e-6))
        with self.assertRaises(ValueError):u.unit(torch.tensor([[513.,0.]]),strict=True)
    def test_all_methods_both_action_spaces(self):
        for dataset,dim in [('pusht',2),('cube',5)]:
            for method in METHODS:
                with self.subTest(dataset=dataset,method=method):
                    s=Spec(dataset,method,dim=16,width=32,primitive_dim=dim)
                    world=LeWMWorld(TinyLeWM(),ActionUnits([-1]*dim,[1]*dim,[0]*dim,[1]*dim))
                    m=build(s);bank=Bank(s);batch=fixture(s)
                    loss,parts=m.loss(batch,world,bank,step=2)
                    self.assertTrue(torch.isfinite(loss));loss.backward()
                    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
                    self.assertTrue(all(p.grad is None for p in world.parameters()))
                    m.eval();world.calls=0;ctrl=Controller(m,world,bank,s)
                    z,g=batch['z'][:1,0],batch['goal'][:1]
                    action=ctrl.plan(z,g)
                    self.assertEqual(tuple(action.shape),(s.ad,))
                    self.assertTrue(torch.isfinite(action).all());self.assertLessEqual(float(action.abs().max()),1.)
                    expected={'vanilla_action_flow':0,'diffusion_policy_port':0,'leflow':320,
                              'execution_revision':480,'action_flow_search':480,'sage_port':480,
                              'retrieval_repair_ttt':320,'feedback_wm_port':9}
                    self.assertEqual(world.calls,expected[method])
                    ctrl.observe(z+.03);ctrl.plan(z+.03,g)
                    ctrl.begin_episode();self.assertEqual(ctrl.history,[])
    def test_expired_budget_returns_no_action(self):
        s=Spec('pusht','vanilla_action_flow',dim=16,width=32)
        w=LeWMWorld(TinyLeWM(),ActionUnits([-1]*2,[1]*2,[0]*2,[1]*2))
        c=Controller(build(s),w,None,s)
        with self.assertRaises(PlanningBudgetExceeded):
            c.plan(torch.randn(1,1,16),torch.randn(1,1,16),budget=PlanningBudget(0))


if __name__=='__main__':unittest.main()
