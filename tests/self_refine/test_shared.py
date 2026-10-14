"""Verify transfer preserves the retained baseline and its deadline contract."""
from copy import deepcopy
from dataclasses import replace
import pytest,torch
from crossbench.spec import Spec
from crossbench.models import build
from crossbench.controller import Controller
from crossbench.world import ActionUnits,LeWMWorld
from flow_jepa.budget import PlanningBudget,PlanningBudgetExceeded
from self_refine.shared_eval import SharedController

class Tiny(torch.nn.Module):
    def __init__(self):super().__init__();self.bias=torch.nn.Parameter(torch.zeros(16))
    def action_encoder(self,a):return a.mean(-1,keepdim=True).expand(*a.shape[:-1],16)
    def predict(self,z,a):return z+.03*a+self.bias

@pytest.mark.parametrize('task,pd',[('pusht',2),('cube',5)])
def test_exact_retained_action_and_rng(task,pd):
    torch.set_num_threads(1);s=Spec(task,'leflow',dim=16,primitive_dim=pd)
    w=LeWMWorld(Tiny(),ActionUnits([-1]*pd,[1]*pd,[0]*pd,[1]*pd))
    m=build(s).eval();z,g=torch.randn(1,1,16),torch.randn(1,1,16)
    old=Controller(m,w,None,s);new=SharedController(m,w,s,{'name':'leflow','num_samples':64,'flow_steps':16,'refinement':{'mode':'native'}})
    torch.manual_seed(31);expected=old.plan(z,g);rng=torch.get_rng_state().clone()
    torch.manual_seed(31);actual=new.plan(z,g)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
    assert new.trace[-1]['world_transitions']==320

def test_guidance_frozen_and_deadline():
    torch.set_num_threads(1);s=Spec('cube','leflow',dim=16,primitive_dim=5)
    w=LeWMWorld(Tiny(),ActionUnits([-1]*5,[1]*5,[0]*5,[1]*5));m=build(s)
    before=deepcopy(m.state_dict());z,g=torch.randn(1,1,16),torch.randn(1,1,16)
    c=SharedController(m,w,s,{'name':'guided_pnp','num_samples':4,'flow_steps':4,
        'refinement':{'mode':'guided_pnp','indices':[1,2],'rounds':1,'guidance_batch_size':2}})
    with torch.inference_mode():a=c.plan(z,g)
    assert a.shape==(25,)and a.abs().max()<=1
    assert c.trace[-1]['backward_calls']==4
    assert all(torch.equal(v,m.state_dict()[k])for k,v in before.items())
    assert all(p.grad is None for p in m.parameters())
    with pytest.raises(PlanningBudgetExceeded):c.plan(z,g,budget=PlanningBudget(0))
    assert not c.trace[-1]['plan_completed']
