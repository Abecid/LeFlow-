"""Causal, bounded action repair and exact controller accounting."""
import copy
import unittest
import torch
from flow_jepa.execution.repair_flow_ttt import RepairFlowTTTPolicy, RepairFlowTTTController
from test_execution import World, FakeBank
import test_progress_ttt as fixtures


class RepairFlowTTTTests(unittest.TestCase):
    def setUp(self):
        fixture=fixtures.ProgressTTTTests();fixture.setUp()
        self.base=fixture.base;self.c=fixture.c
        self.c['flow_reasoning']['rounds']=2
        self.c['repair_flow_ttt']=dict(radius=.25,solver_steps=4,loss_weight=1.,depth=2)

    def context(self,model):
        b=self.base.batch();w=World();z=b['z']
        ctx=model.prepare(z[:,0],z[:,2],z[:,3],model.factual_history(b,w))
        actions=model.propose_context(ctx,4)
        pred=w.rollout(z[:,0,None].expand(-1,4,-1,-1).flatten(0,1),actions.flatten(0,1))
        scores=model.score_candidates(ctx,actions,pred.reshape(3,4,5,2,8),z[:,2])
        return b,w,ctx,actions,scores

    def test_best_anchor_detached_and_candidate_permutation_preserves_repair(self):
        m=RepairFlowTTTPolicy(self.c).eval();b,w,ctx,a,s=self.context(m)
        a.requires_grad_();r=m.revise(ctx,a,s)
        expected=a[torch.arange(3),s['cost'].argmin(1)]
        self.assertTrue(torch.equal(r['repair_anchor'],expected));self.assertFalse(r['repair_anchor'].requires_grad)
        perm=torch.tensor([2,0,3,1]);ss={k:v[:,perm] for k,v in s.items()}
        other=m.revise(ctx,a[:,perm],ss)
        self.assertTrue(torch.equal(r['repair_anchor'],other['repair_anchor']))
        self.assertTrue(torch.allclose(r['thought'],other['thought'],atol=2e-6))
        predicted=w.rollout(b['z'][:,0,None].expand(-1,4,-1,-1).flatten(0,1),a.flatten(0,1)).reshape(3,4,5,2,8)
        rescored=m.score_candidates(r,a,predicted,b['z'][:,2])
        self.assertTrue(torch.equal(rescored['cost'],s['cost']))

    def test_full_gaussian_source_unchanged_incumbent_and_bounded_final_actions(self):
        m=RepairFlowTTTPolicy(self.c).eval();_,_,ctx,a,s=self.context(m);r=m.revise(ctx,a,s)
        noise=torch.randn(3,4,5,8)*4;original=noise.clone();seen=[]
        hook=m.repair_flow.register_forward_pre_hook(lambda mod,args:seen.append(args[0].clone()))
        plans=m.propose_context(r,4,noise=noise);hook.remove()
        self.assertTrue(torch.equal(seen[0],noise[:,1:]));self.assertTrue(torch.equal(noise,original))
        self.assertEqual(len(seen),4);self.assertGreater(float(seen[0].abs().max()),1)
        self.assertTrue(torch.equal(plans[:,0],r['repair_anchor']))
        self.assertLessEqual(float((plans-r['repair_anchor'][:,None]).abs().max()),.250001)
        self.assertLessEqual(float(plans.abs().max()),1)

    def test_future_teacher_cannot_change_anchor_or_evidence(self):
        m=RepairFlowTTTPolicy(self.c).eval();b=self.base.batch();w=World();seen=[]
        original=m.revise
        def revise(ctx,a,s):
            result=original(ctx,a,s);seen.append((result['repair_anchor'].clone(),result['thought'].clone()));return result
        m.revise=revise
        torch.manual_seed(81);loss,_=m(b,w,depth=2);before=seen.copy();seen.clear()
        changed={k:v.clone() for k,v in b.items()};changed['a'].neg_();changed['z'][:,1].mul_(-3);changed['z4'].mul_(-3)
        torch.manual_seed(81);new,_=m(changed,w,depth=2)
        self.assertEqual(len(before),1);self.assertTrue(all(torch.equal(x,y) for x,y in zip(before[0],seen[0])))
        self.assertGreater(abs(float((loss-new).detach())),1e-5)

    def test_gradients_and_world_isolation_at_both_depths(self):
        m=RepairFlowTTTPolicy(self.c);w=World();b=self.base.batch();before=copy.deepcopy(w.state_dict())
        for depth in (1,2):
            m.zero_grad(set_to_none=True);loss,parts=m(b,w,depth=depth);loss.backward()
            self.assertTrue(torch.isfinite(loss));self.assertEqual(float(parts['reasoning_depth']),depth)
            required=[m.flow.out[-1].weight,m.fast_keys[0].weight,m.fast_prior,m.fast_condition.weight]
            if depth==2:required += [m.repair_flow.out[-1].weight,m.repair_anchor[0].weight,m.evidence[0].weight]
            for p in required:
                self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.abs().sum()),0)
            if depth==1:self.assertIsNone(m.repair_flow.out[-1].weight.grad)
        self.assertTrue(all(p.grad is None for p in w.parameters()))
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in w.state_dict().items()))
        with self.assertRaisesRegex(ValueError,'inherited'):m(b,w,weight=.1)

    def test_controller_budget_preserved_incumbent_and_causal_memory(self):
        m=RepairFlowTTTPolicy(self.c).eval();w=World();ctrl=RepairFlowTTTController(m,w,self.c,FakeBank())
        start,goal=torch.randn(1,2,8),torch.randn(1,2,8);before=copy.deepcopy(m.state_dict())
        for i in range(3):
            calls=w.calls;action,_,_,_=ctrl.plan(start,goal);trace=ctrl.trace[-1]
            self.assertEqual(w.calls-calls,320);self.assertEqual(trace['flow_evaluations'],12)
            self.assertEqual(trace['surrogate_gradient_evaluations'],0)
            self.assertLessEqual(trace['selected_distance_from_incumbent_linf'],.250001)
            self.assertEqual(trace['fast_support_steps'],i)
            self.assertLessEqual(trace['round_best_scores'][1],trace['round_best_scores'][0])
            pred=w.rollout(start,action[None,None])[:,0];self.assertTrue(torch.allclose(pred,ctrl.pending[0]))
            start=pred+.03;ctrl.observe(start);self.assertTrue(torch.equal(ctrl.action_history[-1],action))
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in m.state_dict().items()))
        ctrl.begin_episode();self.assertFalse(ctrl.history);self.assertFalse(ctrl.action_history)


if __name__=='__main__':unittest.main()
