"""Progress labels, causal correction, objective wiring and fixed controller cost."""
import unittest
import torch
from flow_jepa.execution.progress_objective import ProgressObjectivePolicy, ProgressObjectiveController
from test_execution import World, FakeBank
import test_retrieval_repair as fixtures


class ProgressObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.RetrievalRepairTests();self.f.setUp();self.c=self.f.c
        self.c['progress_objective']=dict(width=4,time_scale_blocks=60,score_weight=.05,
            loss_weight=1.,calibration_weight=.1,error_scale=.1,correction_bound=.25)

    def batch(self):
        b=self.f.batch();b['progress_states']=torch.randn(3,3,2,8)
        b['progress_labels']=torch.tensor([[.8,.2,0.],[.7,.1,0.],[.5,0.,0.]])
        b['sample_goal_offset']=torch.tensor([10,20,40])
        return b

    def test_empty_memory_and_nonfinite_padding_are_not_evidence(self):
        m=ProgressObjectivePolicy(self.c);s=torch.randn(3,2,8);h=m.empty_history(s)
        for k in ('start','next','predicted','action'):h[k].fill_(float('nan'))
        w=m.fit_progress(h,s)
        self.assertTrue(torch.equal(w,m.progress_prior.expand(3,-1)))
        ctx=m.prepare(s,s,s,h)
        self.assertTrue(torch.isfinite(ctx['thought']).all())

    def test_observed_outcomes_update_progress_but_action_suffix_cannot(self):
        m=ProgressObjectivePolicy(self.c);b=self.batch();world=World();z=b['z']
        h=m.factual_history(b,world);h['mask'].fill_(True)
        w=m.fit_progress(h,z[:,3]);h['next']=h['predicted'].clone()
        clean=m.fit_progress(h,z[:,3])
        self.assertGreater(float((w-clean).abs().max().detach()),1e-5)
        ctx=m.prepare(z[:,0],z[:,2],z[:,3],h);a=b['a'][:,None];prefix=world.rollout(z[:,0],b['a'])[:,None,0]
        first=m.progress_correction(ctx,a,prefix);other=a.clone();other[:,:,1:]*=-1
        self.assertTrue(torch.equal(first,m.progress_correction(ctx,other,prefix)))

    def test_calibration_labels_cannot_flatten_progress_readout(self):
        m=ProgressObjectivePolicy(self.c);b=self.batch();h=m.factual_history(b,World())
        m.fit_progress(h,b['z'][:,3]).sum().backward()
        self.assertTrue(all(p.grad is None for p in m.goal_progress.parameters()))

    def test_future_supervision_cannot_change_candidate_evidence(self):
        m=ProgressObjectivePolicy(self.c).eval();b=self.batch();w=World();seen=[];original=m.revise
        def save(ctx,a,s):
            out=original(ctx,a,s);seen.append((a.clone(),s['cost'].clone(),out['thought'].clone()));return out
        m.revise=save
        torch.manual_seed(32);m(b,w,depth=2);before=seen.pop()
        other={k:v.clone() for k,v in b.items()};other['progress_states'].neg_();other['progress_labels'].fill_(.5)
        other['a'].neg_();other['z'][:,1].neg_();other['z4'].neg_()
        torch.manual_seed(32);m(other,w,depth=2);after=seen.pop()
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(before,after)))

    def test_goal_conditioning_and_stage_penalty_use_same_global_goal(self):
        m=ProgressObjectivePolicy(self.c).eval();s=torch.randn(1,2,8);g=torch.randn_like(s)
        self.assertFalse(torch.equal(m.goal_progress(s,g),m.goal_progress(s,-g)))
        ctrl=ProgressObjectiveController(m,World(),self.c,FakeBank());targets=torch.randn(8,2,8);ctx={};cost=torch.rand(8)
        revised=ctrl.target_route_costs(ctx,s,targets,g,cost)
        advance=m.goal_progress(s,g)-m.goal_progress(targets,g)
        expected=.05*((-advance).relu()+(advance-5/60).relu())
        self.assertTrue(torch.allclose(revised-cost,expected,atol=1e-7))

    def test_both_depths_finite_all_deep_parameters_and_frozen_world(self):
        m=ProgressObjectivePolicy(self.c);b=self.batch();w=World()
        for depth in (1,2):
            m.zero_grad(set_to_none=True);loss,parts=m(b,w,depth=depth);loss.backward()
            self.assertTrue(torch.isfinite(loss));self.assertIn('progress_temporal_mae',parts)
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
            self.assertGreater(float(m.goal_progress.project.weight.grad.abs().sum()),0)
            self.assertGreater(float(m.progress_condition.weight.grad.abs().sum()),0)
            if depth==2:self.assertFalse([n for n,p in m.named_parameters() if p.grad is None])
        self.assertTrue(all(p.grad is None for p in w.parameters()))

    def test_incumbent_budget_fixed_score_and_episode_reset(self):
        m=ProgressObjectivePolicy(self.c).eval();w=World();ctrl=ProgressObjectiveController(m,w,self.c,FakeBank())
        s,g=torch.randn(1,2,8),torch.randn(1,2,8)
        for _ in range(2):
            calls=w.calls;a,_,_,_=ctrl.plan(s,g);t=ctrl.trace[-1]
            self.assertEqual(w.calls-calls,320);self.assertEqual(t['flow_evaluations'],12)
            self.assertLessEqual(t['selected_distance_from_incumbent_linf'],.250001)
            self.assertLessEqual(t['round_best_scores'][1],t['round_best_scores'][0])
            self.assertNotIn('goal_progress_observed',t)
            s=w.rollout(s,a[None,None])[:,0]+.01;ctrl.observe(s)
            self.assertIn('goal_progress_observed',t)
        ctrl.begin_episode();self.assertFalse(ctrl.history);self.assertIsNone(ctrl.pending_progress)


if __name__=='__main__':unittest.main()
