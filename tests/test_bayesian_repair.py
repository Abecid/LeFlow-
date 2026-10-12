"""Reference posterior algebra, factual supervision, and bounded decisions."""
import copy
import unittest

import torch

from flow_jepa.execution.bayesian_repair import (BayesianRepairPolicy,bayesian_update,
    predictive_moments,joint_weight_points,choose_with_margin)
from flow_jepa.execution.bayesian_controller import BayesianRepairController
from test_execution import World,FakeBank
from test_retrieval_repair import RetrievalRepairTests


class BayesianRepairTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.fixture=RetrievalRepairTests();self.fixture.setUp();self.c=copy.deepcopy(self.fixture.c)
        self.c['bayesian_repair_ttt']=dict(training_rounds=3,acceptance_beta=1.644854,
            nll_weight=1.,deployed_loss_weight=1.,initial_noise_variance=.01,
            noise_floor=1e-6,precision_floor=1e-4)

    def test_alpaca_normal_equations_and_predictive_density(self):
        torch.manual_seed(3)
        x=torch.randn(2,4,3,dtype=torch.float64);y=torch.randn(2,4,dtype=torch.float64)
        mask=torch.tensor([[1,1,0,0],[1,1,1,1]],dtype=torch.bool)
        prior=torch.randn(3,dtype=torch.float64);precision=torch.diag(torch.tensor([.2,.7,2.],dtype=torch.float64))
        mean,factor=bayesian_update(x,y,mask,prior,precision)
        q=torch.randn(2,5,3)
        mu,ep,pred=predictive_moments(q,mean,factor,torch.tensor(.07))
        for i in range(2):
            xi=x[i,mask[i]];yi=y[i,mask[i]]
            cov=torch.linalg.inv(precision+xi.T@xi)
            expected=cov@(precision@prior+xi.T@yi)
            self.assertTrue(torch.allclose(mean[i].double(),expected,atol=2e-6))
            expected_var=.07*(1+torch.einsum('ni,ij,nj->n',q[i].double(),cov,q[i].double()))
            self.assertTrue(torch.allclose(pred[i].double(),expected_var,atol=2e-6))
        self.assertTrue(torch.allclose(pred-ep,torch.full_like(pred,.07)))
        self.assertTrue(torch.allclose(mu,(q*mean[:,None]).sum(-1)))

    def test_joint_posterior_points_preserve_covariance_and_common_mode(self):
        precision=torch.tensor([[2.,.7],[.7,1.]])[None]
        factor=torch.linalg.cholesky(precision);mean=torch.tensor([[.2,-.1]])
        points=joint_weight_points(mean,factor,torch.tensor(.3))
        centered=points-mean[:,None]
        self.assertTrue(torch.allclose(points.mean(1),mean))
        self.assertTrue(torch.allclose(centered.transpose(1,2)@centered/4,.3*torch.linalg.inv(precision),atol=1e-6))
        # Shared uncertainty cancels exactly for identical features/actions.
        joint=torch.stack((points.sum(-1),points.sum(-1)-.01),1)
        chosen,diagnostic=choose_with_margin(joint.mean(-1),joint,1.644854)
        self.assertEqual(int(chosen),1)
        self.assertLess(float(diagnostic['uncertainty'].max()),1e-7)
        # An apparent improvement with uncertain sign must preserve incumbent.
        joint=torch.tensor([[[0.,0.,0.,0.],[-.3,.2,-.3,.2]]])
        chosen,_=choose_with_margin(joint.mean(-1),joint,1.644854)
        self.assertEqual(int(chosen),0)

    def test_single_observation_per_transition_padding_and_isolation(self):
        model=BayesianRepairPolicy(self.c);batch=self.fixture.batch();world=World()
        history=model.factual_history(batch,world);q=batch['z'][:,2]
        mean,factor=model.fit_posterior(history,q)
        changed=copy.deepcopy(history);changed['next'][0].mul_(-1)
        other,_=model.fit_posterior(changed,q)
        self.assertTrue(torch.equal(mean[1:],other[1:]));self.assertFalse(torch.equal(mean[0],other[0]))
        empty=model.empty_history(q)
        for k in ('start','next','predicted','action'):empty[k].fill_(float('nan'))
        m,l=model.fit_posterior(empty,q)
        self.assertTrue(torch.equal(m,model.fast_prior.expand(3,-1)))
        self.assertTrue(torch.allclose(l@l.transpose(-1,-2),model.prior_precision()[None].expand(3,-1,-1)))
        keys=model.transition_keys(history['start'],history['action'],history['predicted'],q[:,None].expand_as(history['start'])).mean(-2)
        x=torch.where(history['mask'][...,None],keys,0.)
        self.assertTrue(torch.allclose(factor@factor.transpose(-1,-2),model.prior_precision()[None]+x.transpose(-1,-2)@x,atol=2e-6))

    def test_all_trained_depths_finite_gradients_world_frozen(self):
        model=BayesianRepairPolicy(self.c);batch=self.fixture.batch();world=World()
        original=copy.deepcopy(world.state_dict())
        for depth in (1,2,3):
            model.zero_grad(set_to_none=True);loss,parts=model(batch,world,depth=depth);loss.backward()
            self.assertTrue(torch.isfinite(loss));self.assertEqual(float(parts['reasoning_depth']),depth)
            required=[model.fast_prior,model.fast_keys[0].weight,model.prior_tril,model.log_noise_variance,model.flow.out[-1].weight]
            if depth>1:required.extend((model.repair_flow.out[-1].weight,model.evidence[0].weight))
            for p in required:
                self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in world.parameters()))
        self.assertTrue(all(torch.equal(v,original[k]) for k,v in world.state_dict().items()))

    def test_future_labels_do_not_enter_any_refinement_round(self):
        model=BayesianRepairPolicy(self.c).eval();batch=self.fixture.batch();seen=[]
        original=model.revise
        def capture(ctx,a,s):
            out=original(ctx,a,s);seen.append((out['repair_anchor'].clone(),out['thought'].clone(),out['posterior_factor'].clone()));return out
        model.revise=capture
        torch.manual_seed(51);first,_=model(batch,World(),depth=3);before=seen.copy();seen.clear()
        changed={k:v.clone() for k,v in batch.items()};changed['a'].neg_();changed['z'][:,1].mul_(-2);changed['z4'].mul_(-2)
        torch.manual_seed(51);second,_=model(changed,World(),depth=3)
        self.assertEqual(len(before),2)
        self.assertTrue(all(torch.equal(x,y) for old,new in zip(before,seen) for x,y in zip(old,new)))
        self.assertNotEqual(float(first.detach()),float(second.detach()))

    def test_rounds_bound_reset_score_identity_and_cached_accounting(self):
        for rounds in (1,2,3):
            c=copy.deepcopy(self.c);c['flow_reasoning']['rounds']=rounds
            model=BayesianRepairPolicy(c).eval();world=World();ctrl=BayesianRepairController(model,world,c,FakeBank())
            original=copy.deepcopy(model.state_dict());start,goal=torch.randn(1,2,8),torch.randn(1,2,8)
            for step in range(2):
                calls=world.calls;action,_,_,_=ctrl.plan(start,goal);trace=ctrl.trace[-1]
                self.assertEqual(world.calls-calls,160+120*(rounds-1))
                self.assertEqual(trace['candidate_world_transitions'],160+120*(rounds-1))
                self.assertEqual(trace['cached_world_transitions'],40*(rounds-1))
                self.assertEqual(trace['fast_support_steps'],step)
                self.assertLessEqual(trace['selected_distance_from_incumbent_linf'],.250001)
                for a,b in zip(trace['round_best_scores'],trace['round_best_scores'][1:]):self.assertLessEqual(b,a+1e-7)
                for entry in trace['posterior_comparisons'][1:]:
                    for route,selected in enumerate(entry['selected']):
                        if selected:self.assertGreater(entry['lower'][route][selected],0.)
                pred=world.rollout(start,action[None,None])[:,0]
                self.assertTrue(torch.allclose(pred,ctrl.pending[0]));start=pred+.01;ctrl.observe(start)
                self.assertIn('observed_prefix_cost_error',trace)
            ctrl.begin_episode();self.assertFalse(ctrl.history);self.assertFalse(ctrl.action_history);self.assertIsNone(ctrl.previous_plan)
            self.assertTrue(all(torch.equal(v,original[k]) for k,v in model.state_dict().items()))


if __name__=='__main__':unittest.main()
