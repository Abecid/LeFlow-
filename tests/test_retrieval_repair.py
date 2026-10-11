"""Cross-episode evidence, teacher isolation, exact old controller budget."""
import copy
import unittest
from types import SimpleNamespace
import torch
from flow_jepa.models import distance
from flow_jepa.execution.retrieval_repair import TrainingRetrieval, RetrievalRepairPolicy, RetrievalRepairController
from test_execution import World, FakeBank
import test_repair_flow_ttt as repair_fixtures


class RetrievalRepairTests(unittest.TestCase):
    def setUp(self):
        f=repair_fixtures.RepairFlowTTTTests();f.setUp();self.c=f.c;self.base=f.base

    def batch(self):
        b=self.base.batch();b.update(retrieval_enabled=torch.tensor([True,False,True]),
            bank_episode_index=torch.tensor([0,1,2]),retrieved_episode_index=torch.tensor([1,2,0]),
            retrieved_actions=torch.rand(3,5,8)*2-1)
        return b

    def test_exact_reranking_excludes_entire_own_episode(self):
        raw=torch.randn(12,2,8)
        bank=SimpleNamespace(raw=raw,routes=torch.tensor([[i,i+1,i+1,5] for i in (0,2,4,6,8,10)]),
            episode=torch.tensor([0,0,0,0,1,1,1,1,2,2,2,2]),chunks=torch.randn(12,5,8))
        bank.states=lambda ids:raw[ids]
        bank.key=lambda z:z.flatten(1)
        bank.keys=bank.key(raw)
        q=TrainingRetrieval(bank);starts=raw[[0,4,8]];targets=raw[[1,5,9]];excluded=torch.tensor([0,1,2])
        actions,eps=q.query(starts,targets,excluded)
        for j in range(3):
            eligible=q.episodes!=excluded[j]
            cost=distance(raw[q.start_ids],starts[j])+distance(raw[q.end_ids],targets[j])
            cost[~eligible]=float('inf');i=cost.argmin()
            self.assertTrue(torch.equal(actions[j],bank.chunks[q.start_ids[i]]))
            self.assertNotEqual(int(eps[j]),int(excluded[j]))
        with self.assertRaisesRegex(ValueError,'cross-episode'):
            bank.episode.zero_();TrainingRetrieval(bank).query(starts[:1],targets[:1],torch.tensor([0]))

    def test_teacher_changes_cannot_change_retrieval_or_planning_evidence(self):
        m=RetrievalRepairPolicy(self.c).eval();b=self.batch();w=World();seen=[]
        original=m.revise
        def record(ctx,actions,scores):
            result=original(ctx,actions,scores);seen.append((actions.clone(),result['repair_anchor'].clone(),result['thought'].clone()));return result
        m.revise=record
        torch.manual_seed(31);before_loss,_=m(b,w,depth=2);before=seen.pop()
        other={k:v.clone() for k,v in b.items()};other['a'].neg_();other['z'][:,1]*=-3;other['z4']*=-3
        torch.manual_seed(31);after_loss,_=m(other,w,depth=2);after=seen.pop()
        for x,y in zip(before,after):self.assertTrue(torch.equal(x,y))
        self.assertGreater(abs(float((before_loss-after_loss).detach())),1e-6)
        self.assertTrue(torch.equal(before[0][[0,2],-1],b['retrieved_actions'][[0,2]]))
        self.assertFalse(torch.equal(before[0][1,-1],b['retrieved_actions'][1]))
        other['retrieved_episode_index']=other['bank_episode_index']
        with self.assertRaisesRegex(ValueError,'leak'):m(other,w,depth=2)

    def test_retrieved_slot_is_verified_by_world_and_can_anchor_repairs(self):
        m=RetrievalRepairPolicy(self.c).eval();b=self.batch();w=World();original=m.score_candidates
        def choose_last(ctx,actions,predicted,target):
            s=original(ctx,actions,predicted,target)
            if actions.shape[1] == 4:
                self.assertTrue(torch.equal(actions[[0,2],-1],b['retrieved_actions'][[0,2]]))
                s['cost']=torch.ones_like(s['cost']);s['cost'][:,-1]=0
            return s
        m.score_candidates=choose_last
        _,parts=m(b,w,depth=2)
        self.assertAlmostEqual(float(parts['retrieved_anchor_fraction']),2/3,places=6)

    def test_both_depth_gradients_and_frozen_world(self):
        m=RetrievalRepairPolicy(self.c);b=self.batch();w=World();before=copy.deepcopy(w.state_dict())
        for depth in (1,2):
            m.zero_grad(set_to_none=True);loss,parts=m(b,w,depth=depth);loss.backward()
            self.assertTrue(torch.isfinite(loss));self.assertGreater(float(m.flow.out[-1].weight.grad.abs().sum()),0)
            if depth==2:
                for p in (m.repair_flow.out[-1].weight,m.repair_anchor[0].weight,m.evidence[0].weight):
                    self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.abs().sum()),0)
            else:self.assertIsNone(m.repair_flow.out[-1].weight.grad)
        self.assertTrue(all(p.grad is None for p in w.parameters()))
        self.assertTrue(all(torch.equal(before[k],v) for k,v in w.state_dict().items()))

    def test_controller_provenance_without_inference_expansion(self):
        m=RetrievalRepairPolicy(self.c).eval();w=World();ctrl=RetrievalRepairController(m,w,self.c,FakeBank())
        start,goal=torch.randn(1,2,8),torch.randn(1,2,8)
        for _ in range(2):
            calls=w.calls;a,_,_,_=ctrl.plan(start,goal);t=ctrl.trace[-1]
            self.assertEqual(w.calls-calls,320);self.assertEqual(t['flow_evaluations'],12)
            self.assertIn(t['repair_anchor_source'],('generated','retrieved','shifted'))
            self.assertLessEqual(t['selected_distance_from_incumbent_linf'],.250001)
            self.assertLessEqual(t['round_best_scores'][1],t['round_best_scores'][0])
            start=w.rollout(start,a[None,None])[:,0]+.01;ctrl.observe(start)
        ctrl.begin_episode();self.assertFalse(ctrl.history);self.assertIsNone(ctrl.previous_plan)


if __name__=='__main__':unittest.main()
