"""Previous-decision causality, exact shift, gradients and unchanged deployment."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import h5py
import numpy as np
import torch
from flow_jepa.execution.shifted_repair import ShiftedRepairSegments, ShiftedTrainingRetrieval, ShiftedRepairPolicy, ShiftedRepairController
from flow_jepa.execution.retrieval_repair import RetrievalRepairSegments, RetrievalRepairPolicy, RetrievalRepairController
from test_execution import World, FakeBank
import test_retrieval_repair as fixtures


class ShiftedRepairTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.RetrievalRepairTests();self.fixture.setUp();self.c=self.fixture.c

    def batch(self):
        b=self.fixture.batch()
        for k in ('history_start','history_next','history_action','history_mask'):
            b['previous_'+k]=b[k].clone()
        b.update(previous_start=torch.randn(3,2,8),previous_available=torch.tensor([True,False,True]),
                 previous_retrieved_actions=torch.rand(3,5,8)*2-1,
                 previous_retrieved_episode_index=torch.tensor([1,-1,0]))
        b['previous_history_mask'][0]=False
        return b

    def test_dataset_previous_history_ends_before_current_transition(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'sample.h5';states=np.arange(12*16,dtype=np.float32).reshape(12,2,8)
            acts=np.arange(11*8,dtype=np.float32).reshape(11,8)
            with h5py.File(path,'w') as f:f['features']=states;f['actions']=acts
            ds=ShiftedRepairSegments.__new__(ShiftedRepairSegments)
            ds.c=self.c;ds.root=Path(d);ds.mean=0.;ds.std=1.;ds.state_key='features'
            base=dict(z=torch.zeros(4,2,8))
            for start in (0,1,6):
                ds.sample_spec=lambda i:({'path':'sample.h5'},start,5)
                with patch.object(RetrievalRepairSegments,'__getitem__',return_value=base.copy()):a=ds[0]
                self.assertEqual(bool(a['previous_available']),start>0)
                self.assertEqual(int(a['previous_history_mask'].sum()),min(max(start-1,0),4))
                if start:self.assertTrue(torch.equal(a['previous_start'],torch.from_numpy(states[start-1])))
                if start>1:
                    self.assertTrue(torch.equal(a['previous_history_next'][-1],a['previous_start']))
                    self.assertTrue(torch.equal(a['previous_history_action'][-1],torch.from_numpy(acts[start-2])))
                if start==6:
                    with h5py.File(path,'r+') as f:f['features'][start:]=999;f['actions'][start-1:]=999
                    with patch.object(RetrievalRepairSegments,'__getitem__',return_value=base.copy()):b=ds[0]
                    for k in a:self.assertTrue(torch.equal(a[k],b[k]),k)

    def test_prior_plan_cannot_observe_current_state_teacher_or_outcome(self):
        m=ShiftedRepairPolicy(self.c).eval();b=self.batch();w=World()
        torch.manual_seed(82);a=m.previous_plans(b,w)
        changed={k:v.clone() for k,v in b.items()}
        changed['z'][:,:2].normal_(50,3);changed['z4'].neg_();changed['a'].neg_()
        for k in ('history_start','history_next','history_action'):changed[k].normal_(20,3)
        torch.manual_seed(82);other=m.previous_plans(changed,w)
        self.assertTrue(torch.equal(a,other));self.assertFalse(a.requires_grad)
        self.assertEqual(float(a[1].abs().sum()),0);self.assertLessEqual(float(a.abs().max()),1)
        changed['previous_retrieved_episode_index'][0]=changed['bank_episode_index'][0]
        with self.assertRaisesRegex(ValueError,'exclude'):m.previous_plans(changed,w)

    def test_exact_shift_fresh_tail_and_teacher_isolation(self):
        m=ShiftedRepairPolicy(self.c).eval();b=self.batch();w=World();seen=[];proposed=[]
        previous=torch.arange(3*5*8,dtype=torch.float32).reshape(3,5,8)/200
        m.previous_plans=lambda batch,world:previous
        propose=m.propose_context;revise=m.revise
        def observe_proposal(*a,**kw):
            v=propose(*a,**kw);proposed.append(v.clone());return v
        def observe_context(ctx,a,s):
            result=revise(ctx,a,s);seen.append((a.clone(),result['repair_anchor'].clone(),ctx['pool_provenance'].clone()));return result
        m.propose_context=observe_proposal;m.revise=observe_context
        torch.manual_seed(19);loss,_=m(b,w,depth=2);before=seen.pop();fresh=proposed.pop()
        self.assertTrue(torch.equal(before[0][[0,2],1,:-1],previous[[0,2],1:]))
        self.assertTrue(torch.equal(before[0][:,1,-1],fresh[:,1,-1]))
        self.assertTrue(torch.equal(before[0][1,1],fresh[1,1]))
        self.assertTrue(torch.equal(before[2][:,1],torch.tensor([2,0,2])))
        changed={k:v.clone() for k,v in b.items()};changed['a'].neg_();changed['z'][:,1]*=-3;changed['z4']*=-3
        torch.manual_seed(19);other,_=m(changed,w,depth=2);after=seen.pop()
        for x,y in zip(before,after):self.assertTrue(torch.equal(x,y))
        self.assertGreater(abs(float((loss-other).detach())),1e-6)

    def test_previous_retrieval_uses_only_valid_prior_starts_targets_and_exclusion(self):
        q=ShiftedTrainingRetrieval.__new__(ShiftedTrainingRetrieval);b=self.batch();seen=[]
        def query(s,t,e):
            seen.append((s.clone(),t.clone(),e.clone()))
            return torch.zeros(len(s),5,8),(e+1)%3
        q.query=query
        out=q.augment(b);valid=b['previous_available']
        self.assertEqual(len(seen),2)
        self.assertTrue(torch.equal(seen[1][0],b['previous_start'][valid]))
        self.assertTrue(torch.equal(seen[1][1],b['z'][valid,2]))
        self.assertTrue(torch.equal(seen[1][2],b['bank_episode_index'][valid]))
        self.assertEqual(int(out['previous_retrieved_episode_index'][1]),-1)
        b['previous_available'].zero_();seen.clear();q.augment(b);self.assertEqual(len(seen),1)

    def test_deep_gradients_and_frozen_world(self):
        m=ShiftedRepairPolicy(self.c);w=World();b=self.batch();before=copy.deepcopy(w.state_dict())
        for depth in (1,2):
            m.zero_grad(set_to_none=True);loss,parts=m(b,w,depth=depth);loss.backward()
            self.assertTrue(torch.isfinite(loss));self.assertGreater(float(m.flow.out[-1].weight.grad.abs().sum()),0)
            if depth==2:
                for p in (m.repair_flow.out[-1].weight,m.repair_anchor[0].weight,m.evidence[0].weight):
                    self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.abs().sum()),0)
                self.assertLessEqual(float(parts['shifted_anchor_fraction']),2/3+1e-7)
            else:self.assertIsNone(m.repair_flow.out[-1].weight.grad)
        self.assertTrue(all(p.grad is None for p in w.parameters()))
        self.assertTrue(all(torch.equal(before[k],v) for k,v in w.state_dict().items()))

    def test_deployment_identical_for_identical_weights_rng_and_observations(self):
        old=RetrievalRepairPolicy(self.c).eval();new=ShiftedRepairPolicy(self.c).eval();new.load_state_dict(old.state_dict())
        world=World()
        a=RetrievalRepairController(old,world,self.c,FakeBank());b=ShiftedRepairController(new,copy.deepcopy(world),self.c,FakeBank())
        start,goal=torch.randn(1,2,8),torch.randn(1,2,8)
        for i in range(3):
            torch.manual_seed(42+i);x=a.plan(start,goal)
            torch.manual_seed(42+i);y=b.plan(start,goal)
            self.assertTrue(torch.equal(x[0],y[0]));self.assertTrue(torch.equal(x[1],y[1]));self.assertEqual(x[2:],y[2:])
            self.assertEqual(a.world.calls,320*(i+1));self.assertEqual(b.world.calls,320*(i+1))
            self.assertEqual(a.trace[-1]['flow_evaluations'],12);self.assertEqual(b.trace[-1]['flow_evaluations'],12)
            start=start+.03;a.observe(start);b.observe(start)
        a.begin_episode();b.begin_episode();self.assertIsNone(b.previous_plan);self.assertFalse(b.history)


if __name__=='__main__':unittest.main()
