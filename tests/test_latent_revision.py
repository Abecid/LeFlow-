"""Causality, factual labels, execution handoff and frozen-world checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from flow_jepa.common import digest
from flow_jepa.execution.model import ExecutionSegments
from flow_jepa.execution.revision import RevisionPolicy, RevisionSegments
from flow_jepa.execution.revision_controller import RevisionController
from test_execution import World, FakeBank


class RevisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(3072)
        self.c=dict(encoder=dict(dim=8,token_grid=[1,1,2]),
            controller_grounded=dict(width=16,depth=1,mixtures=4),
            latent_revision=dict(history_steps=4,workspace_slots=4,reasoning_steps=4,
                                 error_scale=.02,correction_bound=.1,calibration_weight=.1))

    def batch(self):
        return dict(z=torch.randn(3,4,2,8),z4=torch.randn(3,2,8),a=torch.rand(3,5,8)*2-1,
            history_start=torch.randn(3,4,2,8),history_next=torch.randn(3,4,2,8),
            history_action=torch.rand(3,4,8)*2-1,history_mask=torch.ones(3,4,dtype=torch.bool))

    def test_history_changes_workspace_but_masked_padding_does_not(self):
        model=RevisionPolicy(self.c).eval();z=torch.randn(2,2,8)
        history=model.empty_history(z)
        a=model.prepare(z,z+.2,z+.5,history)['thought']
        changed={k:v.clone() for k,v in history.items()}
        for key in ('start','next','predicted'):changed[key].normal_(100,10)
        b=model.prepare(z,z+.2,z+.5,changed)['thought']
        self.assertTrue(torch.equal(a,b))
        changed['mask'][:,-1]=True
        c=model.prepare(z,z+.2,z+.5,changed)['thought']
        self.assertGreater(float((a-c).abs().max()),1e-5)

    def test_prefix_prediction_has_no_unexecuted_suffix_dependency(self):
        model=RevisionPolicy(self.c).eval()
        with torch.no_grad():model.prefix_error[-1].weight.normal_(0,.1)
        thought=torch.randn(3,16);actions=torch.randn(3,5,8);pred=torch.randn(3,5,2,8);target=torch.randn(3,2,8)
        first=model.corrections(thought,actions,pred,target)[:,0]
        actions[:,1:].normal_(5,1);pred[:,1:].normal_(5,1)
        second=model.corrections(thought,actions,pred,target)[:,0]
        self.assertTrue(torch.equal(first,second))

    def test_future_outcomes_are_labels_not_prediction_inputs(self):
        model=RevisionPolicy(self.c).eval();world=World();batch=self.batch();seen=[]
        handle=model.prefix_error.register_forward_hook(lambda module,inputs,out:seen.append(out.detach().clone()))
        torch.manual_seed(9);loss1,_=model(batch,world,weight=0.)
        first=[x.clone() for x in seen];seen.clear()
        changed={k:v.clone() for k,v in batch.items()}
        changed['z'][:,1]=torch.randn_like(changed['z'][:,1])*3
        changed['z4']=torch.randn_like(changed['z4'])*3
        torch.manual_seed(9);loss2,_=model(changed,world,weight=0.)
        handle.remove()
        self.assertEqual(len(first),2)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(first,seen)))
        self.assertGreater(abs(float((loss1-loss2).detach())),1e-5)

    def test_backprop_updates_policy_calibrator_not_world(self):
        model=RevisionPolicy(self.c);world=World();original=copy.deepcopy(world.state_dict())
        loss,metrics=model(self.batch(),world,weight=.1);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        self.assertGreater(float(model.output.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.terminal_error[-1].weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in world.parameters()))
        self.assertTrue(all(torch.equal(v,original[k]) for k,v in world.state_dict().items()))
        self.assertIn('recorded_corrected_prefix_mae',metrics)

    def test_exact_prefix_budget_observation_order_and_episode_reset(self):
        model=RevisionPolicy(self.c).eval();world=World()
        controller=RevisionController(model,world,self.c,FakeBank())
        start,goal=torch.randn(1,2,8),torch.randn(1,2,8)
        action,_,_,_=controller.plan(start,goal)
        self.assertEqual(world.calls,480);self.assertEqual(len(controller.history),0)
        predicted=world.rollout(start,action[None,None])[:,0]
        self.assertTrue(torch.allclose(predicted,controller.pending[0]))
        actual=predicted+.03
        controller.observe(actual);self.assertEqual(len(controller.history),1)
        self.assertTrue(torch.equal(controller.history[-1][1],actual))
        controller.observe(actual);self.assertEqual(len(controller.history),1)
        self.assertIn('corrected_prefix_progress_error',controller.trace[-1])
        controller.begin_episode()
        self.assertEqual(controller.history,[]);self.assertIsNone(controller.pending)

    def test_dataset_draws_match_and_history_ends_at_current_state(self):
        c=copy.deepcopy(self.c)
        c.update(training_tasks=['toy'],model=dict(chunk_steps=5,segments=12,state_representation='static'),
                 data=dict(action_repeat=2))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            row=dict(id='train/toy/0',path='episode.h5',task='toy',steps=64,
                     split='train',mode='expert',expert_success=True,first_success_action=100)
            manifest=dict(protocol=digest(c),mean=[0]*8,std=[1]*8,entries=[row])
            (root/'manifest.json').write_text(json.dumps(manifest))
            with h5py.File(root/'episode.h5','w') as f:
                f['image_goals']=np.broadcast_to(np.arange(65,dtype=np.float32)[:,None,None],(65,2,8))
                f['actions']=np.arange(64*8,dtype=np.float32).reshape(64,8)*.001
            old=ExecutionSegments(root,c,'controller_grounded',3072,16)
            new=RevisionSegments(root,c,'latent_revision',3072,16)
            for i in range(16):
                a,b=old[i],new[i]
                self.assertTrue(torch.equal(a['z'],b['z']))
                self.assertTrue(torch.equal(a['a'],b['a']))
                now=int(b['z'][0,0,0]);mask=b['history_mask']
                self.assertEqual(int(mask.sum()),min(4,now))
                self.assertTrue(bool((b['history_next'][mask,0,0]<=now).all()))
                self.assertTrue(bool((b['history_start'][mask,0,0]<now).all()))
            before=new[0]
            now=int(before['z'][0,0,0])
            with h5py.File(root/'episode.h5','a') as f:f['image_goals'][now+1:]=999.
            after=new[0]
            for key in ('history_start','history_next','history_action','history_mask'):
                self.assertTrue(torch.equal(before[key],after[key]))


if __name__=='__main__':unittest.main()
