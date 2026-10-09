"""Coverage, causal feedback, effective scoring and fixed-search-budget checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from flow_jepa.common import digest
from flow_jepa.execution.aligned import AlignedController, AlignedSegments
from flow_jepa.execution.revision import RevisionPolicy, RevisionSegments
from test_execution import World, FakeBank


class AlignedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(3072)
        self.c=dict(encoder=dict(dim=8,token_grid=[1,1,2]),
            controller_grounded=dict(width=16,depth=1,mixtures=4,goal_offsets=[5,10,20,40,60]),
            latent_revision=dict(history_steps=4,workspace_slots=4,reasoning_steps=4,
                                 error_scale=.02,correction_bound=.1,calibration_weight=.1),
            execution_revision=dict(prefix_weight=.5,stall_memory=8,stall_min_failures=3,
                stall_state_radius=.005,stall_target_radius=.005,stall_decay=.8,stall_penalty_cap=.02))

    def controller(self):
        return AlignedController(RevisionPolicy(self.c).eval(),World(),self.c,FakeBank())

    def test_late_action_coverage_feasible_goals_and_same_episode_draws(self):
        c=copy.deepcopy(self.c)
        c.update(training_tasks=['toy'],model=dict(chunk_steps=5,segments=12,state_representation='static'),
                 data=dict(action_repeat=2))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rows=[dict(id=f'train/toy/{i}',path=f'{i}.h5',task='toy',steps=100,
                split='train',mode='expert',expert_success=True,first_success_action=success)
                for i,success in enumerate([198,30])]
            (root/'manifest.json').write_text(json.dumps(dict(protocol=digest(c),mean=[0]*8,std=[1]*8,entries=rows)))
            for row in rows:
                with h5py.File(root/row['path'],'w') as f:
                    f['image_goals']=np.broadcast_to(np.arange(101,dtype=np.float32)[:,None,None],(101,2,8))
                    f['actions']=np.broadcast_to(np.arange(100,dtype=np.float32)[:,None],(100,8))
            new=AlignedSegments(root,c,'execution_revision',3072,4096)
            old=RevisionSegments(root,c,'latent_revision',3072,4096)
            starts=set(); last_index=None
            for i in range(len(new)):
                row,start,delta=new.sample_spec(i)
                self.assertEqual(row['id'],old.sample_spec(i)[0]['id'])
                self.assertLessEqual(start,row['first_success_action']//2)
                self.assertLessEqual(start+delta,100)
                self.assertGreaterEqual(delta,5)
                if row['id'].endswith('/0'):starts.add(start)
                if start==95:last_index=i
            self.assertEqual(starts,set(range(96)))
            example=new[last_index]
            self.assertEqual(int(example['a'][0,0]),95)
            self.assertEqual(int(example['a'][-1,0]),99)
            self.assertEqual(int(example['z'][2,0,0]),100)
            self.assertEqual(int(example['z'][3,0,0]),100)
            self.assertTrue(torch.equal(example['history_next'][-1],example['z'][0]))
            before={k:v.clone() for k,v in example.items() if k.startswith('history_')}
            with h5py.File(root/'0.h5','a') as f:f['image_goals'][96:]=999.
            after=new[last_index]
            self.assertTrue(all(torch.equal(v,after[k]) for k,v in before.items()))

    def test_prefix_error_changes_action_ranking_without_terminal_change(self):
        ctl=self.controller()
        target=torch.zeros(2,2,8);target[:,:,0]=1
        predicted=target[:,None].expand(-1,5,-1,-1).clone()
        predicted[1,0,:,1]=.1
        correction=torch.zeros(2,2)
        before,_,_,_=ctl.candidate_costs(predicted,target,correction)
        self.assertEqual(int(before.argmin()),0)
        correction[0,0]=.02
        after,_,_,_=ctl.candidate_costs(predicted,target,correction)
        self.assertEqual(int(after.argmin()),1)
        correction.fill_(-.1)
        conservative,raw,_,_=ctl.candidate_costs(predicted,target,correction)
        self.assertTrue(torch.equal(conservative,raw))

    def test_reuse_only_after_observation_exact_prefix_and_budget(self):
        ctl=self.controller();start,goal=torch.randn(1,2,8),torch.randn(1,2,8)
        action,_,_,_=ctl.plan(start,goal)
        self.assertEqual(ctl.world.calls,480)
        self.assertIsNone(ctl.previous_plan)
        plan=ctl.proposed_plan.clone()
        predicted=ctl.world.rollout(start,action[None,None])[:,0]
        self.assertTrue(torch.allclose(predicted,ctl.pending[0]))
        ctl.observe(predicted+.01)
        self.assertTrue(torch.equal(ctl.previous_plan,plan))
        context=ctl.model.prepare(start.expand(8,-1,-1),goal.expand(8,-1,-1),goal.expand(8,-1,-1),ctl.model_history(start,8))
        actions=ctl.initialize_actions(context,torch.ones(7,5,8))
        self.assertTrue(torch.equal(actions[:,1,:4],plan[None,1:].expand(8,-1,-1)))
        self.assertTrue(torch.equal(actions[:7,-1],torch.ones(7,5,8)))
        self.assertFalse(torch.equal(actions[7,-1],actions[7,0]))
        calls=ctl.world.calls
        next_action,_,_,_=ctl.plan(predicted+.01,goal)
        self.assertEqual(ctl.world.calls-calls,480)
        self.assertTrue(ctl.trace[-1]['warm_start_available'])
        self.assertTrue(torch.allclose(ctl.world.rollout(predicted+.01,next_action[None,None])[:,0],ctl.pending[0]))
        ctl.begin_episode()
        self.assertIsNone(ctl.previous_plan);self.assertIsNone(ctl.proposed_plan)
        self.assertEqual(ctl.stall_history,[]);self.assertEqual(ctl.history,[])

    def test_stalls_require_repeated_observations_and_decay_locally(self):
        ctl=self.controller()
        start=torch.zeros(1,2,8);start[:,:,0]=1
        target=start.clone();target[:,:,1]=.2
        predicted=target.clone()
        for index in range(3):
            ctl.pending=(predicted,start,target)
            ctl.trace.append(dict(corrected_prefix_target_progress=.02))
            ctl.observe(start)
            penalty,count=ctl.stall_penalties(start,target)
            self.assertEqual(int(count),index+1)
            if index<2:self.assertEqual(float(penalty),0)
        self.assertGreater(float(penalty),0)
        self.assertLessEqual(float(penalty),.020001)
        count_before=len(ctl.stall_history)
        ctl.observe(start)
        self.assertEqual(len(ctl.stall_history),count_before)
        far=torch.zeros_like(start);far[:,:,7]=1
        self.assertEqual(float(ctl.stall_penalties(far,target)[0]),0)
        self.assertEqual(float(ctl.stall_penalties(start,far)[0]),0)
        for _ in range(8):
            ctl.pending=(target,start,target)
            ctl.trace.append(dict(corrected_prefix_target_progress=.02))
            ctl.observe(target)
        self.assertEqual(float(ctl.stall_penalties(start,target)[0]),0)


if __name__=='__main__':unittest.main()
