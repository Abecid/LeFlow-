import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
import h5py
import torch
from crossbench.spec import Spec
from crossbench.data import Windows, RouteBank
from crossbench.evaluate import cases, episode
from crossbench.world import ActionUnits
from flow_jepa.common import save_json, file_hash


def make_cache(root,task='pusht',dim=16):
    root=Path(root);pd=2 if task=='pusht' else 4
    rng=np.random.default_rng(17);rows=[]
    with h5py.File(root/'cache.h5','w') as f:
        z=rng.normal(size=(800,1,dim)).astype('f4')
        a=rng.uniform(-1,1,size=(800,pd)).astype('f4')
        f.create_dataset('z',data=z);f.create_dataset('a',data=a)
    for i in range(5):rows.append(dict(id=i,length=160,source_offset=160*i,cache_offset=160*i,split='train' if i<3 else 'val'))
    meta=dict(schema=1,task=task,dim=dim,patches=1,action_block=5,primitive_dim=pd,
        action_low=[-1]*pd,action_high=[1]*pd,action_mean=[0]*pd,action_std=[1]*pd,
        checkpoint_sha256='synthetic-world',source_signature='synthetic-source',
        source_path='unused-in-injected-tests',cache_sha256=file_hash(root/'cache.h5'),
        episodes=rows,fixture=True,test_episodes_read=0)
    save_json(root/'manifest.json',meta)
    return meta


class FakeEnvironment:
    def __init__(self,success_step=2):self.steps=0;self.success_step=success_step;self.reset_audit={}
    def reset_from_source(self,src,row):
        self.steps=0
        return np.zeros((8,8,3),np.uint8),np.ones((8,8,3),np.uint8)
    def step(self,a):
        self.steps+=1
        return np.zeros((8,8,3),np.uint8),0.,self.steps==self.success_step,False


class EncodingWorld:
    def __init__(self):self.calls=0;self.units=ActionUnits([-1]*2,[1]*2,[0]*2,[1]*2)
    def encode(self,rgb):return torch.zeros(len(rgb),1,16)


class ConstantController:
    def begin_episode(self):self.trace=[]
    def observe(self,z):pass
    def plan(self,z,g,budget=None):
        if budget is not None:budget.check()
        return torch.zeros(10)


class TestDataEvaluation(unittest.TestCase):
    def test_window_shapes_causality_and_split(self):
        with tempfile.TemporaryDirectory() as root:
            meta=make_cache(root);s=Spec('pusht','retrieval_repair_ttt',dim=16,width=32)
            ds=Windows(root,s,length=100)
            for i in range(30):
                b=ds[i];self.assertLess(int(b['episode']),3)
                self.assertEqual(b['z'].shape,(6,1,16));self.assertEqual(b['a'].shape,(5,10))
                if b['history_mask'].any():self.assertTrue(torch.equal(b['history_next'][-1],b['z'][0]))
                self.assertTrue(torch.equal(b['a'],ds[i]['a']))
            with self.assertRaises(ValueError):Windows(root,s,split='test')
            bank=RouteBank(root,s,'cpu');self.assertLess(int(bank.episode.max()),3)
            b=ds[0];a=bank.training_query(b['z'][:1],b['z'][-1:],b['episode'][None])
            own=bank.chunks[bank.episode==b['episode']]
            self.assertFalse(any(torch.equal(a[0],v) for v in own))
            self.assertEqual(bank.query(b['z'][:1],b['goal'][None])[0].shape,(7,1,16))
    def test_case_inventory_independent_of_method(self):
        with tempfile.TemporaryDirectory() as root:
            meta=make_cache(root)
            a=Spec('pusht','leflow',eval_episodes=2)
            b=Spec('pusht','vanilla_action_flow',eval_episodes=2)
            self.assertEqual(cases(meta,a),cases(meta,b))
            self.assertTrue(all(r['episode']>=3 for r in cases(meta,a)))
    def test_success_stops_inside_action_block(self):
        s=Spec('pusht','vanilla_action_flow',dim=16,width=32)
        row=dict(id='fixture',seed=2)
        env=FakeEnvironment()
        record,_=episode(env,EncodingWorld(),ConstantController(),None,row,s,'cpu')
        self.assertTrue(record['success']);self.assertEqual(record['steps'],2)
        self.assertEqual(env.steps,2)
    def test_no_command_after_deadline(self):
        s=Spec('pusht','vanilla_action_flow',dim=16,width=32)
        times=iter(range(0,1000,20));env=FakeEnvironment()
        record,_=episode(env,EncodingWorld(),ConstantController(),None,dict(id='fixture',seed=2),s,'cpu',clock=lambda:next(times))
        self.assertTrue(record['timeout']);self.assertEqual(env.steps,0)


if __name__=='__main__':unittest.main()
