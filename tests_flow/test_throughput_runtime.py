import copy
import json
from pathlib import Path
import pytest
import torch
from torch.utils.data import DistributedSampler, Subset
from flow_jepa.models import System
from flow_jepa.runtime.fused_training import FusedSystem
from flow_jepa.runtime.async_eval import EvaluationPool, child_env

@pytest.mark.parametrize('weight',[0.0,0.03,0.1])
def test_fused_rng_loss_gradient_and_consistency_subset(small_config,weight):
    torch.set_num_threads(1);torch.manual_seed(123)
    c=copy.deepcopy(small_config);c['model']['state_parameterization']='endpoint'
    old=System(c,'joint_flow_consistent');new=FusedSystem(c,'joint_flow_consistent')
    # Nonzero trained-like heads are needed to exercise all trunk gradients.
    for p in old.parameters():
        if p.ndim>1:torch.nn.init.normal_(p,0,.08)
    new.load_state_dict(old.state_dict())
    world=System(c,'world').world.eval().requires_grad_(False)
    b={'z':torch.randn(16,3,2,8),'a':torch.randn(16,2,2,4)}
    kw=dict(consistency_weight=weight,consistency_batch=2,consistency_steps=2)
    state=torch.get_rng_state();total=0.
    for lo in range(0,16,4):
        loss,_=old({k:v[lo:lo+4] for k,v in b.items()},world,**kw)
        (loss/4).backward();total+=float(loss.detach())/4
    final=torch.get_rng_state();torch.set_rng_state(state)
    loss,_=new(b,world,**kw);loss.backward()
    assert torch.equal(final,torch.get_rng_state())
    assert abs(total-float(loss.detach()))<1e-6
    for a,b in zip(old.parameters(),new.parameters()):
        torch.testing.assert_close(a.grad,b.grad,atol=1e-6,rtol=2e-4)


def test_resume_samples_same_for_physical_batch_16():
    data=Subset(range(1280000),range(6000*64,1280000))
    for rank in range(4):
        indices=list(DistributedSampler(data,4,rank,shuffle=False,drop_last=True))[:32]
        original=[indices[i:i+4] for i in range(0,32,4)]
        fused=[indices[i:i+16] for i in range(0,32,16)]
        assert [x for row in original[:4] for x in row]==fused[0]
        assert [x for row in original[4:] for x in row]==fused[1]
        assert [data[x] for x in fused[0]]==list(range(6000*64+rank,6001*64,4))


def test_async_durable_checkpoint_final_pool_and_ack(tmp_path):
    run=tmp_path/'run';run.mkdir();pool=EvaluationPool(tmp_path,run,tmp_path/'config',tmp_path/'world',3072)
    saved=dict(step=20000,method='joint_flow_consistent',protocol='p',manifest='m',code='c',world_hash='w')
    pool.submit(saved,final=True)
    path=pool.directory/'step_0020000.json';job=json.loads(path.read_text())
    assert job['devices']==list(range(8))
    with pytest.raises(ValueError):pool.submit(saved,final=True)
    out=Path(job['output']);out.parent.mkdir();out.write_text(json.dumps({'step':20000,'metrics':{'success_macro':.2}}))
    ready=pool.poll();assert len(ready)==1 and pool.outstanding()
    pool.acknowledge(path);assert not pool.outstanding() and pool.poll()==[]


def test_eval_child_does_not_inherit_training_rank(monkeypatch):
    monkeypatch.setenv('RANK','3');monkeypatch.setenv('MASTER_PORT','1234');monkeypatch.setenv('TORCHELASTIC_RUN_ID','old')
    env=child_env(range(8));assert 'RANK' not in env and 'MASTER_PORT' not in env and 'TORCHELASTIC_RUN_ID' not in env
    assert env['CUDA_VISIBLE_DEVICES']=='0,1,2,3,4,5,6,7'
    assert len(env['FLOW_EGL_DEVICES'].split(','))==8
