"""Training-data-only GPU verification. No optimizer updates or eval episodes."""
import argparse,json,time,os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from ..common import config,save_json,seed_all,file_hash
from ..baselines.prepare import prepare
from ..baselines.train import load_world
from ..baselines.data import BaselineSegments
from .data import Segments
from .model import AdaptiveLeFlow
from .controller import AdaptiveController
from .common import load_common_adapter


def main(a):
    c=config(a.config);torch.set_num_threads(2);seed_all(3072);device=torch.device('cuda:0')
    prepare(c,a.source,a.root,a.world,a.source_config)
    model=AdaptiveLeFlow(c).to(device);common=load_common_adapter(model,c)
    packed=json.loads((Path(c['baseline']['packed_cache'])/'complete.json').read_text())
    model.action_mean.copy_(torch.tensor(packed['action_mean'],device=device));model.action_std.copy_(torch.tensor(packed['action_std'],device=device))
    world=load_world(c,a.world,device)
    ds=Segments(a.root,c,'leflow_ttt',10000);base=BaselineSegments(a.root,c,'leflow_release',10000)
    for i in range(256):
        got,native=ds[i],base[i]
        assert torch.equal(got['z'],native['z']) and torch.equal(got['a'],native['a'])
        assert int(got['history_mask'].sum())==min(int(got['query_start'])//5,4)
        if got['history_mask'][-1]:assert torch.equal(got['history_next'][-1],got['z'][0])
    batch=next(iter(DataLoader(ds,batch_size=8,num_workers=0)))
    batch={k:v.to(device) for k,v in batch.items()}
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    results=[];begin=time.perf_counter()
    for depth in c['leflow_ttt']['train_inner_steps']:
        model.inner_steps=depth;model.train();model.zero_grad(set_to_none=True)
        torch.cuda.synchronize();tick=time.perf_counter();loss,parts=model(batch,world);loss.backward();torch.cuda.synchronize()
        assert torch.isfinite(loss)
        for name,p in model.named_parameters():
            if p.requires_grad:assert p.grad is not None and torch.isfinite(p.grad).all(),name
        assert all(p.grad is None for p in world.parameters())
        results.append(dict(inner_steps=depth,seconds=time.perf_counter()-tick,loss=float(loss),parts={k:float(v) for k,v in parts.items()}))
    model.eval();model.inner_steps=c['leflow_ttt']['inner_steps']
    ctl=AdaptiveController(model,world,c);start=batch['z'][:1,0];goal=batch['z'][:1,-1]
    latencies=[]
    for i in range(5):
        torch.cuda.synchronize();tick=time.perf_counter();action,_,_,_=ctl.plan(start,goal);torch.cuda.synchronize()
        assert action.shape==(40,)
        latencies.append(1000*(time.perf_counter()-tick))
        if i:assert ctl.decisions[-1]['history_chunks']==min(i,4)
        ctl.observe(batch['z'][:1,-1])
    assert any(d['fast_weight_delta']>0 for d in ctl.decisions[1:])
    assert all(d['world_transitions']==325 for d in ctl.decisions)
    ctl.begin_episode();assert not ctl.history and ctl.pending is None and not ctl.decisions
    for k,v in model.state_dict().items():assert torch.equal(v.cpu(),state[k]),k
    result=dict(common_adapter=common,checks=results,controller_latency_ms=latencies,
        warm_controller_mean_ms=sum(latencies[1:])/4,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
        seconds=time.perf_counter()-begin,optimizer_updates=0,simulator_episodes=0,
        same_baseline_queries_checked=256,causal_macro_history_verified=True,
        all_parameters_unchanged=True,world_gradients_none=True,episode_reset_verified=True,
        parameters=sum(p.numel() for p in model.parameters()))
    save_json(a.output,result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('config','source','root','world','source-config','output'):p.add_argument('--'+k,required=True)
    main(p.parse_args())
