"""Bounded checks on training examples only; no parameter updates or simulator."""
import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from ..common import config, file_hash, save_json, seed_all
from ..vision import Encoder
from ..models import distance
from .model import ChunkPolicy, ExecutionSegments
from .train import load_world, component_types
from .bank import RouteBank
from .controller import ExecutionController


def run(args):
    torch.cuda.set_device(0);torch.set_num_threads(2)
    started=time.perf_counter();device=torch.device('cuda:0')
    c=config(args.config);root=Path(args.root);seed_all(3072)
    method,Policy,Dataset,ControllerType=component_types(c)
    manifest=json.loads((root/'manifest.json').read_text())
    model=Policy(c).to(device)
    world,_=load_world(c,args.world,device)
    data=Dataset(root,c,method,3072,64)
    examples=[data[i] for i in range(8)]
    batch={k:torch.stack([row[k] for row in examples]).to(device) for k in examples[0]}
    data_checks={}
    if method=='latent_revision':
        previous=ExecutionSegments(root,c,'controller_grounded',3072,64)
        for i,example in enumerate(examples):
            old=previous[i]
            assert torch.equal(example['z'],old['z']) and torch.equal(example['a'],old['a'])
            if example['history_mask'][-1]:
                assert torch.equal(example['history_next'][-1],example['z'][0])
        data_checks=dict(exact_previous_main_sample_checks=8,causal_history_end_checks=True)
    if method in ('execution_revision','flow_reasoning','progress_ttt','guided_ttt'):
        from .revision import RevisionSegments
        previous=RevisionSegments(root,c,'latent_revision',3072,10000)
        specs=[data.sample_spec(i) for i in range(10000)]
        for i,(row,start,delta) in enumerate(specs):
            assert row['id']==previous.sample_spec(i)[0]['id']
            assert start+5<=row['steps'] and start+delta<=row['steps']
            assert start<=row['first_success_action']//c['data']['action_repeat']
        late=[i for i,(_,start,_) in enumerate(specs) if start>40]
        assert late, 'Repaired sampler must expose late action windows'
        for i in list(range(8))+late[:8]:
            example=data[i]
            if example['history_mask'][-1]:
                assert torch.equal(example['history_next'][-1],example['z'][0])
        data_checks=dict(same_task_episode_draws=10000,window_and_goal_bounds_checked=10000,
            late_windows_in_first_10000=len(late),late_real_examples_checked=8,
            causal_history_end_checks=True,sampling_change='all valid five-step pre-success starts')
    world_hash_before={k:v.clone() for k,v in world.state_dict().items()}
    loss,parts=model(batch,world,weight=0. if method in ('flow_reasoning','progress_ttt','guided_ttt') else 0.1);loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None for p in world.parameters())
    assert all(torch.equal(v,world_hash_before[k]) for k,v in world.state_dict().items())
    bank=RouteBank(args.bank,c,file_hash(root/'manifest.json'),device)
    encoder=Encoder(c['encoder'],root,device)
    row=next(r for r in manifest['entries'] if r['split']=='train' and r['expert_success'])
    with h5py.File(root/row['path'],'r') as f:
        initial=f['initial_rgb'][:]
        goal=(torch.tensor(f['goal'][:],device=device).float()-bank.mean)/bank.std
    model.eval();controller=ControllerType(model,world,c,bank)
    if method in ('latent_revision','execution_revision','flow_reasoning','progress_ttt','guided_ttt'):
        history=model.factual_history(batch,world)
        # Training transitions only; exercise the populated memory path for
        # latency, without pretending these are newly executed policy actions.
        for j in range(model.history_steps):
            if history['mask'][0,j]:
                controller.history.append(tuple(history[k][0:1,j].clone() for k in ('start','next','predicted')))
                if method in ('progress_ttt','guided_ttt'):
                    controller.action_history.append(history['action'][0,j].clone())
        if method in ('execution_revision','flow_reasoning','progress_ttt','guided_ttt'):
            # Exercise warm-start and stall-memory compute using only recorded
            # training transitions. These are NOT new policy execution outcomes.
            controller.previous_plan=batch['a'][0].detach().clone()
            for j in range(c['execution_revision']['stall_memory']):
                sample=j % len(batch['a']);q=batch['z'][sample:sample+1,2]
                past=batch['history_start'][sample:sample+1,-1]
                actual=batch['history_next'][sample:sample+1,-1]
                predicted=history['predicted'][sample:sample+1,-1]
                progress=distance(past,q)-distance(actual,q)
                forecast=distance(past,q)-distance(predicted,q)
                error=(forecast-progress).clamp_min(0)*((forecast>0)&(progress<=0))
                controller.stall_history.append((past,q,error))
    latencies=[]
    for i in range(6):
        torch.cuda.synchronize();begin=time.perf_counter()
        clip=np.repeat(initial[None],c['data']['history_frames'],axis=0)
        z=(encoder(clip[None]).float()-bank.mean)/bank.std
        action,_,_,_=controller.plan(z,goal[None])
        torch.cuda.synchronize();elapsed=time.perf_counter()-begin
        if i:latencies.append(elapsed*1000)
        assert torch.isfinite(action).all() and float(action.abs().max())<=1
    report=dict(training_examples_only=True,optimizer_updates=0,simulator_episodes=0,
        batch_loss=float(loss),parts={k:float(v) for k,v in parts.items()},
        full_controller_latency_ms=latencies,mean_ms=float(np.mean(latencies)),
        max_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
        seconds=time.perf_counter()-started,gpu_count=1,
        gpu_hours=(time.perf_counter()-started)/3600,
        parameters=sum(p.numel() for p in model.parameters()),
        frozen_world_unchanged=True,bank_states=len(bank.raw),method=method,**data_checks)
    save_json(args.output,report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('config','root','world','bank','output'):p.add_argument('--'+name,required=True)
    run(p.parse_args())
