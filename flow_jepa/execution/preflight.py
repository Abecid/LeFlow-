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
    world_hash_before={k:v.clone() for k,v in world.state_dict().items()}
    loss,parts=model(batch,world,weight=0.1);loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None for p in world.parameters())
    assert all(torch.equal(v,world_hash_before[k]) for k,v in world.state_dict().items())
    bank=RouteBank(args.bank,c,file_hash(root/'manifest.json'),device)
    encoder=Encoder(c['encoder'],root,device)
    row=next(r for r in manifest['entries'] if r['split']=='train' and r['expert_success'])
    with h5py.File(root/row['path'],'r') as f:
        initial=f['initial_rgb'][:]
        goal=(torch.tensor(f['goal'][:],device=device).float()-bank.mean)/bank.std
    model.eval();controller=ControllerType(model,world,c,bank)
    if method=='latent_revision':
        history=model.factual_history(batch,world)
        # Training transitions only; exercise the populated memory path for
        # latency, without pretending these are newly executed policy actions.
        for j in range(model.history_steps):
            if history['mask'][0,j]:
                controller.history.append(tuple(history[k][0:1,j].clone() for k in ('start','next','predicted')))
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
