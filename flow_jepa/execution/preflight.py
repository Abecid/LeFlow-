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
from .train import load_world
from .bank import RouteBank
from .controller import ExecutionController


def run(args):
    torch.cuda.set_device(0);torch.set_num_threads(2)
    started=time.perf_counter();device=torch.device('cuda:0')
    c=config(args.config);root=Path(args.root);seed_all(3072)
    manifest=json.loads((root/'manifest.json').read_text())
    model=ChunkPolicy(c).to(device)
    world,_=load_world(c,args.world,device)
    data=ExecutionSegments(root,c,'controller_grounded',3072,64)
    batch={k:torch.stack([data[i][k] for i in range(8)]).to(device) for k in ('z','a')}
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
    model.eval();controller=ExecutionController(model,world,c,bank)
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
        frozen_world_unchanged=True,bank_states=len(bank.raw))
    save_json(args.output,report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('config','root','world','bank','output'):p.add_argument('--'+name,required=True)
    run(p.parse_args())
