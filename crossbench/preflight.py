"""Actual TRAIN-data/native-reset/encoder/actuator check before optimization."""
import argparse
import json
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from flow_jepa.common import save_json, seed_all
from .spec import Spec
from .models import build
from .world import load_world
from .data import RouteBank, Windows
from .controller import Controller
from .environment import NativeEnv
from .evaluate import cases, synchronize
from .train import RETRIEVAL, code_hash


def run(cache,task,method,output):
    import hdf5plugin
    torch.cuda.set_device(0);device=torch.device('cuda',0);torch.set_num_threads(2)
    started=time.perf_counter();root=Path(cache);meta=json.loads((root/'manifest.json').read_text())
    if meta.get('fixture',False):raise ValueError('Native preflight rejects synthetic cache')
    s=Spec(task,method,dim=meta['dim'],primitive_dim=meta['primitive_dim']);seed_all(s.seed)
    world=load_world(meta,device);model=build(s).to(device).eval()
    bank=RouteBank(root,s,device) if method in RETRIEVAL else None
    ctrl=Controller(model,world,bank,s);env=NativeEnv(task,s.action_budget)
    row=cases(meta,s,count=1,split='train')[0];times=[]
    try:
        with h5py.File(meta['source_path'],'r') as src:
            rgb,goal=env.reset_from_source(src,row);a=dict(env.reset_audit)
            rgb2,goal2=env.reset_from_source(src,row)
            if a!=env.reset_audit or not np.array_equal(rgb,rgb2) or not np.array_equal(goal,goal2):
                raise ValueError('Native reset is not deterministic')
            for _ in range(3):
                synchronize(device);tick=time.perf_counter()
                z=world.encode(rgb[None]);g=world.encode(goal[None]);action=ctrl.plan(z,g)
                if action.shape!=(s.ad,) or not torch.isfinite(action).all():raise ValueError('Invalid preflight action')
                synchronize(device);times.append((time.perf_counter()-tick)*1000)
            native_actions=world.units.physical(action).reshape(s.action_block,s.primitive_dim).cpu().numpy()
            for command in native_actions:
                _,_,success,done=env.step(command)
                if success or done:break
            # Compare the exact cached representation to a fresh encode, without
            # using labels or evaluation cases to choose controller settings.
            cachedrow=next(r for r in meta['episodes'] if r['id']==row['episode'])
            with h5py.File(root/'cache.h5','r') as f:
                cached=torch.tensor(f['z'][cachedrow['cache_offset']+row['start']],device=device)
            delta=float((cached-z[0]).abs().max())
            if delta>1.e-3:raise ValueError(f'Fresh encoder differs from cached features: max_abs={delta}')
    finally:env.close()
    synchronize(device);seconds=time.perf_counter()-started
    report=dict(status='passed',task=task,method=method,source=code_hash(),optimizer_updates=0,
                training_cases_only=True,validation_cases_read=0,test_cases_read=0,
                reset_audit=a,action_dim=s.primitive_dim,block_dim=s.ad,latent_dim=s.dim,
                decision_ms=times,warm_mean_ms=float(np.mean(times[1:])),
                fresh_cache_max_abs=delta,seconds=seconds,gpu_hours=seconds/3600)
    save_json(output,report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('cache','task','method','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();run(a.cache,a.task,a.method,a.output)
