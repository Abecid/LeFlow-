"""Case-level native-success evaluation with hard action and controller clocks."""
from __future__ import annotations
import hashlib
import json
import math
import time
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.distributed as dist
from flow_jepa.common import save_json, digest, seed_all
from flow_jepa.budget import PlanningBudget, PlanningBudgetExceeded
from .controller import Controller
from .environment import NativeEnv


def cases(meta,s,count=None,split='val'):
    if split not in ('train','val'):raise ValueError('Sealed test cases are not exposed')
    count=count or s.eval_episodes
    rows=[r for r in meta['episodes'] if r['split']==split and r['length']>s.goal_offset]
    if len(rows)<count:raise ValueError(f'Need {count} eligible {split} episodes, found {len(rows)}')
    rng=np.random.default_rng(s.seed+(719 if split=='val' else 193))
    selected=rng.choice(len(rows),size=count,replace=False)
    out=[]
    for i,j in enumerate(selected):
        r=rows[int(j)];t=int(rng.integers(r['length']-s.goal_offset))
        out.append(dict(id=f'{s.dataset}/{split}/{r["id"]:07d}/{t:06d}',episode=r['id'],
                        source_offset=r['source_offset'],start=t,goal_offset=s.goal_offset,
                        seed=s.seed+100000+i))
    return out


def synchronize(device):
    if torch.device(device).type=='cuda':torch.cuda.synchronize(device)


@torch.no_grad()
def episode(env,world,controller,source,row,s,device,clock=time.perf_counter):
    rgb,goal_rgb=env.reset_from_source(source,row)
    controller.begin_episode();seed_all(s.seed+row['seed'])
    spent=0.;actions_done=0;success=False;timeout=False;terminated=False;reward=0.
    latency=[];frames=[rgb.copy()];commands=[];goal=None;before=world.calls
    while actions_done<s.action_budget:
        synchronize(device);started=clock()
        budget=PlanningBudget(s.controller_seconds-spent,clock=clock)
        try:
            budget.check()
            z=world.encode(rgb[None])
            if goal is None:goal=world.encode(goal_rgb[None])
            controller.observe(z)
            unit=controller.plan(z,goal,budget=budget)
            if unit.shape!=(s.ad,) or not torch.isfinite(unit).all():
                raise ValueError('Controller returned an invalid executable block')
            physical=world.units.physical(unit).reshape(s.action_block,s.primitive_dim).cpu().numpy()
            synchronize(device);budget.check()
        except PlanningBudgetExceeded:
            timeout=True;synchronize(device)
        elapsed=clock()-started;spent+=elapsed;latency.append(elapsed*1000)
        if timeout:break  # No action computed after the deadline is executed.
        for a in physical[:s.action_budget-actions_done]:
            rgb,r,hit,done=env.step(a);actions_done+=1;reward+=r
            commands.append(a.tolist());frames.append(rgb.copy())
            success=success or hit;terminated=done
            if success or done:break
        if success or terminated:break
    reason='success' if success else ('controller_timeout' if timeout else
             ('native_termination' if terminated else 'action_limit'))
    result=dict(**row,success=bool(success),steps=actions_done,timeout=timeout,
        failure_reason=reason,controller_seconds=spent,decision_latency_ms=latency,
        world_transitions=world.calls-before,total_reward=reward,
        reset_audit=getattr(env,'reset_audit',{}),actions=commands,trace=controller.trace)
    indices=np.linspace(0,len(frames)-1,min(6,len(frames)),dtype=int)
    return result,dict(frames=np.stack([frames[j] for j in indices]),
                      action_indices=indices,goal=goal_rgb)


def summarize(records):
    n=len(records)
    if not n:raise ValueError('No evaluation cases')
    wins=sum(r['success'] for r in records);p=wins/n;z=1.96
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    latency=[v for r in records for v in r['decision_latency_ms']]
    return dict(successes=wins,episodes=n,success_rate=p,
        wilson95=[max(0.,center-half),min(1.,center+half)],
        timeouts=sum(r['timeout'] for r in records),
        mean_decision_ms=float(np.mean(latency)) if latency else 0.,
        mean_steps=float(np.mean([r['steps'] for r in records])),
        world_transitions=sum(r['world_transitions'] for r in records),
        failure_ids=[r['id'] for r in records if not r['success']])


def evaluate(model,world,bank,s,root,out,checkpoint_hash,device,rank=0,size=1):
    import hdf5plugin  # Native source codec registration; never reads test rows.
    root,out=Path(root),Path(out);out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((root/'manifest.json').read_text());plan=cases(meta,s)
    identity=dict(spec=s.identity(),cache=meta['cache_sha256'],checkpoint=checkpoint_hash,
                  cases=digest(plan),source_signature=meta['source_signature'])
    if rank==0:
        ip=out/'identity.json'
        if ip.exists() and json.loads(ip.read_text())!=identity:raise ValueError('Evaluation journal identity changed')
        save_json(ip,identity)
    if size>1:dist.barrier()
    controller=Controller(model,world,bank,s);records=[];env=NativeEnv(s.dataset,s.action_budget)
    try:
        with h5py.File(meta['source_path'],'r') as src:
            for idx in range(rank,len(plan),size):
                row=plan[idx];path=out/'cases'/f'{idx:05d}.json'
                if path.exists():
                    record=json.loads(path.read_text())
                    if record['identity']!=identity or record['id']!=row['id']:
                        raise ValueError('Mismatched resumed case record')
                else:
                    record,images=episode(env,world,controller,src,row,s,device)
                    record['identity']=identity
                    imagepath=out/'trajectories'/f'{idx:05d}.npz';imagepath.parent.mkdir(exist_ok=True)
                    np.savez_compressed(imagepath,**images)
                    save_json(path,record)
                records.append(record)
    finally:env.close()
    if size>1:
        gathered=[None]*size;dist.all_gather_object(gathered,records)
        records=[r for shard in gathered for r in shard]
    records.sort(key=lambda r:r['id'])
    if len({r['id'] for r in records})!=len(plan):raise ValueError('Incomplete or duplicate case inventory')
    metrics=summarize(records)
    if rank==0:save_json(out/'report.json',dict(identity=identity,metrics=metrics,records=records))
    return metrics
