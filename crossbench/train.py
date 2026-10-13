"""Resumable DDP policy training, physical validation and separate resource accounting."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from flow_jepa.common import save_json, file_hash, checkpoint, seed_all
from .spec import Spec, METHODS
from .models import build
from .data import Windows, RouteBank
from .world import load_world
from .evaluate import evaluate, synchronize


RETRIEVAL={'execution_revision','action_flow_search','retrieval_repair_ttt'}


def code_hash():
    import hashlib
    root=Path(__file__).resolve().parents[1]
    paths=sorted([*root.joinpath('crossbench').glob('*.py'),*root.joinpath('flow_jepa').rglob('*.py'),
                  root/'latent_planner.py',root/'jepa.py',root/'module.py',root/'btm_jepa/data.py'])
    h=hashlib.sha256()
    for p in paths:h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'].cpu())
    if r['cuda'] is not None:torch.cuda.set_rng_state(r['cuda'].cpu())


class Objective(nn.Module):
    def __init__(self,model,world,bank):
        super().__init__();self.model=model
        # Frozen world/bank are shared, not DDP-managed trainable parameters.
        object.__setattr__(self,'world',world);object.__setattr__(self,'bank',bank)
    def forward(self,batch,step):return self.model.loss(batch,self.world,self.bank,step)


def maximum(value,device,size):
    x=torch.tensor(value,device=device,dtype=torch.float64)
    if size>1:dist.all_reduce(x,op=dist.ReduceOp.MAX)
    return float(x)


def run(s,root,run_dir,device,rank=0,size=1,*,world=None,eval_fn=evaluate,allow_fixture=False,online=True,preflight_only=False):
    root,run_dir=Path(root),Path(run_dir);run_dir.mkdir(parents=True,exist_ok=True)
    lock=None
    if rank==0:
        lock=(run_dir/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    meta=json.loads((root/'manifest.json').read_text())
    if meta.get('fixture',False) and not allow_fixture:raise ValueError('Fixture training requires explicit smoke mode')
    if not meta.get('fixture',False) and torch.device(device).type!='cuda':
        raise ValueError('Formal comparison requires the authorized CUDA runtime')
    if s.global_batch%size:raise ValueError('Global batch must divide rank count')
    if world is None:world=load_world(meta,device)
    seed_all(s.seed);model=build(s).to(device)
    bank=RouteBank(root,s,device) if s.method in RETRIEVAL else None
    optimizer=torch.optim.AdamW(model.parameters(),lr=s.learning_rate,weight_decay=s.weight_decay)
    identity=dict(spec=asdict(s),cache=meta['cache_sha256'],world=meta['checkpoint_sha256'],
                  source=code_hash(),ranks=size,fixture=meta.get('fixture',False))
    original_world={k:v.detach().clone() for k,v in world.state_dict().items()}
    data=Windows(root,s);probe=[data[rank*(s.global_batch//size)+j] for j in range(min(2,s.global_batch//size))]
    batch={k:torch.stack([r[k] for r in probe]).to(device) for k in probe[0]}
    wrapped=Objective(model,world,bank)
    if size>1:wrapped=DDP(wrapped,device_ids=[torch.device(device).index],find_unused_parameters=s.method=='retrieval_repair_ttt')
    checks=[];synchronize(device);pre_start=time.perf_counter()
    for step in (1,2):
        wrapped.zero_grad(set_to_none=True);loss,_=wrapped(batch,step);loss.backward()
        if not torch.isfinite(loss):raise FloatingPointError('Preflight loss is not finite')
        grads=[p.grad.flatten() for p in model.parameters() if p.grad is not None]
        flat=torch.cat(grads)
        if not torch.isfinite(flat).all():raise FloatingPointError('Preflight gradient is not finite')
        if size>1:
            ref=flat.clone();dist.broadcast(ref,0)
            if not torch.equal(flat,ref):raise ValueError('DDP gradient mismatch')
        checks.append(dict(depth=step,loss=float(loss.detach()),gradient_elements=len(flat)))
    if any(p.grad is not None for p in world.parameters()):raise ValueError('Frozen world received gradients')
    if not all(torch.equal(v,original_world[k]) for k,v in world.state_dict().items()):raise ValueError('Frozen world mutated')
    wrapped.zero_grad(set_to_none=True);synchronize(device)
    pretime=maximum(time.perf_counter()-pre_start,device,size)
    if rank==0:save_json(run_dir/'gradient-preflight.json',dict(checks=checks,ranks=size,optimizer_updates=0,
                 seconds=pretime,gpu_hours=pretime*size/3600,identity=identity))
    if preflight_only:return dict(status='preflight_passed',checks=checks)
    # Preflight generated no updates; reset sampling randomness before training.
    seed_all(s.seed+997*rank)
    start=0;spent=0.;eval_spent=0.;done=[];best=-1.;wandb_id=None;saved=None
    last=run_dir/'last.pt'
    if last.exists():
        saved=torch.load(last,map_location=device,weights_only=False)
        if saved['identity']!=identity:raise ValueError('Resume source/data/model/rank identity differs')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        start=saved['step'];spent=saved['spent'];done=saved['done'];best=saved['best'];eval_spent=saved['eval_spent']
        wandb_id=saved.get('wandb_id')
        ledger=json.loads((run_dir/'usage.json').read_text()) if (run_dir/'usage.json').exists() else {}
        # Replayed work after a crash remains charged, even if weights were lost.
        spent=max(spent,ledger.get('optimization_seconds',0.))
        restore_rng(saved['rng'][rank])
    elif (run_dir/'identity.json').exists():
        if json.loads((run_dir/'identity.json').read_text())!=identity:raise ValueError('Existing run directory belongs to another run')
    if rank==0:save_json(run_dir/'identity.json',identity)
    wb=None
    if rank==0 and online:
        import wandb
        wb=wandb.init(project='leflow-crossbench',name=f'{s.dataset}-{s.method}-{s.seed}',
                      group=s.protocol,id=wandb_id,resume='must' if wandb_id else None,
                      dir=str(run_dir),mode='online',config=identity)
        wandb_id=wb.id
        save_json(run_dir/'wandb.json',dict(id=wb.id,url=wb.url))
    if size>1:
        values=[wandb_id];dist.broadcast_object_list(values,0);wandb_id=values[0]
    subset=Subset(data,range(start*s.global_batch+rank,len(data),size))
    loader=DataLoader(subset,batch_size=s.global_batch//size,shuffle=False,num_workers=0,
        generator=torch.Generator().manual_seed(441),drop_last=True)
    step=start;began=time.perf_counter()
    def snapshot(path):
        rngs=[None]*size
        if size>1:dist.all_gather_object(rngs,rng_state())
        else:rngs=[rng_state()]
        if rank==0:checkpoint(path,dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),
            step=step,spent=spent,done=done,best=best,eval_spent=eval_spent,rng=rngs,wandb_id=wandb_id))
        if size>1:dist.barrier()
    def validation():
        nonlocal best,eval_spent
        state=rng_state();model.eval();path=run_dir/'checkpoints'/f'step_{step:07d}.pt';path.parent.mkdir(exist_ok=True)
        need=[not path.exists() if rank==0 else None]
        if size>1:dist.broadcast_object_list(need,0)
        if need[0]:snapshot(path)
        synchronize(device);tick=time.perf_counter()
        metrics=eval_fn(model,world,bank,s,root,run_dir/'validation'/f'step_{step:07d}',file_hash(path),device,rank,size)
        synchronize(device);eval_spent+=maximum(time.perf_counter()-tick,device,size)*size
        if metrics['success_rate']>best:
            best=metrics['success_rate']
            if rank==0:
                import shutil
                shutil.copy2(path,run_dir/'best.pt')
                save_json(run_dir/'best.json',dict(step=step,metrics=metrics,checkpoint_sha256=file_hash(path)))
        done.append(step)
        if rank==0 and wb:wb.log({'validation/'+k:v for k,v in metrics.items() if isinstance(v,(float,int))}|{'update':step})
        restore_rng(state);model.train();snapshot(last)
    try:
        if start==0:snapshot(last)
        # Resume an interrupted scheduled evaluation before taking another update.
        progress=max(step/s.updates,spent/s.optimization_gpu_seconds)
        if step and len(done)<min(4,int(progress*4+1.e-9)) and step not in done:validation()
        for batch in loader:
            if step>=s.updates or spent>=s.optimization_gpu_seconds:break
            synchronize(device);tick=time.perf_counter();step+=1
            model.train();optimizer.zero_grad(set_to_none=True)
            factor=min(1.,step/1000)*(.01+.99*(1+math.cos(math.pi*(step-1)/s.updates))/2)
            for group in optimizer.param_groups:group['lr']=s.learning_rate*factor
            batch={k:v.to(device) for k,v in batch.items()}
            loss,parts=wrapped(batch,step)
            if not torch.isfinite(loss):raise FloatingPointError('Training loss is not finite')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            synchronize(device);spent+=maximum(time.perf_counter()-tick,device,size)*size
            if rank==0:
                save_json(run_dir/'usage.json',dict(step=step,optimization_seconds=spent,evaluation_seconds=eval_spent,
                   optimization_gpu_hours=spent/3600,cap=s.optimization_gpu_seconds,
                   final_update_overrun_seconds=max(0.,spent-s.optimization_gpu_seconds)))
                if step==1 or step%50==0:
                    record=dict(step=step,loss=float(loss.detach()),**{k:float(v) for k,v in parts.items()})
                    with (run_dir/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                    if wb:wb.log(record)
            progress=max(step/s.updates,spent/s.optimization_gpu_seconds)
            if len(done)<min(4,int(progress*4+1.e-9)) and step not in done:
                snapshot(last);validation()
            elif step%1000==0:snapshot(last)
        if step not in done:validation()
        snapshot(last)
        if not all(torch.equal(v,original_world[k]) for k,v in world.state_dict().items()):raise ValueError('Frozen world changed during training')
        report=dict(status='completed',step=step,selected_success_rate=best,validation_steps=done,
            optimization_gpu_hours=spent/3600,evaluation_gpu_hours=eval_spent/3600,
            wall_seconds=time.perf_counter()-began,identity=identity,
            stop_reason='update_limit' if step>=s.updates else 'compute_cap')
        if rank==0:save_json(run_dir/'complete.json',report)
        return report
    finally:
        if wb:wb.finish()
        if lock:lock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','run-dir','task','method'):p.add_argument('--'+name,required=True)
    p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args();rank=int(os.getenv('RANK','0'));size=int(os.getenv('WORLD_SIZE','1'));local=int(os.getenv('LOCAL_RANK','0'))
    if size not in (1,4):raise ValueError('This campaign allows one or four GPUs per run')
    if not torch.cuda.is_available():raise RuntimeError('CUDA server required for formal training')
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(2)
    if size>1:dist.init_process_group('nccl')
    try:
        meta=json.loads((Path(args.cache)/'manifest.json').read_text())
        if file_hash(Path(args.cache)/'cache.h5')!=meta['cache_sha256']:raise ValueError('Cache checksum mismatch')
        s=Spec(args.task,args.method,dim=meta['dim'],primitive_dim=meta['primitive_dim'])
        run(s,args.cache,args.run_dir,device,rank,size,preflight_only=args.preflight_only)
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
