"""No-update eight-GPU gradient and causal-data verification."""
import argparse, json, os, time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from ..common import config, seed_all, save_json
from .model import AdaptiveLeFlow
from .data import Segments
from .common import load_common_adapter
from .train import load_world


def main(a):
    rank=int(os.environ['LOCAL_RANK']);size=int(os.environ['WORLD_SIZE'])
    assert size==8
    torch.cuda.set_device(rank);torch.set_num_threads(2);dist.init_process_group('nccl')
    began=time.perf_counter();seed_all(3072);device=torch.device('cuda',rank)
    c=config(a.config);model=AdaptiveLeFlow(c).to(device);common=load_common_adapter(model,c)
    packed=json.loads((Path(c['baseline']['packed_cache'])/'complete.json').read_text())
    model.action_mean.copy_(torch.tensor(packed['action_mean'],device=device))
    model.action_std.copy_(torch.tensor(packed['action_std'],device=device))
    world=load_world(c,a.world,device)
    original={k:v.detach().clone() for k,v in model.state_dict().items()}
    wrapped=DDP(model,device_ids=[rank],broadcast_buffers=False)
    ds=Segments(a.root,c,'leflow_ttt',128064)
    rows=[ds[128000+rank*8+i] for i in range(8)]
    batch={k:torch.stack([r[k] for r in rows]).to(device) for k in rows[0]}
    seed_all(3072+997*rank);records=[]
    for depth in c['leflow_ttt']['train_inner_steps']:
        model.inner_steps=depth;wrapped.zero_grad(set_to_none=True)
        torch.cuda.synchronize();tick=time.perf_counter();loss,_=wrapped(batch,world);loss.backward();torch.cuda.synchronize()
        gradients=[]
        for name,p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and torch.isfinite(p.grad).all(),name
                gradients.append(p.grad.flatten())
        flat=torch.cat(gradients);other=flat.clone();dist.broadcast(other,0)
        assert torch.equal(flat,other) and all(p.grad is None for p in world.parameters())
        records.append(dict(inner_steps=depth,loss=float(loss.detach()),seconds=time.perf_counter()-tick,
                            synchronized_gradient_elements=flat.numel()))
    assert all(torch.equal(v,original[k]) for k,v in model.state_dict().items())
    elapsed=torch.tensor(time.perf_counter()-began,device=device);dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
    results=[None]*size;dist.all_gather_object(results,records)
    if rank==0:
        save_json(a.output,dict(gpu_count=8,global_batch=64,optimizer_updates=0,simulator_episodes=0,
            parameters_unchanged=True,gradient_sync_exact=True,world_gradients_absent=True,
            seconds=float(elapsed),gpu_hours=float(elapsed)*8/3600,rank_records=results,common_adapter=common))
        print('Eight-GPU checks passed',flush=True)
    dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('config','root','world','output'):p.add_argument('--'+k,required=True)
    main(p.parse_args())
