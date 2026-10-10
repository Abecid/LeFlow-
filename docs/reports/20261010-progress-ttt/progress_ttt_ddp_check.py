import os,sys,time,json
from pathlib import Path
base=Path('/tmp/mtxu-progress-ttt-check-20261010');sys.path.insert(0,str(base/'repo'))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from flow_jepa.common import config,seed_all,save_json
from flow_jepa.execution.train import component_types,load_world
rank=int(os.environ['LOCAL_RANK']);size=int(os.environ['WORLD_SIZE']);torch.cuda.set_device(rank);torch.set_num_threads(2)
dist.init_process_group('nccl');started=time.perf_counter();seed_all(3072)
c=config(base/'repo/config/flow_metaworld_progress_ttt.json');method,Policy,Dataset,_=component_types(c)
device=torch.device('cuda',rank);model=Policy(c).to(device)
world,_=load_world(c,'/home/mtxu/adam/LeFlow-experiments/20261009-repaired-comparison/campaign/runs/world_3072/best.pt',device)
original={k:v.clone() for k,v in model.state_dict().items()}
module=DDP(model,device_ids=[rank],broadcast_buffers=False,find_unused_parameters=True)
data=Dataset(base/'data',c,method,3072,64)
rows=[data[i] for i in range(rank*8,(rank+1)*8)]
batch={k:torch.stack([r[k] for r in rows]).to(device) for k in rows[0]};seed_all(3072+997*rank)
records=[]
for depth in (1,2,3):
 module.zero_grad(set_to_none=True);torch.cuda.synchronize();t=time.perf_counter()
 loss,metrics=module(batch,world,depth=depth);loss.backward();torch.cuda.synchronize()
 assert torch.isfinite(loss)
 gradients=[p.grad.flatten() for p in model.parameters() if p.grad is not None]
 flat=torch.cat(gradients);other=flat.clone();dist.broadcast(other,0)
 assert torch.isfinite(flat).all() and torch.equal(flat,other)
 assert all(p.grad is None for p in world.parameters())
 records.append(dict(depth=depth,loss=float(loss),forward_backward_seconds=time.perf_counter()-t,synchronized_gradient_elements=flat.numel()))
assert all(torch.equal(v,original[k]) for k,v in model.state_dict().items())
torch.cuda.synchronize();elapsed=torch.tensor(time.perf_counter()-started,device=device);dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
all_records=[None]*size;dist.all_gather_object(all_records,records)
if rank==0:
 save_json(base/'ddp-check.json',dict(gpu_count=size,global_batch=64,optimizer_updates=0,simulator_episodes=0,training_examples_only=True,parameters_unchanged=True,world_gradients_absent=True,gradient_sync_exact=True,depths=[1,2,3],seconds=float(elapsed),gpu_hours=float(elapsed)*size/3600,rank_records=all_records))
 print('DDP CHECK PASSED',float(elapsed),'seconds',flush=True)
dist.destroy_process_group()
