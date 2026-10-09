import contextlib,json,os,time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import default_collate
from flow_jepa.common import distributed
from flow_jepa.data import Segments
from flow_jepa.models import System
from flow_jepa.runtime.fused_training import FusedSystem
from flow_jepa.train import rng_state,restore_rng
rank,size,device=distributed();assert size==4
root=Path('/home/mtxu/adam/LeFlow-experiments/20261009-repaired-comparison/campaign');c=json.loads((root.parent/'repo/config/flow_metaworld_repair.json').read_text())
saved=torch.load(root/'runs/joint_flow_consistent_3072/last.pt',map_location='cpu',weights_only=False)
w=torch.load(root/'runs/world_3072/best.pt',map_location='cpu',weights_only=False)
world=System(c,'world');world.load_state_dict(w['model']);world=world.world.to(device).eval().requires_grad_(False)
old=System(c,'joint_flow_consistent').to(device);old.load_state_dict(saved['model'])
new=FusedSystem(c,'joint_flow_consistent').to(device);new.load_state_dict(saved['model'])
a=DDP(old,device_ids=[device.index]);b=DDP(new,device_ids=[device.index])
data=Segments(root,c,'joint_flow_consistent',3072,1280000)
batch={k:v.to(device) for k,v in default_collate([data[saved['step']*64+rank+i*4] for i in range(16)]).items() if k in ('z','a')}
restore_rng(saved['rng'][rank]);rng=rng_state();kw=dict(consistency_weight=.1,consistency_batch=2,consistency_steps=8)
def step(model,fused):
 model.zero_grad(set_to_none=True);restore_rng(rng)
 if fused:
  loss,_=model(batch,world,**kw);loss.backward()
 else:
  loss=0
  for i in range(4):
   with model.no_sync() if i<3 else contextlib.nullcontext():
    value,_=model({k:v[i*4:i*4+4] for k,v in batch.items()},world,**kw)
    (value/4).backward();loss=loss+value.detach()/4
 torch.cuda.synchronize();return float(loss.detach()),torch.cuda.get_rng_state().clone()
started=time.perf_counter();x=step(a,False);y=step(b,True);assert torch.equal(x[1],y[1])
ga=torch.cat([p.grad.flatten() for p in old.parameters()]);gb=torch.cat([p.grad.flatten() for p in new.parameters()]);rel=float((ga-gb).norm()/ga.norm());assert rel<1e-4,rel
times={}
for name,model,fused in [('original',a,False),('fused',b,True)]:
 step(model,fused);dist.barrier();t=time.perf_counter()
 for _ in range(3):step(model,fused)
 v=torch.tensor((time.perf_counter()-t)/3,device=device);dist.all_reduce(v,op=dist.ReduceOp.MAX);times[name]=float(v)
rows=[None]*4;dist.all_gather_object(rows,dict(rank=rank,gradient_relative_l2=rel,loss_old=x[0],loss_fused=y[0],rng_identical=True))
if rank==0:
 out=dict(checkpoint_step=saved['step'],ranks=rows,seconds_per_update=times,speedup=times['original']/times['fused'],diagnostic_seconds=time.perf_counter()-started,gpu_count=4,campaign_updates=0)
 (root/'fused-ddp-benchmark.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
dist.destroy_process_group()
