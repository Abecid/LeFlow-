#!/usr/bin/env python3
"""Bounded copied-checkpoint benchmark: no optimizer update or campaign mutation."""
import argparse, copy, json, time
from pathlib import Path
import torch
from torch.utils.data import default_collate
from flow_jepa.data import Segments
from flow_jepa.models import System
from flow_jepa.runtime.fused_training import FusedSystem
from flow_jepa.train import rng_state, restore_rng

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
torch.cuda.set_device(0);torch.set_num_threads(2);device=torch.device('cuda:0')
c=json.loads(Path(a.config).read_text());root=Path(a.root)
saved=torch.load(root/'runs/joint_flow_consistent_3072/last.pt',map_location='cpu',weights_only=False)
w=torch.load(root/'runs/world_3072/best.pt',map_location='cpu',weights_only=False)
old=System(c,'joint_flow_consistent').to(device);old.load_state_dict(saved['model']);old.train()
new=FusedSystem(c,'joint_flow_consistent').to(device);new.load_state_dict(saved['model']);new.train()
ws=System(c,'world');ws.load_state_dict(w['model']);world=ws.world.to(device).eval().requires_grad_(False)
d=Segments(root,c,'joint_flow_consistent',3072,1280000)
batch={k:v.to(device) for k,v in default_collate([d[saved['step']*64+i*4] for i in range(16)]).items() if k in ('z','a')}
restore_rng(saved['rng'][0]);rng=rng_state()
kwargs=dict(consistency_weight=.1,consistency_batch=2,consistency_steps=8)
def run(model, fused, backward=True):
 model.zero_grad(set_to_none=True);restore_rng(rng)
 if fused:
  loss,parts=model(batch,world,**kwargs)
  if backward:loss.backward()
 else:
  loss=0.;parts={}
  for lo in range(0,16,4):
   l,ps=model({k:v[lo:lo+4] for k,v in batch.items()},world,**kwargs)
   if backward:(l/4).backward()
   loss=loss+l.detach()/4
   for k,v in ps.items():parts[k]=parts.get(k,0)+v/4
 torch.cuda.synchronize()
 return float(loss),{k:float(v) for k,v in parts.items()},torch.cuda.get_rng_state().clone()
started=time.perf_counter();x=run(old,False);y=run(new,True)
assert torch.equal(x[2],y[2]),'CUDA RNG consumption differs'
old_grads=torch.cat([q.grad.flatten() for q in old.parameters()]);new_grads=torch.cat([q.grad.flatten() for q in new.parameters()])
relative=float(torch.linalg.vector_norm(new_grads-old_grads)/torch.linalg.vector_norm(old_grads))
maximum=float((new_grads-old_grads).abs().max())
assert abs(x[0]-y[0])<1e-5 and relative<1e-4,(x[0],y[0],relative,maximum)
# Check actual AdamW continuation, including saved momentum, on throwaway copies.
o1=torch.optim.AdamW(old.parameters());o1.load_state_dict(copy.deepcopy(saved['optimizer']))
o2=torch.optim.AdamW(new.parameters());o2.load_state_dict(copy.deepcopy(saved['optimizer']))
torch.nn.utils.clip_grad_norm_(old.parameters(),1.);torch.nn.utils.clip_grad_norm_(new.parameters(),1.)
o1.step();o2.step()
param_max=max(float((v-new.state_dict()[k]).abs().max()) for k,v in old.state_dict().items())
assert param_max<2e-6,param_max
# Restore both copies; timings never advance or save the research model.
old.load_state_dict(saved['model']);new.load_state_dict(saved['model']);times={}
for name,model,fused in [('micro4_accumulate4',old,False),('micro16_fused',new,True)]:
 run(model,fused);t=time.perf_counter()
 for _ in range(5):run(model,fused)
 times[name]=(time.perf_counter()-t)/5
result=dict(checkpoint_step=saved['step'],objective_old=x[:2],objective_fused=y[:2],cuda_rng_identical=True,gradient_relative_l2=relative,gradient_max_abs=maximum,adam_parameter_max_abs=param_max,seconds_per_update=times,speedup=times['micro4_accumulate4']/times['micro16_fused'],peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,benchmark_seconds=time.perf_counter()-started,precision='unchanged FP32; normal batch/reduction roundoff only',campaign_optimizer_updates=0)
Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
