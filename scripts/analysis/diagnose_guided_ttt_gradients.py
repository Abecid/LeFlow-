import sys,time,json
from pathlib import Path
sys.path.insert(0,'/home/mtxu/adam/LeFlow-experiments/20261010-guided-ttt/repo')
import torch
from torch.nn import functional as F
from flow_jepa.execution.train import component_types,load_world
from flow_jepa.execution.progress_ttt import patch_cost
from flow_jepa.common import seed_all

torch.set_num_threads(2);torch.cuda.set_device(0);seed_all(3072);began=time.perf_counter()
p=Path('/home/mtxu/adam/LeFlow-experiments/20261010-guided-ttt');c=json.loads((p/'repo/config/flow_metaworld_guided_ttt.json').read_text());device='cuda:0'
_,Policy,Dataset,_=component_types(c);model=Policy(c).to(device).eval()
saved=torch.load(p/'campaign/runs/guided_ttt_3072/best.pt',map_location='cpu',weights_only=False);model.load_state_dict(saved['model'])
world,_=load_world(c,'/home/mtxu/adam/LeFlow-experiments/20261009-repaired-comparison/campaign/runs/world_3072/best.pt',device)
data=Dataset(p/'campaign',c,'guided_ttt',3072,128)
rows=[data[i] for i in range(32)];b={k:torch.stack([r[k] for r in rows]).to(device) for k in rows[0]};z=b['z'];a=b['a']
with torch.no_grad():h=model.factual_history(b,world);context=model.prepare(z[:,0],z[:,2],z[:,3],h);plans=model.propose_context(context,4)
x=plans[:,:,0].detach().requires_grad_(True)
base,cor=model.energy_maps(context,x)
world_z=world.rollout(z[:,0,None].expand(-1,4,-1,-1).flatten(0,1),x.flatten(0,1)[:,None])[:,0].reshape(32,4,*z.shape[2:])
wcost=patch_cost(world_z,z[:,None,2]).mean(-1)
gs=torch.autograd.grad((base+cor).mean(-1).sum()/model.error_scale,x,retain_graph=True)[0]
gb=torch.autograd.grad(base.mean(-1).sum()/model.error_scale,x,retain_graph=True)[0]
gw=torch.autograd.grad(wcost.sum()/model.error_scale,x,retain_graph=True)[0]
gcw=torch.autograd.grad((wcost+cor.mean(-1)).sum()/model.error_scale,x)[0]
cos=F.cosine_similarity(gs,gw,dim=-1)
cost=(base+cor).mean(-1).detach();wd=wcost.detach();d1=cost[:,:,None]-cost[:,None,:];d2=wd[:,:,None]-wd[:,None,:];mask=torch.triu(torch.ones(4,4,device=device,dtype=torch.bool),1)
pair=((d1*d2)[:,:, :]>0)[:,mask].float()
result=dict(training_examples=32,optimizer_updates=0,simulator_calls=0,final_tests_read=False,selected_step=saved['step'],
 guidance_vs_world_gradient_cosine_mean=float(cos.mean()),guidance_vs_world_gradient_negative_fraction=float((cos<0).float().mean()),
 base_vs_world_gradient_cosine_mean=float(F.cosine_similarity(gb,gw,dim=-1).mean()),
 base_vs_world_gradient_negative_fraction=float((F.cosine_similarity(gb,gw,dim=-1)<0).float().mean()),
 guidance_vs_corrected_world_gradient_cosine_mean=float(F.cosine_similarity(gs,gcw,dim=-1).mean()),
 guidance_vs_corrected_world_negative_fraction=float((F.cosine_similarity(gs,gcw,dim=-1)<0).float().mean()),
 base_world_pairwise_order_agreement=float((((base.mean(-1)[:,:,None]-base.mean(-1)[:,None,:]).detach()*d2)>0)[:,mask].float().mean()),
 guidance_rms=float(gs.square().mean().sqrt()),world_gradient_rms=float(gw.square().mean().sqrt()),
 guidance_world_pairwise_order_agreement=float(pair.mean()),cost_mae=float((cost-wd).abs().mean()),
 plain_source='All diagnostics use first32 fixed training-window draws and frozen selected20k model; world gradients/ordering are model predictions, not physical outcome ground truth.')
# Time the real online batch of8 targets with four populated transitions.
bb={k:v[:8] for k,v in b.items()};zz=bb['z'];hh=model.factual_history(bb,world)
def timed(name,fn):
 values=[];out=None
 for i in range(8):
  torch.cuda.synchronize();t=time.perf_counter();out=fn();torch.cuda.synchronize()
  if i>=3:values.append(1000*(time.perf_counter()-t))
 result[name+'_ms']=sum(values)/len(values)
 return out
with torch.no_grad():
 ctx=timed('prepare',lambda:model.prepare(zz[:,0],zz[:,2],zz[:,3],hh))
 plans=timed('propose',lambda:model.propose_context(ctx,4))
 pred=timed('world_5step',lambda:world.rollout(zz[:,0,None].expand(-1,4,-1,-1).flatten(0,1),plans.flatten(0,1)).reshape(8,4,5,*zz.shape[2:]))
 scores=timed('score',lambda:model.score_candidates(ctx,plans,pred,zz[:,2]))
 revised=timed('revise',lambda:model.revise(ctx,plans,scores))
 from flow_jepa.execution.flow_reasoning import FlowReasoningPolicy
 timed('propose_without_guidance_timing_only',lambda:FlowReasoningPolicy.propose_context(model,ctx,4))
result['seconds']=time.perf_counter()-began;result['gpu_count']=1;result['gpu_hours']=result['seconds']/3600
print(json.dumps(result,indent=2))
