"""P&P transfer to retained LeFlow weights and the original matched execution.

These are checkpoint adaptations, not author-checkpoint/native-paper results.
No training, new representation, action rule, case, or controller limit is added.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json,time,os
from pathlib import Path
import torch
from .core import RefinementConfig, Work, sample_interiors
from .native import state_fingerprint
from .protocol import atomic_json, digest_json, sha256_file


class SharedController:
    def __init__(self,model,world,spec,method):
        if spec.method!='leflow': raise ValueError('Requires the retained LeFlow checkpoint')
        self.model=model.eval().requires_grad_(False);self.world=world.eval().requires_grad_(False)
        self.s=spec;self.method=method;self.config=RefinementConfig.from_dict(method['refinement'])
        self.config.validate(method['flow_steps']);self.begin_episode()

    def begin_episode(self):self.trace=[];self.decisions=0;self.previous_prediction=None
    def observe(self,z):
        if self.previous_prediction is not None:
            predicted,target=self.previous_prediction
            self.trace[-1]['observed_prefix_prediction_mse']=float((predicted-z[:,0]).square().mean())
            self.trace[-1]['observed_prefix_goal_mse']=float((z[:,0]-target).square().mean())
            self.previous_prediction=None

    def plan(self,z,goal,*,budget=None):
        s=self.s;work=Work();tick=time.perf_counter();cost=None;selected=None;complete=False;all_costs=None
        check=budget.check if budget else None
        def decode(paths):
            if check:check()
            raw=self.model.inverse(paths[:,:-1].flatten(0,1),paths[:,1:].flatten(0,1))
            a=self.world.from_world(raw.reshape(len(paths),s.horizon,s.ad))
            work.inverse_calls+=1;work.inverse_transitions+=len(paths)*s.horizon
            return a
        def consequences(start,a):
            pred=self.world.rollout(start[:,None],a,budget=budget)[:,:,0]
            work.world_calls+=s.horizon;work.world_transitions+=len(start)*s.horizon
            return pred
        def energy(clean,start,target,weight,w):
            path=torch.cat((start[:,None],clean,target[:,None]),1)
            pred=consequences(start,decode(path));value=(pred[:,-1]-target).square().mean(-1)
            if weight:value=value+weight*(pred[:,:-1]-clean).square().mean((-1,-2))
            return value
        try:
            if check:check()
            n=self.method['num_samples'];start=z[:,0];g=goal[:,0]
            perturb=torch.Generator(device=z.device).manual_seed(torch.initial_seed()^0x5E1F1E^self.decisions)
            with torch.inference_mode(False),torch.no_grad():
                clean,_=sample_interiors(self.model.flow,start,g,horizon=s.horizon,num_samples=n,
                    flow_steps=self.method['flow_steps'],config=self.config,generator=None,
                    perturb_generator=perturb,energy=energy,work=work,check=check)
                zs=start.repeat_interleave(n,0);zg=g.repeat_interleave(n,0)
                paths=torch.cat((zs[:,None],clean.reshape(len(zs),s.horizon-1,s.dim),zg[:,None]),1)
                a=decode(paths);pred=consequences(zs,a)
                scores=(pred[:,-1]-zg).square().mean(-1);selected=int(scores.argmin());cost=float(scores[selected])
                self.last_candidate_actions=a.detach();self.last_candidate_scores=scores.detach()
                all_costs=scores.detach().cpu().tolist()
                self.previous_prediction=(pred[selected,0].detach().clone()[None],g.detach().clone())
                complete=True;return a[selected,0]
        finally:
            # Keep work from rejected deadline-overrun decisions as well.
            work.controller_ms=(time.perf_counter()-tick)*1000
            self.trace.append({**asdict(work),'local_cost':cost,'selected_candidate':selected,
                               'candidate_costs':all_costs,'plan_completed':complete,'refinement':asdict(self.config)})
            self.decisions+=1


def load(cache,checkpoint,device):
    from crossbench.spec import Spec
    from crossbench.train import code_hash
    from crossbench.models import build
    from crossbench.world import load_world
    cache=Path(cache);saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    identity=saved['identity'];s=Spec(**identity['spec']);meta=json.loads((cache/'manifest.json').read_text())
    if s.method!='leflow' or identity['fixture'] or meta['fixture']:raise ValueError('Not an actual LeFlow trained checkpoint')
    if identity['source']!=code_hash():raise ValueError('Retained baseline source identity changed')
    if identity['cache']!=sha256_file(cache/'cache.h5'):raise ValueError('Training cache identity changed')
    if identity['world']!=meta['checkpoint_sha256']:raise ValueError('World identity changed')
    model=build(s).to(device).eval().requires_grad_(False);model.load_state_dict(saved['model'],strict=True)
    world=load_world(meta,device);return model,world,s,meta,saved


def evaluate(cache,checkpoint,output,method,device='cuda:0'):
    from crossbench.environment import configure_renderer_device,NativeEnv
    configure_renderer_device()
    import h5py,hdf5plugin,numpy as np
    from crossbench.evaluate import cases,episode,summarize
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    model,world,s,meta,saved=load(cache,checkpoint,device);plan=cases(meta,s)
    before={'policy':state_fingerprint(model),'world':state_fingerprint(world)}
    identity={'benchmark_family':'retained_crossbench_leflow','checkpoint_sha256':sha256_file(checkpoint),
        'checkpoint_step':saved['step'],'training_identity':saved['identity'],'case_sha256':digest_json(plan),
        'refinement_source_sha256':digest_json({p.name:sha256_file(p)for p in sorted(Path(__file__).parent.glob('*.py'))}),
        'numerical_precision':{'dtype':'float32','NVIDIA_TF32_OVERRIDE':os.getenv('NVIDIA_TF32_OVERRIDE')},
        'method':method,'optimizer_updates':0,'retained_action_and_controller_limits':True}
    atomic_json(root/'identity.json',identity);atomic_json(root/'cases.json',plan)
    ctrl=SharedController(model,world,s,method);records=[];env=NativeEnv(s.dataset,s.action_budget)
    import wandb
    wb=wandb.init(project='leflow-self-refine',entity='attentionx2023',name=f'{s.dataset}-{method["name"]}-retained',
        dir=str(root),mode='online',config=identity,settings=wandb.Settings(init_timeout=90))
    atomic_json(root/'wandb.json',{'id':wb.id,'url':wb.url})
    started=time.perf_counter()
    try:
        with h5py.File(meta['source_path'],'r')as src:
            for idx,row in enumerate(plan):
                record,images=episode(env,world,ctrl,src,row,s,device);record['identity']=identity
                atomic_json(root/'outcomes'/f'{idx:05d}.json',record)
                dest=root/'trajectories'/f'{idx:05d}.npz';dest.parent.mkdir(exist_ok=True);np.savez_compressed(dest,**images)
                records.append(record);metrics=summarize(records)
                atomic_json(root/'progress.json',{'completed':len(records),'expected':len(plan),'metrics':metrics})
                wb.log({'cases':len(records),**{k:v for k,v in metrics.items()if isinstance(v,(int,float))}})
        after={'policy':state_fingerprint(model),'world':state_fingerprint(world)}
        if before!=after:raise RuntimeError('Frozen policy/world changed')
        report={'status':'completed','identity':identity,'metrics':summarize(records),'records':records,
                'frozen_before':before,'frozen_after':after,'evaluation_wall_seconds':time.perf_counter()-started,
                'optimizer_updates':0,'author_native_checkpoint':False,'sealed_test_cases_read':0}
        atomic_json(root/'report.json',report);wb.finish();return report
    except Exception as error:
        atomic_json(root/'failure.json',{'error':repr(error),'completed_cases':len(records),'optimizer_updates':0})
        wb.finish(exit_code=1);raise
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('cache','checkpoint','output','method-config'):p.add_argument('--'+key,required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    r=evaluate(a.cache,a.checkpoint,a.output,json.loads(Path(a.method_config).read_text()),a.device)
    print(json.dumps(r['metrics']))
