"""Audited noise-search evaluation on retained shared-execution LeFlow weights.

All methods use the deployed crossbench adapter, fixed cases/limits and frozen
weights. This is not evaluation of unavailable original author planner weights.
"""
from __future__ import annotations
import argparse,json,os,time
from dataclasses import asdict
from pathlib import Path
import torch
from .crossbench import RefinementController,source_hash
from .noise_search import noise_variants
from self_refine.shared_eval import load
from self_refine.native import state_fingerprint
from self_refine.protocol import atomic_json,digest_json,sha256_file


def evaluate(cache,checkpoint,output,variant,device='cuda:0'):
    from crossbench.environment import configure_renderer_device,NativeEnv
    configure_renderer_device()
    import h5py,hdf5plugin,numpy as np
    from crossbench.evaluate import cases,episode,summarize
    if torch.device(device).type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    cfg=noise_variants()[variant]
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    model,world,s,meta,saved=load(cache,checkpoint,device);plan=cases(meta,s)
    before={'policy':state_fingerprint(model),'world':state_fingerprint(world)}
    identity={'benchmark_family':'retained_crossbench_leflow_noise_search',
        'checkpoint_sha256':sha256_file(checkpoint),'checkpoint_step':saved['step'],
        'training_identity':saved['identity'],'case_sha256':digest_json(plan),
        'inference_source_sha256':source_hash(),'noise_search':asdict(cfg),'variant':variant,
        'numerical_precision':{'dtype':'float32','NVIDIA_TF32_OVERRIDE':os.getenv('NVIDIA_TF32_OVERRIDE')},
        'optimizer_updates':0,'retained_action_and_controller_limits':True}
    atomic_json(root/'identity.json',identity);atomic_json(root/'cases.json',plan)
    ctrl=RefinementController(model,world,s,cfg);records=[];env=NativeEnv(s.dataset,s.action_budget)
    import wandb
    wb=wandb.init(project='leflow-noise-search',entity='attentionx2023',name=f'{s.dataset}-{variant}-retained',
        dir=str(root),mode='online',config=identity,settings=wandb.Settings(init_timeout=90))
    atomic_json(root/'wandb.json',{'id':wb.id,'url':wb.url})
    started=time.perf_counter()
    try:
        with h5py.File(meta['source_path'],'r')as src:
            for idx,row in enumerate(plan):
                record,images=episode(env,world,ctrl,src,row,s,device);record['identity']=identity
                atomic_json(root/'outcomes'/f'{idx:05d}.json',record)
                dest=root/'trajectories'/f'{idx:05d}.npz';dest.parent.mkdir(exist_ok=True)
                np.savez_compressed(dest,**images);records.append(record);metrics=summarize(records)
                atomic_json(root/'progress.json',{'completed':len(records),'expected':len(plan),'metrics':metrics})
                wb.log({'cases':len(records),**{k:v for k,v in metrics.items()if isinstance(v,(int,float))}})
        after={'policy':state_fingerprint(model),'world':state_fingerprint(world)}
        if before!=after:raise RuntimeError('Frozen policy/world changed')
        if len(records)!=s.eval_episodes or len({r['id']for r in records})!=s.eval_episodes:
            raise RuntimeError('Incomplete or duplicate physical evaluation')
        report={'status':'completed','identity':identity,'metrics':summarize(records),'records':records,
                'frozen_before':before,'frozen_after':after,'evaluation_wall_seconds':time.perf_counter()-started,
                'optimizer_updates':0,'author_native_checkpoint':False,'sealed_test_cases_read':0}
        atomic_json(root/'report.json',report);wb.finish();return report
    except Exception as error:
        atomic_json(root/'failure.json',{'error':repr(error),'completed_cases':len(records),'optimizer_updates':0})
        wb.finish(exit_code=1);raise
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('cache','checkpoint','output','variant'):p.add_argument('--'+key,required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    print(json.dumps(evaluate(a.cache,a.checkpoint,a.output,a.variant,a.device)['metrics']))
