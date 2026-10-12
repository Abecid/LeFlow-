"""No-update TRAIN-only check of factual posterior prediction support.

The old retrieval model supplies frozen features for this diagnostic ONLY.
No parameter/checkpoint from this probe enters the new fresh training run.
Past recorded transitions predict the next recorded action's observed error.
This cannot validate the physical outcomes of rejected/generated actions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from flow_jepa.common import config, file_hash, save_json, seed_all
from flow_jepa.execution.bayesian_repair import BayesianRepairPolicy, predictive_moments
from flow_jepa.execution.progress_ttt import patch_cost
from flow_jepa.execution.retrieval_repair import RetrievalRepairSegments
from flow_jepa.execution.train import load_world


@torch.no_grad()
def main(args):
    began=time.perf_counter();torch.set_num_threads(2);seed_all(3072)
    device=torch.device('cuda:0');c=config(args.config)
    checkpoint=torch.load(args.reference,map_location='cpu',weights_only=False)
    assert checkpoint['method']=='retrieval_repair_ttt' and checkpoint['seed']==3072
    model=BayesianRepairPolicy(c).to(device).eval()
    missing,unexpected=model.load_state_dict(checkpoint['model'],strict=False)
    assert set(missing)=={'prior_tril','log_noise_variance'} and not unexpected
    world,_=load_world(c,args.world,device)
    data=RetrievalRepairSegments(Path(args.root),c,'bayesian_repair_ttt',3072,1024)
    rows=[]
    for offset in range(0,1024,32):
        examples=[data[i] for i in range(offset,offset+32)]
        batch={k:torch.stack([r[k] for r in examples]).to(device) for k in examples[0]}
        history=model.factual_history(batch,world);q=batch['z'][:,2]
        mean,factor=model.fit_posterior(history,q)
        predicted=world.rollout(batch['z'][:,0],batch['a'][:,:1])[:,0]
        features=model.transition_keys(batch['z'][:,0],batch['a'][:,0],predicted,q).mean(-2)
        label=(patch_cost(batch['z'][:,1],q)-patch_cost(predicted,q)).mean(-1)/model.error_scale
        posterior,epistemic,variance=predictive_moments(features[:,None],mean,factor,model.noise_variance())
        prior=(features*model.fast_prior).sum(-1)
        for j in range(len(q)):
            episode,start,_=data.sample_spec(offset+j)
            assert episode['split']=='train'
            assert int(batch['history_mask'][j].sum())==min(start,model.history_steps)
            if start:assert torch.equal(batch['history_next'][j,-1],batch['z'][j,0])
            split=int(hashlib.sha256(episode['id'].encode()).hexdigest(),16)%2
            rows.append(dict(episode=episode['id'],start=start,split=split,
                history=int(batch['history_mask'][j].sum()),label=float(label[j]),
                prior=float(prior[j]),posterior=float(posterior[j,0]),
                variance_multiplier=float(variance[j,0]/model.noise_variance()),
                epistemic=float(epistemic[j,0])))
    # Estimate a SINGLE residual noise scale on disjoint TRAIN episodes; report
    # next-outcome calibration on the other TRAIN episodes. No optimizer steps.
    train=[r for r in rows if r['split']==0 and r['history']]
    check=[r for r in rows if r['split']==1 and r['history']]
    assert {r['episode'] for r in train}.isdisjoint({r['episode'] for r in check})
    sigma2=float(np.mean([(r['label']-r['posterior'])**2/r['variance_multiplier'] for r in train]))
    def summary(rs):
        prior=np.array([r['label']-r['prior'] for r in rs]);post=np.array([r['label']-r['posterior'] for r in rs])
        var=sigma2*np.array([r['variance_multiplier'] for r in rs])
        return dict(n=len(rs),prior_mae=float(np.abs(prior).mean()*model.error_scale),
            posterior_mae=float(np.abs(post).mean()*model.error_scale),
            prior_mse=float(np.square(prior).mean()*model.error_scale**2),
            posterior_mse=float(np.square(post).mean()*model.error_scale**2),
            central_90_coverage=float(np.mean(post**2<=1.644854**2*var)),
            standardized_error=float(np.mean(post**2/var)),
            variance_multiplier_range=[min(r['variance_multiplier'] for r in rs),max(r['variance_multiplier'] for r in rs)])
    result=summary(check)
    supported=result['posterior_mae']<result['prior_mae'] and result['posterior_mse']<result['prior_mse']
    report=dict(training_only=True,optimizer_updates=0,simulator_episodes=0,
        source_reference_sha256=file_hash(args.reference),reference_step=checkpoint['step'],
        world_sha256=file_hash(args.world),manifest_sha256=file_hash(Path(args.root)/'manifest.json'),
        observation_unit='one physical transition; chronological past-only history',
        examples=len(rows),fit_examples=len(train),disjoint_train_episode_check=result,
        fitted_noise_std=float(sigma2**.5*model.error_scale),supports_factual_adaptation=supported,
        limitations=['Frozen old features, not the new trained model.',
            'Residual variance estimated on TRAIN only; not injected into new training.',
            'Does not validate candidate comparison, task success or a formal confidence bound.',
            'Within-episode correlated observations; repeated contexts are not independent trials.'],
        seconds=time.perf_counter()-began,gpu_count=1,gpu_hours=(time.perf_counter()-began)/3600,
        factual_records=rows)
    save_json(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k!='factual_records'}),flush=True)
    if not supported:raise RuntimeError('Frozen-feature factual adaptation diagnostic does not support launch')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('config','root','world','reference','output'):p.add_argument('--'+key,required=True)
    main(p.parse_args())
