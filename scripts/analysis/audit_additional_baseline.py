"""CPU-only identity/budget/checkpoint audit of an additional baseline.

Reads saved records and tensor files. No model forward, optimization, simulator,
or reserved-test outcome access. Run from the reporting checkout after completion.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import torch

if '--record' in sys.argv:
    frozen_repo=Path(sys.argv[sys.argv.index('--record')+1])/'repo'
    sys.path.insert(0,str(frozen_repo))
from flow_jepa.common import digest, file_hash, save_json


def audit(a):
    torch.set_num_threads(2)
    record=Path(a.record);repo=record/'repo';root=record/'campaign'
    method=a.method;run=root/'runs'/f'{method}_3072'
    c=json.loads((repo/'config'/f'flow_metaworld_{method}.json').read_text())
    identity=json.loads((run/'run.json').read_text())
    complete=json.loads((run/'complete.json').read_text())
    usage=json.loads((run/'compute_usage.json').read_text())
    runtime=json.loads((record/'runtime-files.json').read_text())
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==identity['code']==runtime['commit']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    for path,sha in runtime['files'].items():assert file_hash(repo/path)==sha,path
    manifest=json.loads((root/'manifest.json').read_text())
    reuse=json.loads((root/'reuse.json').read_text())
    previous=Path(reuse['source_root'])
    original=json.loads((previous/'manifest.json').read_text())
    assert digest(c)==manifest['protocol']==identity['protocol']
    assert file_hash(root/'manifest.json')==identity['manifest']
    assert digest(manifest['entries'])==reuse['entries_sha256']
    for key in reuse['unchanged_fields']:assert manifest[key]==original[key],key
    assert file_hash(previous/'manifest.json')==reuse['world_source_manifest']
    assert file_hash(previous/'runs/world_3072/best.pt')==identity['world_hash']==c['controller_grounded']['world_sha256']
    assert c['seeds']==[3072] and c['training']['global_batch']==64 and not c['execution']['test_enabled']
    assert complete['step']<=20000 and complete['validation_steps']==usage['validation_steps']
    assert complete['validation_steps'][-1]==complete['step']
    assert usage['optimization_gpu_seconds']<=28800+usage['last_update_overrun_gpu_seconds']+1e-6
    assert usage['gpu_count']==identity['world_size']==4
    assert abs(complete['optimization_gpu_hours']-usage['optimization_gpu_seconds']/3600)<1e-9
    bank_meta=json.loads((root/'route-bank.json').read_text())
    assert file_hash(a.bank)==identity['bank_hash']==bank_meta['sha256']
    bank=torch.load(a.bank,map_location='cpu',mmap=True,weights_only=False)
    old=torch.load(a.previous_bank,map_location='cpu',mmap=True,weights_only=False)
    assert len(bank['episode_ids'])==6222 and all(i.startswith('train/') for i in bank['episode_ids'])
    for key in ('episode_ids','episode_hashes'):assert bank[key]==old[key]
    for key in ('raw','chunks','routes','episode','time','mean','std'):
        x,y=bank[key],old[key];assert x.shape==y.shape and x.dtype==y.dtype,key
        for first in range(0,len(x),2048):assert torch.equal(x[first:first+2048],y[first:first+2048]),key
    curves=[];cases=None
    for step in complete['validation_steps']:
        path=run/'validation'/f'step_{step:07d}.json';report=json.loads(path.read_text())
        for key in ('code','protocol','manifest','world_hash','bank_hash','method','seed','world_size'):
            assert report[key]==identity[key],key
        rows=report['records'];keys=sorted(tuple(r[k] for k in ('id','reset_seed','model_seed','episode_sha256','task')) for r in rows)
        assert len(keys)==len(set(keys))==104 and all(r[0].startswith('validation/') for r in keys)
        assert Counter(r['task'] for r in rows)=={task:8 for task in c['training_tasks']}
        if cases is None:cases=keys
        assert keys==cases
        successes=sum(bool(r['success']) for r in rows)
        assert abs(report['metrics']['success_macro']-successes/104)<1e-9
        assert all(r['steps']<=200 for r in rows)
        if method in ('vanilla_action_flow','diffusion_policy_port'):
            assert all(r['world_predictions']==0 for r in rows)
        curves.append(dict(step=step,successes=successes,timeouts=sum(r['controller_budget_exhausted'] for r in rows),sha256=file_hash(path)))
    selected=max(curves,key=lambda x:(x['successes'],-x['step']))
    assert selected['step']==complete['best_step'] and abs(selected['successes']/104-complete['best'])<1e-9
    checkpoints={}
    for name in ['last','best']:
        path=run/(name+'.pt');saved=torch.load(path,map_location='cpu',weights_only=False)
        for key in ('code','protocol','manifest','world_hash','bank_hash','method','seed','world_size'):
            assert saved[key]==identity[key],key
        assert not saved['fixture'] and all(torch.isfinite(v).all() for v in saved['model'].values())
        if method in ('vanilla_action_flow','action_flow_search'):
            assert any(k.startswith('flow.') for k in saved['model'])
            assert not any(k.startswith(('core.','repair_flow.','fast_')) for k in saved['model'])
        elif method=='sage_port':
            assert any(k.startswith('subgoal.') for k in saved['model']) and any(k.startswith('actor.') for k in saved['model'])
        else:
            assert any(k.startswith('net.') for k in saved['model']) and 'alpha_bar' in saved['model']
        if name=='best':assert saved['step']==selected['step']
        checkpoints[name]={'path':str(path),'sha256':file_hash(path),'step':saved['step']}
    for point in curves:
        path=run/'checkpoints'/f"step_{point['step']:07d}.pt"
        saved=torch.load(path,map_location='cpu',weights_only=False)
        assert saved['step']==point['step'] and saved['code']==identity['code']
        assert all(torch.isfinite(v).all() for v in saved['model'].values())
        if point['step']==selected['step']:
            best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
            assert all(torch.equal(v,best['model'][k]) for k,v in saved['model'].items())
        checkpoints[str(point['step'])]={'path':str(path),'sha256':file_hash(path),'step':point['step']}
    ddp=json.loads((root/'ddp-check.json').read_text());preflight=json.loads((root/'preflight.json').read_text())
    assert ddp['optimizer_updates']==0 and ddp['gradient_sync_exact'] and ddp['parameters_unchanged']
    assert preflight['exact_previous_consumed_training_fields']==8 and preflight['frozen_world_unchanged']
    out={'method':method,'source':identity['code'],'runtime_files_verified':len(runtime['files']),
         'train_episodes':6222,'case_identities_match':True,'source_data_bank_checkpoint_audit_passed':True,
         'curve':curves,'selected':selected,'checkpoints':checkpoints,'complete':complete,'usage':usage,
         'model_forward_calls':0,'optimizer_updates':0,'simulator_calls':0,'final_tests_read':False}
    save_json(a.output,out);print(json.dumps({'method':method,'selected':selected,'audit_passed':True}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('record','method','bank','previous-bank','output'):p.add_argument('--'+name,required=True)
    audit(p.parse_args())
