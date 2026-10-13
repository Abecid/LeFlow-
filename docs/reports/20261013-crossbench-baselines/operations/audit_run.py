from pathlib import Path
import argparse,hashlib,json,sys,torch
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--method',required=True);a=p.parse_args()
base=Path('/home/mtxu/adam/LeFlow-experiments/20261013-crossbench-baselines');repo=base/'runtime';sys.path.insert(0,str(repo))
from crossbench.train import code_hash
from crossbench.spec import Spec
from crossbench.evaluate import cases,summarize
from flow_jepa.common import digest,file_hash
r=base/'runs'/a.dataset/a.method;c=json.loads((r/'complete.json').read_text());identity=c['identity'];s=Spec(**identity['spec']);assert s.dataset==a.dataset and s.method==a.method and not identity['fixture'];assert identity['ranks']==4 and identity['source']==code_hash()
source=json.loads((base/'runtime-files.json').read_text());assert all(file_hash(repo/k)==h for k,h in source['files'].items())
meta=json.loads((base/'cache'/a.dataset/'manifest.json').read_text());assert meta['cache_sha256']==identity['cache'];assert meta['checkpoint_sha256']==identity['world'] and meta['test_episodes_read']==0 and not meta['raw_actions_clipped'];assert file_hash(meta['lewm_checkpoint'])==identity['world']
plan=cases(meta,s);expected={x['id']:x for x in plan};curve=[];points=[];checkpoint_info={}
for step in c['validation_steps']:
 path=r/'validation'/f'step_{step:07d}'/'report.json';x=json.loads(path.read_text());assert len(x['records'])==50;assert x['identity']['cases']==digest(plan)
 for row in x['records']:
  assert all(row[k]==expected[row['id']][k]for k in ['episode','source_offset','start','goal_offset','seed']);assert row['steps']<=50 and len(row['actions'])==row['steps']
 m=summarize(x['records']);assert m==x['metrics'];curve.append({'step':step,**m});points.append((m['successes'], -step,step))
 q=r/'checkpoints'/f'step_{step:07d}.pt';d=torch.load(q,map_location='cpu',weights_only=False);assert d['identity']==identity and d['step']==step;assert all(torch.isfinite(t).all()for t in d['model'].values());checkpoint_info[str(step)]={'path':str(q),'sha256':file_hash(q)}
b=json.loads((r/'best.json').read_text());best=max(points)[2];assert b['step']==best and file_hash(r/'best.pt')==b['checkpoint_sha256'];chosen=torch.load(r/'best.pt',map_location='cpu',weights_only=False);matched=torch.load(r/'checkpoints'/f'step_{best:07d}.pt',map_location='cpu',weights_only=False);assert all(torch.equal(t,matched['model'][k])for k,t in chosen['model'].items())
assert c['step']<=20000 and c['optimization_gpu_hours']*3600<=28800+json.loads((r/'usage.json').read_text())['final_update_overrun_seconds']+1e-6
import wandb
w=json.loads((r/'wandb.json').read_text());online=wandb.Api(timeout=45).run('attentionx2023/leflow-crossbench/'+w['id']);assert online.state=='finished'
out={'dataset':a.dataset,'method':a.method,'source_commit':source['commit'],'runtime_files_verified':len(source['files']),'identity':identity,'curves':curve,'selected_step':best,'selected_successes':max(points)[0],'checkpoints':checkpoint_info,'best_checkpoint_sha256':b['checkpoint_sha256'],'wandb_finished':True,'wandb_url':w['url'],'audit_passed':True,'optimizer_updates':0,'model_forward_calls':0,'simulator_calls':0,'final_test_outcomes_read':False}
(r/'final-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'dataset':a.dataset,'method':a.method,'selected':max(points)[0],'audit_passed':True}))
