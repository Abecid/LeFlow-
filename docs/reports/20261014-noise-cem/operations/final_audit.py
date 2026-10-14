from pathlib import Path
import json,hashlib,os,datetime,subprocess,sys,importlib.metadata
B=Path('/home/mtxu/adam/LeFlow-experiments/20261014-noise-cem');R=B/'runtime';sys.path.insert(0,str(R))
from self_refine.protocol import atomic_json,sha256_file
from crossbench.evaluate import summarize
from leflow_refinement.crossbench import source_hash
q=json.loads((B/'complete.json').read_text());assert q['all_twelve_completed'] and len(q['results'])==12
manifest=json.loads((B/'runtime-files.json').read_text());assert all(sha256_file(R/k)==v for k,v in manifest['files'].items())
items=[];inventory={}
for job in q['results']:
 p=B/'jobs'/f"{job['task']}-{job['method']}";x=json.loads((p/'report.json').read_text());assert x['status']=='completed' and x['optimizer_updates']==0
 assert x['frozen_before']==x['frozen_after'];assert x['identity']['inference_source_sha256']==source_hash()
 assert len(x['records'])==50 and len({r['id']for r in x['records']})==50
 assert summarize(x['records'])==x['metrics']
 assert all(r['identity']==x['identity'] for r in x['records'])
 assert len(list((p/'outcomes').glob('*.json')))==50 and len(list((p/'trajectories').glob('*.npz')))==50
 inventory[p.name]={f.name:sha256_file(f)for f in sorted((p/'trajectories').glob('*.npz'))}
 items.append({'task':job['task'],'method':job['method'],'report_sha256':sha256_file(p/'report.json'),'cases':50,'frozen_state_equal':True,'successes':x['metrics']['successes'],'timeouts':x['metrics']['timeouts'],'wall_seconds':x['evaluation_wall_seconds'],'wandb':json.loads((p/'wandb.json').read_text())})
# Inspect process command lines only for this owned runtime/queue; do not
# interact with any other user's processes or publish arbitrary process args.
owned=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  cmd=(p/'cmdline').read_bytes().decode(errors='replace').split('\0');cwd=(p/'cwd').resolve()
  if cwd==R or any(str(B/'run_noise_campaign.py')==a for a in cmd):owned.append(int(p.name))
 except (OSError,PermissionError):pass
assert not owned,owned
import wandb
api=wandb.Api(timeout=90);states={r['wandb']['id']:api.run('attentionx2023/leflow-noise-search/'+r['wandb']['id']).state for r in items}
assert all(v=='finished'for v in states.values())
versions={n:importlib.metadata.version(n) for n in ['torch','torchvision','numpy','mujoco','dm-control','ogbench','stable-worldmodel','wandb']}
atomic_json(B/'trajectory-inventory.json',inventory)
atomic_json(B/'final-audit.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'completed','runtime_source':manifest['source'],'runtime_files_verified':len(manifest['files']),'completed_jobs':items,'physical_cases':600,'optimizer_updates':0,'owned_processes':owned,'wandb_states':states,'versions':versions,'all_600_trajectories_stored_and_hashed':True,'visual_review_claimed':False})
print(json.dumps({'complete':True,'jobs':len(items),'cases':600,'optimizer_updates':0,'owned_processes':owned,'all_wandb_finished':True}))
