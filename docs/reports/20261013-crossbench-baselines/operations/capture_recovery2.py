from pathlib import Path
import datetime,gzip,hashlib,io,json,os,sys,tarfile
BASE=Path('/home/mtxu/adam/LeFlow-experiments/20261013-crossbench-baselines');DATA=Path('/tmp/mtxu-leflow-20261004/data');items={};receipts={}
def add(name,path,compress=False):
 if not path.exists():return
 raw=path.read_bytes()
 if path.suffix=='.jsonl' and raw and not raw.endswith(b'\n'):raw=raw.rsplit(b'\n',1)[0]+b'\n'
 data=gzip.compress(raw,mtime=0)if compress else raw;items[name]=data
 receipts[name]={'source':str(path),'original_sha256':hashlib.sha256(raw).hexdigest(),'stored_sha256':hashlib.sha256(data).hexdigest()}
for name in ['launcher.json','allocation.json','plan.json','queue-status.json','queue-complete.json','runtime-files.json','initial-preparation.json','pusht-cache-repair.json','cube-recovery-launcher.json','cube-download-progress.json','cube-recovery-launcher-file-transfer-race.json','pre-recovery-queue-status.json','pre-recovery-allocation.json','recovery-plan.json','recovery-launcher.json','recovery-queue-complete.json','precision-recovery.json','cube-numerical-diagnostic.json','cube-precision-env-verification.json','recovery2-plan.json','recovery2-launcher.json','recovery2-queue-complete.json']:
 add(name,BASE/name)
for name in ['cube-vanilla_action_flow-preflight.log','cube-cache.log','cube-cache-strict-fp32.log']:
 add('preparation-logs/'+name+'.gz',BASE/'logs'/name,True)
add('cache/cube-tf32-original/manifest.json.gz',BASE/'cache/cube-tf32-original/manifest.json',True)
summary={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':[],'final_test_outcomes_read':False}
for task in ['pusht','cube']:
 add('cache/'+task+'/manifest.json.gz',BASE/'cache'/task/'manifest.json',True)
 add('source/'+task+'/source.json.gz',DATA/'prepared'/task/'source.json',True)
for run in sorted((BASE/'runs').glob('*/*')):
 if not run.is_dir():continue
 task,method=run.parent.name,run.name;r={'dataset':task,'method':method}
 for name in ['identity.json','usage.json','native-preflight.json','gradient-preflight.json','wandb.json','best.json','complete.json','final-audit.json']:
  p=run/name;add('runs/'+task+'/'+method+'/'+name,p)
  if p.exists() and name in ['usage.json','wandb.json','best.json','complete.json']:r[name]=json.loads(p.read_text())
 add('runs/'+task+'/'+method+'/metrics.jsonl.gz',run/'metrics.jsonl',True)
 r['curve']=[]
 for p in sorted((run/'validation').glob('*/report.json')):
  add('runs/'+task+'/'+method+'/validation/'+p.parent.name+'/report.json.gz',p,True)
  x=json.loads(p.read_text());r['curve'].append({'checkpoint':p.parent.name,**x['metrics']})
 # Selected measured examples only; all case outcomes and all curves are archived.
 if (run/'complete.json').exists():
  best=json.loads((run/'best.json').read_text());folder=run/'validation'/f"step_{best['step']:07d}"/'trajectories'
  for p in sorted(folder.glob('*.npz'))[:10]:add('runs/'+task+'/'+method+'/selected-trajectories/'+p.name,p)
 summary['runs'].append(r)
for p in (BASE/'failures').rglob('*'):
 if p.is_file():add(str(p.relative_to(BASE))+('.gz' if p.suffix=='.log' else ''),p,p.suffix=='.log')
for p in (BASE/'analysis').glob('*'):
 if p.is_file():add('analysis/'+p.name,p)
items['snapshot.json']=(json.dumps(summary,indent=2)+'\n').encode();items['capture-receipt.json']=(json.dumps({'utc':summary['utc'],'artifacts':receipts,'final_test_outcomes_read':False},indent=2)+'\n').encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w:gz')as t:
 for name,data in items.items():p=tarfile.TarInfo(name);p.size=len(data);t.addfile(p,io.BytesIO(data))
