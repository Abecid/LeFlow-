from pathlib import Path
import datetime,gzip,hashlib,io,json,sys,tarfile
B=Path('/home/mtxu/adam/LeFlow-experiments/20261014-self-refine');items={};receipt={}
def add(name,path,zip=False):
 if not path.exists():return
 raw=path.read_bytes();data=gzip.compress(raw,mtime=0)if zip else raw;items[name]=data
 receipt[name]={'source':str(path),'original_sha256':hashlib.sha256(raw).hexdigest(),'stored_sha256':hashlib.sha256(data).hexdigest()}
for n in ['launcher.json','registration.json','status.json','complete.json','runtime-files.json','allocation.json','batching-pusht.json','batching-cube.json','batching-validation-history.json']:
 add(n,B/n)
for p in (B/'preflight').glob('*.json'):add('preflight/'+p.name,p)
for p in (B/'logs').glob('*.log'):add('logs/'+p.name+'.gz',p,True)
for p in (B/'jobs').glob('*'):
 if not p.is_dir():continue
 for n in ['identity.json','cases.json','progress.json','report.json','wandb.json','failure.json']:
  add('jobs/'+p.name+'/'+n+('.gz'if n in ('cases.json','report.json')else''),p/n,n in ('cases.json','report.json'))
 if (p/'report.json').exists():
  for q in sorted((p/'trajectories').glob('*.npz'))[:10]:add('jobs/'+p.name+'/selected-trajectories/'+q.name,q)
 else:
  for q in (p/'outcomes').glob('*.json'):add('jobs/'+p.name+'/partial-outcomes/'+q.name+'.gz',q,True)
for n in ['run_shared_campaign.py','preflight_shared.py','batching_probe.py']:add('operations/'+n,B/n)
items['capture-receipt.json']=(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':receipt,'optimizer_updates':0},indent=2)+'\n').encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w:gz')as t:
 for name,data in items.items():q=tarfile.TarInfo(name);q.size=len(data);t.addfile(q,io.BytesIO(data))
