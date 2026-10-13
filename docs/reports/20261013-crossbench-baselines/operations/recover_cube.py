"""Four resumable byte-range downloads, checked against the official LFS SHA256."""
from pathlib import Path
import concurrent.futures,hashlib,json,os,re,subprocess,sys,threading,time
from huggingface_hub import hf_hub_url,get_hf_file_metadata
import requests
BASE=Path('/home/mtxu/adam/LeFlow-experiments/20261013-crossbench-baselines');DATA=Path('/tmp/mtxu-leflow-20261004/data')
REPO='quentinll/lewm-cube';REV='02a19a67a0dc8c9d6215f89c19e0a597691e152a';NAME='cube_single_expert.tar.zst';TOTAL=46184624478
url=hf_hub_url(REPO,NAME,repo_type='dataset',revision=REV,endpoint='https://hf-mirror.com')
meta=get_hf_file_metadata(url);expected=meta.etag.strip('"');assert len(expected)==64 and meta.size==TOTAL
root=DATA/'downloads/cube';root.mkdir(parents=True,exist_ok=True);temp=root/(NAME+'.ranges.partial');status=BASE/'cube-download-progress.json';chunk=512*1024*1024
lock=threading.Lock();done=set();progress={};copied=0
if status.exists() and temp.exists():
 old=json.loads(status.read_text());assert old['sha256']==expected;done=set(old['completed_ranges']);copied=old['prefix_bytes']
else:
 old=next((p for p in root.rglob('*.incomplete')if expected in p.name),None)
 if old:
  copied=old.stat().st_size
  with old.open('rb')as src,temp.open('wb')as out:
   remaining=copied
   while remaining:
    buf=src.read(min(8*1024*1024,remaining));out.write(buf);remaining-=len(buf)
 with temp.open('ab')as f:f.truncate(TOTAL)
 done=set(range(copied//chunk))
fd=os.open(temp,os.O_RDWR)
def save():
 p=status.with_suffix('.partial');p.write_text(json.dumps({'revision':REV,'filename':NAME,'sha256':expected,'bytes':TOTAL,'prefix_bytes':copied,'completed_ranges':sorted(done),'active_range_bytes':progress,'completed':False},indent=2)+'\n');p.replace(status)
with lock:save()
def fetch(i):
 start=i*chunk;end=min(TOTAL,(i+1)*chunk);pos=max(start,min(end,copied));attempt=0
 while pos<end:
  try:
   location=get_hf_file_metadata(url).location
   with requests.get(location,headers={'Range':f'bytes={pos}-{end-1}'},stream=True,timeout=(30,120))as response:
    response.raise_for_status();assert response.status_code==206
    assert response.headers.get('Content-Range')==f'bytes {pos}-{end-1}/{TOTAL}',response.headers.get('Content-Range')
    for data in response.iter_content(1024*1024):
     if not data:continue
     assert pos+len(data)<=end
     written=0
     while written<len(data):written+=os.pwrite(fd,data[written:],pos+written)
     pos+=len(data)
   assert pos==end
  except Exception as error:
   attempt+=1
   with lock:progress[str(i)]=pos-start;save()
   print(json.dumps({'range':i,'retry':attempt,'bytes':pos-start,'error_type':type(error).__name__}),flush=True)
   if attempt>=20:raise
   time.sleep(min(30,attempt*2))
 with lock:done.add(i);progress.pop(str(i),None);save()
 print(json.dumps({'range_complete':i,'ranges_complete':len(done)}),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:list(pool.map(fetch,[i for i in range((TOTAL+chunk-1)//chunk)if i not in done]))
os.close(fd);h=hashlib.sha256()
with temp.open('rb')as f:
 for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
assert h.hexdigest()==expected,'Archive checksum mismatch; never extract unverified bytes'
archive=root/NAME;temp.replace(archive)
record=json.loads(status.read_text());record.update(completed=True,archive_sha256_verified=True);status.write_text(json.dumps(record,indent=2)+'\n')
sys.path.insert(0,str(BASE/'runtime'))
from scripts.prepare_data import extract
from btm_jepa.data import TASKS
spec=TASKS['cube'];target=DATA/'datasets'/(spec['name']+'.h5');extract(archive,target,spec['name'])
subprocess.run([sys.executable,'-u','scripts/prepare_data.py','--task','cube'],cwd=BASE/'runtime',check=True)
print('OFFICIAL_CUBE_SOURCE_READY',flush=True)
