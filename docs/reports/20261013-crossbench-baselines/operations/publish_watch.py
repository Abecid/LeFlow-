"""CPU-only publication watcher. Never launches or changes GPU experiments."""
from pathlib import Path
import datetime,fcntl,json,os,subprocess,sys,tarfile,time

WORK=Path(__file__).resolve().parent
REPO=WORK/'publication-checkout'
BASE='/home/mtxu/adam/LeFlow-experiments/20261013-crossbench-baselines'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','target_server_2_cf']
PYTHON='/tmp/mtxu-crossbench-20261013/env/bin/python'
SOURCE='747f635ad36af2cbe3e2bbb0b7997efe9caccaa1'
owned=(WORK/'publication-watch.lock').open('a+')
fcntl.flock(owned,fcntl.LOCK_EX|fcntl.LOCK_NB)

def call(args,**kw):return subprocess.run(args,check=True,timeout=240,**kw)
def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,text=True,timeout=120).strip()
def state(**values):
 p=WORK/'publication-watch-status.json'
 values.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),source=SOURCE)
 temp=p.with_suffix('.partial');temp.write_text(json.dumps(values,indent=2)+'\n');temp.replace(p)

if not (REPO/'.git').exists():
 call(['git','clone','--filter=blob:none','--no-checkout','https://github.com/Abecid/LeFlow-.git',str(REPO)])
 call(['git','fetch','origin','main'],cwd=REPO)
 call(['git','checkout','--detach','origin/main'],cwd=REPO)
 call(['git','config','user.name','Adam Lee'],cwd=REPO)
 call(['git','config','user.email','cmlee3211@gmail.com'],cwd=REPO)
published=set();pending_commit=None
while True:
 try:
  response=call(SSH+['python3 -'],input=("from pathlib import Path;import json;"
   f"b=Path('{BASE}');q=json.loads((b/'queue-status.json').read_text());"
   "print(json.dumps({'source':q['source'],'failed':q['failed'],"
   "'results':[(r['dataset'],r['method'])for r in q['results']],"
   "'queue_complete':json.loads((b/'recovery-queue-complete.json').read_text())if(b/'recovery-queue-complete.json').exists()else None}))").encode(),capture_output=True)
  q=json.loads(response.stdout);assert q['source']==SOURCE
  done={tuple(r)for r in q['results']}
  if done!=published or q['failed']:
   for task,method in sorted(done):
    assert task in ('pusht','cube') and method in ('leflow','vanilla_action_flow','execution_revision','action_flow_search','sage_port','diffusion_policy_port','feedback_wm_port','retrieval_repair_ttt')
    script=("from pathlib import Path;import subprocess;"
     f"p=Path('{BASE}/runs/{task}/{method}/final-audit.json');"
     f"subprocess.run(['{PYTHON}','{BASE}/audit_run.py','--dataset','{task}','--method','{method}'],check=True)if not p.exists()else None")
    call(SSH+['python3 -'],input=script.encode())
   # Private reporting checkout only. Preserve any unpushed report commit.
   if git('status','--porcelain'):
    dirty=set(git('diff','--name-only').splitlines())|set(git('diff','--cached','--name-only').splitlines())|set(git('ls-files','--others','--exclude-standard').splitlines())
    prefixes=('docs/CROSSBENCH_RESULTS.md','docs/crossbench-results-ledger.json','docs/crossbench-results/','docs/reports/20261013-crossbench-baselines/')
    if not all(any(path==prefix or path.startswith(prefix)for prefix in prefixes)for path in dirty):
     raise RuntimeError('Unrelated private-checkout changes; refuse overwrite')
    call(['git','add','docs/CROSSBENCH_RESULTS.md','docs/crossbench-results-ledger.json','docs/crossbench-results','docs/reports/20261013-crossbench-baselines'],cwd=REPO)
    call(['git','commit','-m','Preserve interrupted native publication before retry'],cwd=REPO)
    pending_commit=git('rev-parse','HEAD')
   if pending_commit:call(['git','branch','publication-retry-'+str(time.time_ns()),pending_commit],cwd=REPO)
   call(['git','fetch','origin','main'],cwd=REPO)
   call(['git','checkout','--detach','origin/main'],cwd=REPO)
   capture=WORK/'watch-capture.tar.gz'
   with capture.open('wb')as f:call(SSH+['python3 -'],input=(WORK/'capture.py').read_bytes(),stdout=f)
   archive=REPO/'docs/reports/20261013-crossbench-baselines';archive.mkdir(parents=True,exist_ok=True)
   with tarfile.open(capture,'r:gz')as t:t.extractall(archive,filter='data')
   call([sys.executable,str(WORK/'publish_snapshot.py'),'--repo',str(REPO)])
   call(['git','add','docs/CROSSBENCH_RESULTS.md','docs/crossbench-results-ledger.json','docs/crossbench-results','docs/reports/20261013-crossbench-baselines'],cwd=REPO)
   if git('diff','--cached','--name-only'):
    call(['git','commit','-m',f'Preserve {len(done)} completed native cross-benchmark runs and full evaluations'],cwd=REPO)
   pending_commit=git('rev-parse','HEAD')
   call(['git','-c','http.postBuffer=524288000','push','origin','HEAD:main'],cwd=REPO)
   verify=call([sys.executable,str(WORK/'verify_publication.py'),'--repo',str(REPO)],capture_output=True)
   receipt=json.loads(verify.stdout);assert receipt['completed_native_runs']==len(done)
   published=done;pending_commit=None
   state(state='published',completed=len(done),verification=receipt,failed_queue=q['failed'])
  if q['failed']:
   state(state='queue_failure_hold',completed=len(done),message='GPU queue failure preserved; no recovery or additional compute launched');break
  if q['queue_complete'] and q['queue_complete']['all_sixteen_succeeded']:
   assert len(published)==16
   state(state='complete',completed=16,verification=receipt);break
  state(state='monitoring',completed=len(published),last_verified_remote=git('rev-parse','HEAD'))
 except Exception as e:
  state(state='publication_retry',completed=len(published),error=repr(e),pending_commit=pending_commit)
 time.sleep(60)
