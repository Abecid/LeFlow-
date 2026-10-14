from pathlib import Path
import concurrent.futures,hashlib,json,os,subprocess,threading,time,datetime,sys
B=Path('/home/mtxu/adam/LeFlow-experiments/20261014-self-refine');R=B/'runtime';P='/tmp/mtxu-crossbench-20261013/env/bin/python';OLD=Path('/home/mtxu/adam/LeFlow-experiments/20261013-crossbench-baselines')
sys.path.insert(0,str(R))
from flow_jepa.campaign import acquire_gpus
from self_refine.protocol import atomic_json,sha256_file
from self_refine.campaign import run_bounded
cfg=json.loads((R/'config/self_refine/shared_comparison_20261014.json').read_text());manifest=json.loads((B/'runtime-files.json').read_text())
assert cfg['training_updates']==0 and cfg['tasks']==['pusht','cube']
for t in cfg['tasks']:
 assert json.loads((B/'preflight'/f'{t}.json').read_text())['retained_baseline_action_and_RNG_bitwise']
 assert len(json.loads((B/f'batching-{t}.json').read_text()))==2
pending=[(t,m['name'])for m in cfg['methods']for t in cfg['tasks']];results=[];lanes={};failed=False;lock=threading.Lock()
(B/'methods').mkdir(exist_ok=True)
for m in cfg['methods']:atomic_json(B/'methods'/f'{m["name"]}.json',m)
atomic_json(B/'registration.json',{'config':cfg,'jobs':pending,'baseline_workspace':str(OLD),'optimizer_updates':0,'author_checkpoint_comparison_pending':['reacher','tworoom'],'source':manifest['source']})
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write():atomic_json(B/'status.json',{'utc':stamp(),'source':manifest['source'],'optimizer_updates':0,'lanes':lanes,'pending':pending,'results':results,'failed':failed})
def verify():
 assert all(sha256_file(R/k)==h for k,h in manifest['files'].items())
def lane(gpu):
 global failed
 while True:
  with lock:
   if failed:return
   if not pending:lanes[gpu]={'state':'completed'};write();return
   task,name=pending.pop(0);out=B/'jobs'/f'{task}-{name}'
   if out.exists():raise RuntimeError('No completed/incomplete job replay allowed')
   lanes[gpu]={'state':'running','task':task,'method':name};write()
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',LIBGL_ALWAYS_SOFTWARE='1',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',LD_LIBRARY_PATH='/tmp/mtxu-flow-jepa-20261007/env/lib',__EGL_VENDOR_LIBRARY_FILENAMES='/tmp/mtxu-flow-jepa-20261007/env/share/glvnd/egl_vendor.d/50_mesa.json',SDL_VIDEODRIVER='dummy',NVIDIA_TF32_OVERRIDE='0')
  if task=='cube':env['NVIDIA_TF32_OVERRIDE']='0'
  try:
   verify();start=time.time()
   cmd=[P,'-u','-m','self_refine.shared_eval','--cache',str(OLD/'cache'/task),'--checkpoint',str(OLD/'runs'/task/'leflow/best.pt'),'--output',str(out),'--method-config',str(B/'methods'/f'{name}.json')]
   with(B/'logs'/f'{task}-{name}.log').open('w')as log:rc=run_bounded(cmd,cwd=R,env=env,log=log,timeout=cfg['job_timeout_seconds'])
   if rc:raise RuntimeError(f'{task}-{name} exited {rc}')
   report=json.loads((out/'report.json').read_text());assert report['status']=='completed' and report['optimizer_updates']==0 and len(report['records'])==50
   with lock:
    results.append({'task':task,'method':name,'gpu':gpu,'returncode':rc,'completed':stamp(),'successes':report['metrics']['successes'],'evaluation_wall_seconds':report['evaluation_wall_seconds'],'job_wall_seconds':time.time()-start});write()
  except BaseException as error:
   with lock:failed=True;lanes[gpu]['state']='failed_hold';lanes[gpu]['error']=repr(error);write()
   return
verify();devices,held=acquire_gpus(8,B,wait_hours=1,required=8)
atomic_json(B/'allocation.json',{'devices':devices,'one_job_per_gpu':True,'optimizer_updates':0})
with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:
 futures=[pool.submit(lane,i)for i,u in devices]
 for f in futures:f.result()
for f in held:f.close()
atomic_json(B/'complete.json',{'utc':stamp(),'all_twelve_completed':len(results)==12 and not failed,'results':results,'optimizer_updates':0})
