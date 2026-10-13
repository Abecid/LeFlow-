"""Two four-GPU lanes; exactly the registered sixteen fresh native comparisons."""
import concurrent.futures,datetime,hashlib,json,os,subprocess,sys,threading,time
from pathlib import Path
BASE=Path('/home/mtxu/adam/LeFlow-experiments/20261013-crossbench-baselines')
REPO=BASE/'runtime';PYTHON='/tmp/mtxu-crossbench-20261013/env/bin/python'
DATA=Path('/tmp/mtxu-leflow-20261004/data');sys.path.insert(0,str(REPO))
from flow_jepa.campaign import acquire_gpus
from flow_jepa.common import save_json
from crossbench.spec import matrix
source=json.loads((BASE/'runtime-files.json').read_text());lock=threading.Lock()
order=['vanilla_action_flow','retrieval_repair_ttt','leflow','execution_revision','action_flow_search','sage_port','diffusion_policy_port','feedback_wm_port']
old=json.loads((BASE/'queue-complete.json').read_text())
assert old['source']==source['commit'] and not old['all_sixteen_succeeded']
results=old['results'];done={(r['dataset'],r['method'])for r in results}
assert len(done)==5 and all(r['complete']['status']=='completed' for r in results)
pending=[(t,m)for m in order for t in ('pusht','cube')if(t,m)not in done];states={};cube_preparing=False;failed=False
began=time.perf_counter();save_json(BASE/'recovery-plan.json',{'matrix':matrix(),'retained_completed':sorted(done),'remaining':pending,'no_consumed_optimization_replayed':True})
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def verify():
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==source['commit']
 assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=REPO,text=True).strip()
 assert all(hashlib.sha256((REPO/p).read_bytes()).hexdigest()==h for p,h in source['files'].items())
def record():save_json(BASE/'queue-status.json',dict(source=source['commit'],utc=stamp(),lanes=states,pending=pending,results=results,failed=failed))
def env_for(devices):
 e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES=','.join(devices),CUDA_MODULE_LOADING='LAZY',
  STABLEWM_HOME=str(DATA),HF_HOME='/tmp/mtxu-leflow-20261004/hf-cache',HF_ENDPOINT='https://hf-mirror.com',HF_HUB_DISABLE_XET='1',
  WANDB_MODE='online',WANDB_ENTITY='attentionx2023',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
  SDL_VIDEODRIVER='dummy',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',LIBGL_ALWAYS_SOFTWARE='1',GALLIUM_DRIVER='llvmpipe',LP_NUM_THREADS='2',
  LD_LIBRARY_PATH='/tmp/mtxu-flow-jepa-20261007/env/lib',__EGL_VENDOR_LIBRARY_FILENAMES='/tmp/mtxu-flow-jepa-20261007/env/share/glvnd/egl_vendor.d/50_mesa.json')
 return e
def execute(args,env,name,lane):
 verify();path=BASE/'logs'/name;path.parent.mkdir(exist_ok=True)
 with path.open('ab')as f:child=subprocess.Popen(args,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
 with lock:states[lane].update(pid=child.pid,command=args);record()
 rc=child.wait()
 if rc:raise RuntimeError(f'{name} exited {rc}; inspect preserved log before resuming')
def lane_run(lane,devices):
 global cube_preparing,failed
 e=env_for(devices)
 while True:
  with lock:
   if failed:return
   if not pending:states[lane]={'state':'completed','physical_gpus':devices};record();return
   cache_ready={t:(BASE/'cache'/t/'manifest.json').exists()for t in ('pusht','cube')}
   prep=(DATA/'prepared/cube/source.json').exists() and not cache_ready['cube'] and not cube_preparing
   job=next((j for j in pending if cache_ready[j[0]]),None)
   if prep:
    cube_preparing=True;states[lane]={'state':'preparing_cube_cache','physical_gpus':devices};record()
   elif job:
    pending.remove(job);states[lane]={'state':'native_preflight','dataset':job[0],'method':job[1],'physical_gpus':devices};record()
   else:states[lane]={'state':'waiting_for_prepared_data','physical_gpus':devices};record()
  try:
   if prep:
    e['NVIDIA_TF32_OVERRIDE']='0'
    args=[PYTHON,'-u','-m','crossbench.prepare','--task','cube','--source-json',str(DATA/'prepared/cube/source.json'),'--output',str(BASE/'cache/cube'),'--gpus','4','--batch-size','512']
    execute(args,e,'cube-cache-strict-fp32.log',lane);continue
   if job is None:time.sleep(20);continue
   task,method=job;e=env_for(devices)
   if task=='cube':e['NVIDIA_TF32_OVERRIDE']='0'
   cache=BASE/'cache'/task;run=BASE/'runs'/task/method;run.mkdir(parents=True,exist_ok=True)
   if (run/'complete.json').exists():raise RuntimeError('Unexpected completed directory; refuse duplicate optimization')
   one=e.copy();one['CUDA_VISIBLE_DEVICES']=devices[0]
   execute([PYTHON,'-u','-m','crossbench.preflight','--task',task,'--method',method,'--cache',str(cache),'--output',str(run/'native-preflight.json')],one,f'{task}-{method}-preflight.log',lane)
   with lock:states[lane]['state']='training_and_periodic_evaluation';record()
   execute([PYTHON,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4','-m','crossbench.train','--task',task,'--method',method,'--cache',str(cache),'--run-dir',str(run)],e,f'{task}-{method}.log',lane)
   # A completed process is recorded only after physical reports/checkpoints exist.
   complete=json.loads((run/'complete.json').read_text());best=json.loads((run/'best.json').read_text())
   assert complete['status']=='completed' and len(complete['validation_steps'])==4
   assert complete['identity']['fixture'] is False and complete['identity']['ranks']==4
   assert (run/'validation'/f"step_{best['step']:07d}"/'report.json').exists()
   with lock:
    results.append({'dataset':task,'method':method,'returncode':0,'utc':stamp(),'complete':complete,'best':best})
    record()
    subprocess.run([PYTHON,'-m','crossbench.report','--workspace',str(BASE)],cwd=REPO,env=e,check=True)
    with(BASE/'publication-requests.jsonl').open('a')as f:f.write(json.dumps({'dataset':task,'method':method,'utc':stamp()})+'\n')
  except BaseException as error:
   with lock:failed=True;states[lane].update(state='failed_hold',error=repr(error));record()
   return
verify();selected,held=acquire_gpus(8,BASE,wait_hours=24,required=8)
save_json(BASE/'allocation.json',{'devices':selected,'ranks_per_run':4,'queue_source':source['commit']})
for lane,dev in [('A',[i for i,u in selected[:4]]),('B',[i for i,u in selected[4:]])]:states[lane]={'state':'queued','physical_gpus':dev}
with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
 futures=[pool.submit(lane_run,lane,[i for i,u in selected[sl]])for lane,sl in [('A',slice(0,4)),('B',slice(4,8))]]
 for future in futures:future.result()
for f in held:f.close()
save_json(BASE/'recovery-queue-complete.json',dict(utc=stamp(),all_sixteen_succeeded=not failed and len(results)==16,seconds=time.perf_counter()-began,source=source['commit'],results=results,final_tests_read=False))
