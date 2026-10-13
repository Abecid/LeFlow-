import concurrent.futures,datetime,hashlib,json,os,subprocess,threading,time
from pathlib import Path
base=Path('/home/mtxu/adam/LeFlow-experiments/20261012-additional-baselines')
python='/tmp/mtxu-flow-jepa-20261007/env/bin/python'
previous='/home/mtxu/adam/LeFlow-experiments/20261009-repaired-comparison'
source=json.loads((base/'runtime-files.json').read_text())
lanes=[('A','0,1,2,3',['vanilla_action_flow','action_flow_search','sage_port']),('B','4,5,6,7',['diffusion_policy_port','feedback_wm_port'])]
lock=threading.Lock();state={};began=time.perf_counter()

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write_state():
 tmp=base/'queue-status.partial';tmp.write_text(json.dumps({'source':source['commit'],'updated_utc':stamp(),'lanes':state},indent=2)+'\n');tmp.replace(base/'queue-status.json')
def start_ticks(pid):return int(Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19])
def lane_run(lane,visible,methods):
 results=[]
 for method in methods:
  record=base/method;repo=record/'repo';cache='/tmp/mtxu-additional-baselines-20261012/'+method
  assert not (record/'launcher.json').exists(),str(record)
  assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==source['commit']
  assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True)
  for name,sha in source['files'].items():assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==sha,name
  env=os.environ.copy();runtime='/tmp/mtxu-flow-jepa-20261007/env'
  env.update(CUDA_VISIBLE_DEVICES=visible,CUDA_MODULE_LOADING='LAZY',FLOW_DEVICE='cuda',
      OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',WANDB_MODE='online',WANDB_ENTITY='attentionx2023',
      LD_LIBRARY_PATH=runtime+'/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else ''),
      MUJOCO_GL='egl',FLOW_EGL_DEVICES='0,0,0,0',LIBGL_ALWAYS_SOFTWARE='1',
      GALLIUM_DRIVER='llvmpipe',LP_NUM_THREADS='2',
      __EGL_VENDOR_LIBRARY_FILENAMES=runtime+'/share/glvnd/egl_vendor.d/50_mesa.json')
  command=[python,'-u','-m','scripts.operations.controller_grounded','--record',str(record),
      '--variant',method,'--gpus','4','--previous',previous,'--cache',cache,
      '--reuse-bank','/tmp/mtxu-flow-reasoning-20261010/route_bank.pt']
  with (record/'coordinator.log').open('ab') as log:
   child=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,
       stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  launch={'pid':child.pid,'start_ticks':start_ticks(child.pid),'source':source['commit'],
      'runtime_files_verified':len(source['files']),'utc':stamp(),'gpu_lane':lane,
      'physical_gpus':visible,'command':command,'fresh_seed':3072,'final_tests':False}
  (record/'launcher.json').write_text(json.dumps(launch,indent=2)+'\n')
  with lock:state[lane]={'current':method,'pid':child.pid,'state':'preflight_then_training','remaining':methods[len(results)+1:],'completed':results};write_state()
  rc=child.wait()
  results.append({'method':method,'returncode':rc,'ended_utc':stamp()})
  if rc:
   with lock:state[lane]={'state':'failed_hold','failed_method':method,'completed':results,'remaining':methods[len(results):]};write_state()
   return False
 with lock:state[lane]={'state':'completed','completed':results,'remaining':[]};write_state()
 return True

for lane,visible,methods in lanes:state[lane]={'state':'queued','physical_gpus':visible,'remaining':methods,'completed':[]}
write_state()
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
 outcomes=list(executor.map(lambda x:lane_run(*x),lanes))
(base/'queue-complete.json').write_text(json.dumps({'source':source['commit'],'completed_utc':stamp(),'all_five_succeeded':all(outcomes),'seconds':time.perf_counter()-began,'lanes':state},indent=2)+'\n')
