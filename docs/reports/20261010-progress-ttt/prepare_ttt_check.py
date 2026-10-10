from pathlib import Path
import json,shutil,sys,subprocess,os
base=Path('/tmp/mtxu-progress-ttt-check-20261010'); repo=base/'repo'
sys.path.insert(0,str(repo));os.chdir(repo)
from flow_jepa.common import config,file_hash,digest,save_json
from flow_jepa.campaign import adopt_data_manifest
from flow_jepa.execution.bank import reuse_bank
source=Path('/home/mtxu/adam/LeFlow-experiments/20261010-flow-reasoning/campaign')
root=base/'data';root.mkdir(exist_ok=True)
c=config(repo/'config/flow_metaworld_progress_ttt.json')
for name in ('episodes','weights','vendor'):
 p=root/name
 if not p.exists():p.symlink_to((source/name).resolve(),target_is_directory=True)
if not (root/'manifest.json').exists():
 shutil.copy2(source/'manifest.json',root/'manifest.json')
 adopt_data_manifest(root,config(repo/'config/flow_metaworld_flow_reasoning.json'),c)
 before=json.loads((source/'manifest.json').read_text());after=json.loads((root/'manifest.json').read_text())
 for key in ('entries','encoder','mean','std','goal_screening','expert_success'):assert before[key]==after[key],key
 save_json(base/'data-verification.json',dict(source=str(source),source_manifest=file_hash(source/'manifest.json'),check_manifest=file_hash(root/'manifest.json'),entries_identical=True,training_enabled=False,test_enabled=False))
bank=base/'route_bank.pt'
if not bank.exists():reuse_bank('/tmp/mtxu-flow-reasoning-20261010/route_bank.pt',bank,c,root)
with (base/'tests.log').open('w') as log:
 for name in ('test_execution.py','test_latent_revision.py','test_execution_revision.py','test_flow_reasoning.py','test_progress_ttt.py'):
  subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p',name,'-v'],stdout=log,stderr=subprocess.STDOUT,check=True)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0')
with (base/'preflight.log').open('w') as log:
 subprocess.run([sys.executable,'-u','-m','flow_jepa.execution.preflight','--config',str(repo/'config/flow_metaworld_progress_ttt.json'),'--root',str(root),'--world','/home/mtxu/adam/LeFlow-experiments/20261009-repaired-comparison/campaign/runs/world_3072/best.pt','--bank',str(bank),'--output',str(base/'preflight.json')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
print((base/'preflight.json').read_text())
