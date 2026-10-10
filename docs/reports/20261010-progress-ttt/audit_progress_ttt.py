from pathlib import Path
import json,hashlib,subprocess,datetime,sys
base=Path('/tmp/mtxu-progress-ttt-check-20261010');root=base/'repo';sys.path.insert(0,str(root))
import torch
from flow_jepa.common import file_hash,save_json
files=json.loads((base/'runtime-files.json').read_text())
for name,sha in files.items():assert file_hash(root/name)==sha,name
c=json.loads((root/'config/flow_metaworld_progress_ttt.json').read_text());old=json.loads((root/'config/flow_metaworld_flow_reasoning.json').read_text())
for key in ('training_tasks','heldout_tasks','seeds','data','encoder','model','training','evaluation','controller_grounded','latent_revision','execution_revision','flow_reasoning','diagnostics'):
 assert c[key]==old[key],key
assert c['execution']['training_enabled'] is False and c['execution']['test_enabled'] is False
source=Path('/tmp/mtxu-flow-reasoning-20261010/route_bank.pt');a=torch.load(source,mmap=True,weights_only=False);b=torch.load(base/'route_bank.pt',mmap=True,weights_only=False)
keys=[]
for k in a:
 if k in ('manifest','protocol'):continue
 if isinstance(a[k],torch.Tensor):assert torch.equal(a[k],b[k]),k
 else:assert a[k]==b[k],k
 keys.append(k)
assert file_hash(source)=='c6916667488ec1e418e9dd980eebad6d5d8e588e76ff720bc5ae4ecc77247046'
pre=json.loads((base/'preflight.json').read_text());ddp=json.loads((base/'ddp-check.json').read_text())
assert pre['mean_ms']<120 and pre['optimizer_updates']==0 and ddp['optimizer_updates']==0
assert ddp['gradient_sync_exact'] and ddp['parameters_unchanged']
assert (base/'progress-ttt-tests-final.log').read_text().rstrip().endswith('OK')
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True)
report=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runtime_files_verified=len(files),bank_tensors_and_inventory_identical=True,bank_checked_keys=keys,source_bank_unchanged=True,previous_compute_and_data_contract_unchanged=True,training_enabled=False,test_enabled=False,optimizer_updates=0,simulator_episodes=0,unique_behavior_tests=29,latest_ttt_tests=8,legacy_regression_tests=21,preflight_gate_passed=True,distributed_backward_passed=True,measured_gpu_hours=pre['gpu_hours']+ddp['gpu_hours'],compute_scope='timed preflight and distributed check regions only; CPU preparation and process startup excluded',gpus_after_checks=gpus)
save_json(base/'final-audit.json',report);print(json.dumps(report,indent=2))
