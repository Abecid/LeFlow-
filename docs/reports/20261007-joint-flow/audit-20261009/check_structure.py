import json, sys, subprocess, datetime
from pathlib import Path
import numpy as np
import torch
import h5py

torch.set_num_threads(2)
root=Path('/home/mtxu/adam/LeFlow-experiments/20261007-joint-flow')
repo=root/'repo-throughput'
sys.path.insert(0,str(repo))
from flow_jepa.models import System
c=json.loads((repo/'config/flow_metaworld.json').read_text())
data=root/'campaign'
manifest=json.loads((data/'manifest.json').read_text())
mean=np.asarray(manifest['mean'],np.float32)
std=np.asarray(manifest['std'],np.float32)
rows=[r for r in manifest['entries'] if r['split']=='validation' and r['index']==0]
out={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
     'execution_sha':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
     'device':'cpu','threads':2,'runs':{},'goal_representation':[]}
def cos(x,y):
    num=(x*y).sum(-1)
    den=np.maximum(np.linalg.norm(x,axis=-1)*np.linalg.norm(y,axis=-1),1e-8)
    return float((1-num/den).mean())
for r in rows:
    with h5py.File(data/r['path'],'r') as f:
        indices=[25,50,75]
        for t in indices:
            z=(f['z'][t].astype(np.float32)-mean)/std
            g=(f['image_goals'][t].astype(np.float32)-mean)/std
            before=(f['z'][t-5].astype(np.float32)-mean)/std
            out['goal_representation'].append({'task':r['task'],'control_step':t,
                'same_endpoint_history_vs_static_cosine_distance':cos(z,g),
                'five_control_step_history_change_cosine_distance':cos(before,z)})
with h5py.File(data/rows[0]['path'],'r') as f:
    start=torch.from_numpy((f['z'][0].astype(np.float32)-mean)/std)[None]
    goal=torch.from_numpy((f['image_goals'][60].astype(np.float32)-mean)/std)[None]
for name in ['joint_flow_consistent','leflow_adapted']:
    saved=torch.load(data/'runs'/f'{name}_3072'/'best.pt',map_location='cpu',weights_only=False)
    model=System(c,name).eval()
    model.load_state_dict(saved['model'])
    w=model.planner.zout.weight.detach()
    b=model.planner.zout.bias.detach()
    # Including bias yields an upper bound of 257 movable directions.
    basis=torch.cat([w,b[:,None]],1).double()
    q,_=torch.linalg.qr(basis,mode='reduced')
    q=q.float()
    def residual(x):return x-(x@q)@q.T
    torch.manual_seed(3072)
    nz=torch.randn(1,11,32,1024)
    na=torch.randn(1,12,5,8)
    with torch.no_grad():
        z,a=model.planner.sample(start,goal,steps=8,path_only=name=='leflow_adapted',noise=(nz.clone(),na.clone()))
    rn,rz=residual(nz),residual(z)
    out['runs'][name]={'checkpoint_step':saved['step'],'weight_shape':list(w.shape),
       'maximum_controllable_dimensions_including_bias':basis.shape[1],
       'minimum_unchangeable_dimensions':1024-basis.shape[1],
       'noise_residual_mse_per_coordinate':float(rn.square().mean()),
       'sample_residual_mse_per_coordinate':float(rz.square().mean()),
       'residual_change_max_absolute':float((rn-rz).abs().max()),
       'residual_change_rms':float((rn-rz).square().mean().sqrt())}
print(json.dumps(out,indent=2))
