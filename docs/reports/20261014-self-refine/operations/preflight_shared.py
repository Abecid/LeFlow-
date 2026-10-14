from pathlib import Path
import argparse,json,sys,time,h5py,hdf5plugin,torch
b=Path('/home/mtxu/adam/LeFlow-experiments/20261014-self-refine');sys.path.insert(0,str(b/'runtime'))
from self_refine.shared_eval import load,SharedController
from self_refine.native import state_fingerprint
from self_refine.preflight import validate_native_runtime
from latent_planner import LatentPlannerRuntime
from crossbench.controller import Controller
from crossbench.evaluate import cases
from flow_jepa.common import seed_all
p=argparse.ArgumentParser();p.add_argument('--task',required=True);a=p.parse_args()
old=Path('/home/mtxu/adam/LeFlow-experiments/20261013-crossbench-baselines');cache=old/'cache'/a.task;checkpoint=old/'runs'/a.task/'leflow/best.pt'
torch.cuda.set_device(0);torch.set_num_threads(2);tick=time.perf_counter();m,w,s,meta,saved=load(cache,checkpoint,'cuda:0')
r=LatentPlannerRuntime(w.lewm,m.flow,m.inverse,s.action_block);gate=validate_native_runtime(r,device='cuda:0')
row=cases(meta,s,count=1,split='train')[0]
with h5py.File(meta['source_path'],'r')as f:
 pixels=f['pixels'][row['source_offset']+row['start']];goal=f['pixels'][row['source_offset']+row['start']+row['goal_offset']]
z,g=w.encode(pixels[None]),w.encode(goal[None]);config=json.loads((b/'runtime/config/self_refine/shared_comparison_20261014.json').read_text());records=[]
old_controller=Controller(m,w,None,s);seed_all(s.seed);expected=old_controller.plan(z,g);rng=torch.cuda.get_rng_state().clone()
new_controller=SharedController(m,w,s,config['methods'][0]);seed_all(s.seed);actual=new_controller.plan(z,g)
torch.testing.assert_close(actual,expected,rtol=0,atol=0);torch.testing.assert_close(torch.cuda.get_rng_state(),rng,rtol=0,atol=0)
before={'policy':state_fingerprint(m),'world':state_fingerprint(w)}
for method in config['methods']:
 c=SharedController(m,w,s,method);seed_all(s.seed);torch.cuda.synchronize();start=time.perf_counter()
 with torch.inference_mode():out=c.plan(z,g)
 torch.cuda.synchronize();assert torch.isfinite(out).all()and out.abs().max()<=1
 records.append({'method':method['name'],'seconds':time.perf_counter()-start,'action_dim':out.numel(),'work':c.trace[-1]})
assert before=={'policy':state_fingerprint(m),'world':state_fingerprint(w)}
assert all(v.grad is None for v in m.parameters())and all(v.grad is None for v in w.parameters())
report={'task':a.task,'native_ABI_on_retained_head':gate,'author_native_head':False,'retained_baseline_action_and_RNG_bitwise':True,'weights_unchanged':True,'checks':records,'TRAIN_episode':row['episode'],'validation_cases_read':0,'test_cases_read':0,'optimizer_updates':0,'seconds':time.perf_counter()-tick}
(b/'preflight').mkdir(exist_ok=True);(b/'preflight'/f'{a.task}.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
