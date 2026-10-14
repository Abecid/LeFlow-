from pathlib import Path
from dataclasses import replace
import argparse,json,sys,time,h5py,hdf5plugin,torch
b=Path('/home/mtxu/adam/LeFlow-experiments/20261014-noise-cem');sys.path.insert(0,str(b/'runtime'))
from self_refine.shared_eval import load
from leflow_refinement.crossbench import RefinementController
from leflow_refinement.noise_search import NoiseSearchConfig,noise_variants
from self_refine.native import state_fingerprint
from self_refine.preflight import validate_native_runtime
from latent_planner import LatentPlannerRuntime
from crossbench.controller import Controller
from crossbench.evaluate import cases
from flow_jepa.common import seed_all
from flow_jepa.budget import PlanningBudget,PlanningBudgetExceeded
p=argparse.ArgumentParser();p.add_argument('--task',required=True);a=p.parse_args()
old=Path('/home/mtxu/adam/LeFlow-experiments/20261013-crossbench-baselines');cache=old/'cache'/a.task;checkpoint=old/'runs'/a.task/'leflow/best.pt'
torch.cuda.set_device(0);torch.set_num_threads(2);tick=time.perf_counter();m,w,s,meta,saved=load(cache,checkpoint,'cuda:0')
gate=validate_native_runtime(LatentPlannerRuntime(w.lewm,m.flow,m.inverse,s.action_block),device='cuda:0')
row=cases(meta,s,count=1,split='train')[0]
with h5py.File(meta['source_path'],'r')as f:
 pixels=f['pixels'][row['source_offset']+row['start']];goal=f['pixels'][row['source_offset']+row['start']+row['goal_offset']]
z,g=w.encode(pixels[None]),w.encode(goal[None]);records=[]
old_controller=Controller(m,w,None,s);seed_all(s.seed);expected=old_controller.plan(z,g);rng=torch.cuda.get_rng_state().clone()
new_controller=RefinementController(m,w,s,NoiseSearchConfig(rounds=1));seed_all(s.seed);actual=new_controller.plan(z,g)
torch.testing.assert_close(actual,expected,rtol=0,atol=0);torch.testing.assert_close(torch.cuda.get_rng_state(),rng,rtol=0,atol=0)
before={'policy':state_fingerprint(m),'world':state_fingerprint(w)}
for name,cfg in noise_variants().items():
 c=RefinementController(m,w,s,cfg);seed_all(s.seed);prior=w.calls;torch.cuda.synchronize();start=time.perf_counter()
 with torch.inference_mode():out=c.plan(z,g)
 torch.cuda.synchronize();assert torch.isfinite(out).all()and out.abs().max()<=1
 trace=c.trace[-1];assert trace['flow_candidate_evaluations']==3072 and trace['world_candidate_transitions']==960 and w.calls-prior==960
 assert trace['backward_batches']==0 and trace['noise_search']['rounds_completed']==cfg.rounds
 costs=trace['noise_search']['incumbent_costs'][0];assert all(costs[i]<=costs[i-1]for i in range(1,len(costs)))
 if name=='noise_cem':assert costs[0]==new_controller.trace[-1]['local_cost']
 if cfg.project_shell:assert all(abs(v-(768**.5))<1e-4 for v in trace['noise_search']['mean_input_noise_norm'][0])
 c.observe(z);assert w.calls-prior==960
 records.append({'method':name,'seconds':time.perf_counter()-start,'action_dim':out.numel(),'work':trace})
c=RefinementController(m,w,s,NoiseSearchConfig());prior=w.calls
try:c.plan(z,g,budget=PlanningBudget(-1))
except PlanningBudgetExceeded:pass
else:raise AssertionError('Deadline did not stop search')
assert w.calls==prior and not c.trace[-1]['plan_completed']
assert before=={'policy':state_fingerprint(m),'world':state_fingerprint(w)}
assert all(v.grad is None for v in m.parameters())and all(v.grad is None for v in w.parameters())
report={'task':a.task,'native_ABI_on_retained_head':gate,'author_native_head':False,'retained_baseline_action_and_RNG_bitwise':True,'weights_unchanged':True,'checks':records,'TRAIN_episode':row['episode'],'validation_cases_read':0,'test_cases_read':0,'optimizer_updates':0,'seconds':time.perf_counter()-tick}
(b/'preflight').mkdir(exist_ok=True);(b/'preflight'/f'{a.task}.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
