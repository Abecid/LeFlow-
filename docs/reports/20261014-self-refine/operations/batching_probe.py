from pathlib import Path
import argparse,sys,time,json,copy,h5py,hdf5plugin,torch
b=Path('/home/mtxu/adam/LeFlow-experiments/20261014-self-refine');sys.path.insert(0,str(b/'runtime'))
from self_refine.shared_eval import load,SharedController
from crossbench.evaluate import cases
from flow_jepa.common import seed_all
p=argparse.ArgumentParser();p.add_argument('--task',required=True);arg=p.parse_args()
old=Path('/home/mtxu/adam/LeFlow-experiments/20261013-crossbench-baselines');m,w,s,meta,_=load(old/'cache'/arg.task,old/'runs'/arg.task/'leflow/best.pt','cuda:0');torch.set_num_threads(2)
row=cases(meta,s,count=1,split='train')[0]
with h5py.File(meta['source_path'],'r')as f:
 ix=row['source_offset']+row['start'];z=w.encode(f['pixels'][ix:ix+1]);g=w.encode(f['pixels'][ix+row['goal_offset']:ix+row['goal_offset']+1])
methods=json.loads((b/'runtime/config/self_refine/native_comparison.json').read_text())['methods'];reports=[]
for method in methods[-2:]:
 a=copy.deepcopy(method);d=copy.deepcopy(method);d['refinement']['guidance_batch_size']=64;controllers=[];times=[]
 for cfg in (a,d):
  c=SharedController(m,w,s,cfg);seed_all(s.seed);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
  with torch.inference_mode():c.plan(z,g)
  torch.cuda.synchronize();times.append({'seconds':time.perf_counter()-start,'peak_memory_bytes':torch.cuda.max_memory_allocated()});controllers.append(c)
 p,q=controllers;deltaA=float((p.last_candidate_actions-q.last_candidate_actions).abs().max());deltaJ=float((p.last_candidate_scores-q.last_candidate_scores).abs().max())
 torch.testing.assert_close(p.last_candidate_actions,q.last_candidate_actions,rtol=1e-4,atol=1e-5);torch.testing.assert_close(p.last_candidate_scores,q.last_candidate_scores,rtol=1e-5,atol=1e-6)
 assert p.trace[-1]['selected_candidate']==q.trace[-1]['selected_candidate']
 reports.append({'method':method['name'],'micro8':times[0],'batch64':times[1],'candidate_action_max_abs':deltaA,'candidate_score_max_abs':deltaJ,'selected_same':True,'TRAIN_episode':row['episode'],'optimizer_updates':0,'validation_cases':0,'test_cases':0})
(b/f'batching-{arg.task}.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports))
