from pathlib import Path
import argparse,gzip,hashlib,json,math,subprocess
p=argparse.ArgumentParser();p.add_argument('--repo',default='/Users/abecid/Documents/Codex/2026-10-07/wtf/work/LeFlow');args=p.parse_args()
repo=Path(args.repo);l=json.loads((repo/'docs/crossbench-results-ledger.json').read_text());plans={};n=0
for r in l['completed_runs']:
 meta=json.loads(gzip.decompress((repo/'docs/reports/20261013-crossbench-baselines/cache'/r['dataset']/'manifest.json.gz').read_bytes()))
 for p,h in r['artifact_sha256'].items():assert hashlib.sha256((repo/p).read_bytes()).hexdigest()==h,p
 points=[]
 for c in r['curve']:
  x=json.loads(gzip.decompress((repo/c['report']).read_bytes()));rows=x['records'];assert len(rows)==50
  keys=sorted(tuple(v[k]for k in ['id','episode','source_offset','start','goal_offset','seed'])for v in rows)
  assert keys==plans.setdefault(r['dataset'],keys);assert sum(v['success']for v in rows)==c['successes'];assert sum(v['timeout']for v in rows)==c['timeouts'];assert all(v['steps']<=50 for v in rows);points.append((c['successes'],-c['step'],c['step']));n+=1
  for v in rows:
   assert len(v['actions'])==v['steps']
   assert v['success']==(v['failure_reason']=='success')
   assert v['timeout']==(v['failure_reason']=='controller_timeout')
   assert all(math.isfinite(a) and a>=low-1e-6 and a<=high+1e-6 for command in v['actions'] for a,low,high in zip(command,meta['action_low'],meta['action_high']))
   assert all(len(command)==meta['primitive_dim'] for command in v['actions'])
 assert max(points)[0]==r['selected_successes'] and max(points)[2]==r['selected_step']
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=repo,text=True).split()[0]
subprocess.run(['git','fetch','origin','main'],cwd=repo,check=True,stdout=subprocess.DEVNULL)
subprocess.run(['git','merge-base','--is-ancestor',head,remote],cwd=repo,check=True)
paths={'docs/CROSSBENCH_RESULTS.md','docs/crossbench-results-ledger.json'}|{p for r in l['completed_runs']for p in r['artifact_sha256']}
for p in paths:assert subprocess.check_output(['git','show',remote+':'+p],cwd=repo)==(repo/p).read_bytes(),p
print(json.dumps({'remote_commit':remote,'completed_native_runs':len(l['completed_runs']),'physical_reports':n,'remote_files_verified':len(paths),'paired_case_identities_match':True,'final_test_outcomes_read':False}))
