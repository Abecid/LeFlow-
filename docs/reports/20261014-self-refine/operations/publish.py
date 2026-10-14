from pathlib import Path
import argparse,gzip,hashlib,json,math,subprocess
p=argparse.ArgumentParser();p.add_argument('--repo',required=True);arg=p.parse_args();R=Path(arg.repo);B=R/'docs/reports/20261014-self-refine'
status=json.loads((B/'status.json').read_text());rows=[];reports={}
for item in status['results']:
 task,method=item['task'],item['method'];folder=B/'jobs'/f'{task}-{method}'
 x=json.loads(gzip.decompress((folder/'report.json.gz').read_bytes()));assert x['status']=='completed' and x['optimizer_updates']==0 and x['frozen_before']==x['frozen_after']
 cases=x['records'];assert len(cases)==50 and len({r['id']for r in cases})==50
 assert sum(r['success']for r in cases)==x['metrics']['successes']
 assert all(r['steps']<=50 and len(r['actions'])==r['steps']for r in cases)
 assert all(math.isfinite(v) and -1.000001<=v<=1.000001 for r in cases for a in r['actions']for v in a)
 assert x['identity']['numerical_precision']['NVIDIA_TF32_OVERRIDE']=='0'
 reports[task,method]=x
 rows.append({'task':task,'method':method,'successes':x['metrics']['successes'],'timeouts':x['metrics']['timeouts'],'mean_decision_ms':x['metrics']['mean_decision_ms'],'world_transitions':x['metrics']['world_transitions'],'checkpoint_sha256':x['identity']['checkpoint_sha256'],'cases_sha256':x['identity']['case_sha256'],'optimizer_updates':0,'report':str((folder/'report.json.gz').relative_to(R)),'report_original_sha256':hashlib.sha256(gzip.decompress((folder/'report.json.gz').read_bytes())).hexdigest(),'wall_seconds':x['evaluation_wall_seconds']})
pairs=[]
for (task,m),x in sorted(reports.items()):
 ref=reports.get((task,'leflow'))
 if ref is None:continue
 assert ref['identity']['case_sha256']==x['identity']['case_sha256']and ref['identity']['checkpoint_sha256']==x['identity']['checkpoint_sha256']
 a={r['id']:r for r in ref['records']};d={r['id']:r for r in x['records']};assert a.keys()==d.keys()
 for k in a:
  assert a[k]['reset_audit']==d[k]['reset_audit']
 wins=[k for k in a if not a[k]['success']and d[k]['success']];losses=[k for k in a if a[k]['success']and not d[k]['success']]
 pairs.append({'task':task,'method':m,'wins':wins,'regressions':losses})
lines=['# Frozen LeFlow P&P transfer — measured development results','',f"{len(rows)}/12 inference runs completed; zero training updates. These use retained PushT/Cube trained heads, not author checkpoints. Native Reacher/TwoRoom rows remain pending checkpoint access. All six variants share strict FP32, each task’s frozen checkpoint, original50 cases and50-action/10-controller-second allowance.",'','| Dataset | Method | Success /50 | Timeouts | Mean decision ms |','| --- | --- | ---: | ---: | ---: |']
for r in sorted(rows,key=lambda r:(r['task'],r['method'])):lines.append(f"|{r['task']}|{r['method']}|{r['successes']}|{r['timeouts']}|{r['mean_decision_ms']:.2f}|")
lines+=['','| Dataset | Method versus current LeFlow reference | Additional successes | Lost successes |','| --- | --- | ---: | ---: |']
for r in pairs:lines.append(f"|{r['task']}|{r['method']}|{len(r['wins'])}|{len(r['regressions'])}|")
lines+=['',f"Queue failure hold: {status['failed']}. Complete case reports, prediction-error traces, deadlines, candidates, full measured calls and checkpoint/frozen-state hashes are preserved in [the archive](reports/20261014-self-refine/).",'','All scalar outcomes are actual native simulator success flags. Timings are measured, not flow-call counts presented as elapsed compute. Executed-prefix prediction errors are factual; rejected candidates have no known physical outcomes. Different candidate counts do not preserve nested pools across all later decisions. No claim of TTT, training improvement, original-checkpoint reproduction, SOTA or held-out final generalization is supported by this one-seed reused-development pilot. Strict-FP32 PushT is an explicit numerical setting versus prior default-TF32 records; old baseline files remain untouched. No stored trajectory is claimed visually reviewed until it is rendered and inspected.','','[Registration and formulation](SELF_REFINE_SHARED_RUN_20261014.md).']
(R/'docs/SELF_REFINE_SHARED_RESULTS_20261014.md').write_text('\n'.join(lines)+'\n')
ledger={'benchmark_family':'retained_crossbench_leflow','source':status['source'],'completed':rows,'paired':pairs,'optimizer_updates':0,'expected_runs':12,'native_author_rows_pending':['reacher','tworoom'],'artifact_sha256':{str(q.relative_to(R)):hashlib.sha256(q.read_bytes()).hexdigest()for q in B.rglob('*')if q.is_file()}}
(R/'docs/self-refine-shared-ledger.json').write_text(json.dumps(ledger,indent=2)+'\n');print(json.dumps({'completed':len(rows),'records':len(rows)*50,'optimizer_updates':0}))
