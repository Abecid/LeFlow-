from pathlib import Path
import argparse,gzip,hashlib,json
p=argparse.ArgumentParser();p.add_argument('--repo',default='/Users/abecid/Documents/Codex/2026-10-07/wtf/work/LeFlow');args=p.parse_args()
repo=Path(args.repo);base=repo/'docs/reports/20261013-crossbench-baselines';x=json.loads((base/'snapshot.json').read_text())
lines=['# Native PushT/Cube campaign — measured development results','', 'Sixteen authorized fresh runs; completed MetaWorld results are reused. Training source is frozen at `747f635ad36af2cbe3e2bbb0b7997efe9caccaa1`. All runs use seed3072/global64, 20k updates OR8 optimizer-GPUh, four fixed50-case evaluations,50 primitive actions AND10 cumulative controller seconds. Native success and backbone differ from MetaWorld; do not pool them as SOTA.','','| Dataset | Method | State | Success curve /50 | Best measured |','| --- | --- | --- | --- | ---: |']
ledger={'schema':1,'source':'747f635ad36af2cbe3e2bbb0b7997efe9caccaa1','registered_runs':16,'completed_runs':[],'active_runs':[],'queued_runs':[],'final_tests_read':False}
selected_reports={}
for r in x['runs']:
 task,m=r['dataset'],r['method'];folder=base/'runs'/task/m;curves=r.get('curve',[]);done='complete.json' in r and (folder/'final-audit.json').exists()
 best=max((p['successes']for p in curves),default=None);lines.append(f"| {task} | {m} | {'completed, audited'if done else 'training / periodic evaluation'} | {' / '.join(str(p['successes'])for p in curves)or 'pending'} | {best if best is not None else 'pending'} |")
 entry={'dataset':task,'method':m,'curve':[{'step':int(p['checkpoint'].split('_')[1]),'successes':p['successes'],'timeouts':p['timeouts'],'report':str((folder/'validation'/p['checkpoint']/'report.json.gz').relative_to(repo))}for p in curves], 'evidence':str(folder.relative_to(repo)),'online':r.get('wandb.json',{})}
 entry['artifact_sha256']={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest()for p in folder.rglob('*')if p.is_file()}
 if done:
  c=r['complete.json'];a=json.loads((folder/'final-audit.json').read_text());assert a['audit_passed'];entry.update(selected_step=a['selected_step'],selected_successes=a['selected_successes'],optimization_gpu_hours=c['optimization_gpu_hours'],evaluation_gpu_hours=c['evaluation_gpu_hours']);ledger['completed_runs'].append(entry)
  selected=json.loads(gzip.decompress((folder/'validation'/f"step_{a['selected_step']:07d}"/'report.json.gz').read_bytes()))
  selected_reports[task,m]=selected
  failures=[v for v in selected['records']if not v['success']]
  report=['# '+task+' — '+m+' completed run','',f"The fixed four-checkpoint success curve is **{' / '.join(str(p['successes'])for p in curves)} /50**, selecting **{a['selected_successes']}/50 at {a['selected_step']:,} updates**. The fresh policy reached {c['step']:,} updates, consuming {c['optimization_gpu_hours']:.6f} optimizer GPUh and {c['evaluation_gpu_hours']:.6f} evaluation GPUh. Train/eval wall time: {c['wall_seconds']/60:.2f} minutes.",'',f"[Online run]({a['wandb_url']}) is finished. The CPU source/data/case/checkpoint audit passes {a['runtime_files_verified']} runtime files, same case-plan hash at every checkpoint, finite model tensors and exact selected/milestone weights. Full [evidence](../reports/20261013-crossbench-baselines/runs/{task}/{m}/) preserves periodic metrics, all physical outcomes and checkpoint hashes.",'','| Update | Success /50 | Timeouts | Mean decision ms |','| --- | ---: | ---: | ---: |']
  for p in curves:report.append(f"| {int(p['checkpoint'].split('_')[1]):,} | {p['successes']} | {p['timeouts']} | {p['mean_decision_ms']:.2f} |")
  reasons={}
  for v in failures:reasons[v['failure_reason']]=reasons.get(v['failure_reason'],0)+1
  stats=c['identity']['window_sampling']
  cache=json.loads(gzip.decompress((base/'cache'/task/'manifest.json.gz').read_bytes()))
  train_episodes=sum(row['split']=='train' for row in cache['episodes'])
  dataset_note=f"This benchmark uses released frozen LeWM features rather than MetaWorld V-JEPA. The latent cache retains {train_episodes:,} TRAIN episodes; {stats['eligible_episodes']:,} episodes provide {stats['eligible_starts']:,} native-safe starts among {stats['nominal_starts']:,} nominal starts. Raw out-of-box commands are preserved, not clipped or given fabricated labels. Factual history and complete goal prefixes respect native limits; all controllers share this rule. Validation cases remain unchanged. Full-source normalization/backbone exposure are documented in the campaign runbook."
  report +=['',f"Selected failure reasons: {reasons}. Selected mean action count is {selected['metrics']['mean_steps']:.2f}; world transitions total {selected['metrics']['world_transitions']:,}. These are factual selected trajectories, not outcomes of rejected actions.",'',dataset_note,'','One seed and50 repeatedly selected development cases are screening evidence, not independent final evaluation or native-method SOTA. Other controllers continue under the same ceilings; no extra seed, MetaWorld retraining or sealed test was run.']
  dest=repo/'docs/crossbench-results';dest.mkdir(exist_ok=True);(dest/(task+'-'+m+'.md')).write_text('\n'.join(report)+'\n')
 else:ledger['active_runs'].append(entry)
queue=json.loads((base/'queue-status.json').read_text())
methods=('leflow','vanilla_action_flow','execution_revision','action_flow_search','sage_port','diffusion_policy_port','feedback_wm_port','retrieval_repair_ttt')
seen={(r['dataset'],r['method'])for r in x['runs']}
for task,m in [(t,m)for t in ('pusht','cube')for m in methods if(t,m)not in seen]:
 ledger['queued_runs'].append({'dataset':task,'method':m})
 lines.append(f"| {task} | {m} | queued | pending | pending |")
lines+=['','Paired outcomes at each method’s selected development checkpoint:','','| Dataset | Method versus plain flow | Additional successes | Lost successes | Both fail |','| --- | --- | ---: | ---: | ---: |']
ledger['paired_comparisons']=[]
for (task,m),report in sorted(selected_reports.items()):
 ref=selected_reports.get((task,'vanilla_action_flow'))
 if ref is None or m=='vanilla_action_flow':continue
 records={v['id']:v for v in report['records']};reference={v['id']:v for v in ref['records']}
 assert records.keys()==reference.keys()
 wins=[k for k,v in records.items()if v['success']and not reference[k]['success']]
 losses=[k for k,v in records.items()if not v['success']and reference[k]['success']]
 both_fail=[k for k,v in records.items()if not v['success']and not reference[k]['success']]
 ledger['paired_comparisons'].append({'dataset':task,'method':m,'reference':'vanilla_action_flow','wins':wins,'regressions':losses,'both_fail':both_fail})
 lines.append(f"| {task} | {m} | {len(wins)} | {len(losses)} | {len(both_fail)} |")
lines+=['','These pairs compare different trained systems on identical development cases. They do not isolate the effect of TTT or search, or reveal rejected-action counterfactuals.','',f"Snapshot: {x['utc']}. Queue failure hold: {queue['failed']}. Lane states: { {k:v['state'] for k,v in queue['lanes'].items()} }.",'','The registered methods use two disjoint four-GPU lanes. No missing score is estimated. All archived case outcomes are actual native simulator success flags. CPU/synthetic checks alone are not benchmark results.','','Source/data/protocol details: [launch record](CROSSBENCH_RUNS_20261013.md). Complete/interim training logs and all physical milestone reports: [archive](reports/20261013-crossbench-baselines/).']
(repo/'docs/CROSSBENCH_RESULTS.md').write_text('\n'.join(lines)+'\n');(repo/'docs/crossbench-results-ledger.json').write_text(json.dumps(ledger,indent=2)+'\n');print('Published rows prepared',len(ledger['completed_runs']),'completed',len(ledger['active_runs']),'active')
