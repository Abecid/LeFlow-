from pathlib import Path
import subprocess,json,hashlib,gzip,datetime
R=Path('/Users/abecid/Documents/Codex/2026-10-07/wtf/work/LeFlow');B=R/'docs/reports/20261014-noise-cem'
subprocess.run(['git','fetch','origin','main'],cwd=R,check=True,stdout=subprocess.DEVNULL)
remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=R,text=True).split()[0]
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip();assert remote==head,(remote,head)
ledger=json.loads((R/'docs/noise-cem-shared-ledger.json').read_text());assert len(ledger['completed'])==12 and ledger['processes_exited_after_logging']==12
paths=set(ledger['artifact_sha256'])|{'docs/NOISE_CEM_SHARED_RUN_20261014.md','docs/NOISE_CEM_SHARED_RESULTS_20261014.md','docs/NOISE_CEM_SHARED_FAILURE_ANALYSIS_20261014.md','docs/noise-cem-shared-ledger.json','config/leflow_refinement/noise_comparison_20261014.json','leflow_refinement/noise_eval.py','leflow_refinement/crossbench.py','tests/test_noise_cem_adapters.py','AGENTS.md','docs/PROGRESS.md'}
for name in sorted(paths):
 local=(R/name).read_bytes();published=subprocess.check_output(['git','show',f'{remote}:{name}'],cwd=R)
 assert local==published,name
 if name in ledger['artifact_sha256']:assert hashlib.sha256(published).hexdigest()==ledger['artifact_sha256'][name],name
for row in ledger['completed']:
 raw=gzip.decompress(subprocess.check_output(['git','show',f'{remote}:{row["report"]}'],cwd=R));assert hashlib.sha256(raw).hexdigest()==row['report_original_sha256']
 x=json.loads(raw);assert len(x['records'])==50 and x['frozen_before']==x['frozen_after'] and x['optimizer_updates']==0 and not x['metrics']['timeouts']
# Old report/ledger source remains exactly as previously verified; no reruns.
subprocess.run(['git','diff','--exit-code','dd3a719',remote,'--','docs/results-ledger.json','docs/crossbench-results-ledger.json','docs/CROSSBENCH_RESULTS.md','docs/self-refine-shared-ledger.json','docs/SELF_REFINE_SHARED_RESULTS_20261014.md'],cwd=R,check=True,stdout=subprocess.DEVNULL)
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'remote_commit':remote,'remote_files_verified':len(paths),'ledger_artifact_hashes_verified':len(ledger['artifact_sha256']),'completed_runs':12,'physical_cases':600,'optimizer_updates':0,'timeouts':0,'original_author_checkpoints_used':False,'sealed_tests_read':False,'prior_results_unchanged':True}
(B/'publication-final-verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
