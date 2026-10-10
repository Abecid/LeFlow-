import json,subprocess,time,traceback,datetime
from pathlib import Path
record=Path('/home/mtxu/adam/LeFlow-experiments/20261010-leflow-ttt')
run=record/'campaign/runs/leflow_ttt_3072';analysis=record/'analysis'
ref=analysis/'references/docs/reports/20261010-release-baselines/leflow'
ref.mkdir(parents=True,exist_ok=True)
if not (ref/'validation').exists():(ref/'validation').symlink_to('/home/mtxu/adam/LeFlow-experiments/20261010-baseline-release/leflow_3072/validation',target_is_directory=True)
python='/tmp/mtxu-flow-jepa-20261007/env/bin/python'
last=0
try:
 while True:
  count=len(list((run/'validation').glob('step_*.json')))
  if count>last:
   subprocess.run([python,str(analysis/'leflow_ttt_details.py'),'--run-dir',str(run),'--repo',str(analysis/'references'),'--output',str(analysis/f'analysis-round-{count}.json')],check=True)
   last=count
  if (run/'registry.json').exists():
   subprocess.run([python,str(analysis/'audit_leflow_ttt.py'),'--record',str(record),'--output',str(analysis/'final-audit.json')],check=True)
   (analysis/'complete.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rounds=last))+'\n');break
  if (run/'process-exit.json').exists() and json.loads((run/'process-exit.json').read_text())['returncode']:
   raise RuntimeError('Training exited unsuccessfully; no new run will be launched')
  time.sleep(15)
except BaseException:
 (analysis/'failure.txt').write_text(traceback.format_exc());raise
