"""One-run, read-only analysis continuation; never trains or opens test cases."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main(a):
    record = Path(a.record); run = record/'campaign/runs/guided_ttt_3072'
    output = record/'analysis'; output.mkdir(exist_ok=True)
    helper = Path(__file__).parent
    env = {**os.environ, 'PYTHONPATH':str(record/'repo'), 'OMP_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'2'}
    deadline = time.monotonic()+3*3600
    seen = 0
    def status(state, **fields):
        (output/'status.json').write_text(json.dumps(dict(state=state,pid=os.getpid(),
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**fields),indent=2)+'\n')
    status('waiting_for_registered_validation')
    while time.monotonic()<deadline:
        if (record/'failure.json').exists():
            status('training_failed', failure=json.loads((record/'failure.json').read_text())); return
        paths = sorted((run/'validation').glob('step_*.json'))
        complete = (run/'complete.json').exists()
        if len(paths)>seen or complete:
            subprocess.run([sys.executable,str(helper/'guided_ttt_details.py'),
                '--run-dir',str(run),'--repo',str(record/'repo'),
                '--output',str(output/f'round-{len(paths)}.json')],env=env,check=True)
            seen = len(paths)
            status('analyzed_registered_validation', rounds=seen)
        if complete:
            subprocess.run([sys.executable,str(helper/'audit_guided_ttt_run.py'),
                '--record',str(record),'--bank','/tmp/mtxu-guided-ttt-20261010/route_bank.pt',
                '--previous-bank','/tmp/mtxu-flow-reasoning-20261010/route_bank.pt',
                '--output',str(output/'final-audit.json')],env=env,check=True)
            status('complete',rounds=seen,final_tests_read=False,model_calls=0,simulator_calls=0)
            return
        time.sleep(15)
    status('analysis_wait_deadline',rounds=seen)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--record',required=True)
    try: main(p.parse_args())
    except Exception as error:
        args=p.parse_args(); output=Path(args.record)/'analysis'; output.mkdir(exist_ok=True)
        (output/'failure.json').write_text(json.dumps(dict(error=repr(error)),indent=2)+'\n')
        raise
