#!/usr/bin/env python3
"""One repaired-method run, fixed historical baselines, validation-only review."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from parallel_flow_campaign import alive, checked_signal, child_environment, gpu_rows, identity, now, save, wait_idle

METHOD = 'joint_flow_consistent'


def selected_validation(directory):
    reports = [json.loads(p.read_text()) | {'source_file': str(p)} for p in sorted(directory.glob('step_*.json'))]
    if not reports:
        raise ValueError('No complete validation rounds')
    return max(reports, key=lambda r: (r['metrics']['success_macro'], -r['step']))


def paired_reference_review(reports):
    groups = {r['method']: r for r in reports}
    assert set(groups) == {METHOD, 'leflow_adapted', 'hwm_adapted', 'cem_long'}
    def signature(report):
        rows = report['records']
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate validation cases')
        return sorted((r['id'], r['reset_seed'], r['episode_sha256'], r['model_seed']) for r in rows)
    ref = signature(groups[METHOD])
    if any(signature(r) != ref for r in reports):
        raise ValueError('Historical and current validation cases differ')
    ours = {r['id']: bool(r['success']) for r in groups[METHOD]['records']}
    table = {}
    for method, report in groups.items():
        rows = report['records']
        successes = sum(bool(r['success']) for r in rows)
        table[method] = dict(successes=successes, cases=len(rows), success=successes/len(rows),
            source_file=report['source_file'], selected_step=report['step'])
        if method != METHOD:
            table[method].update(ours_only_success=sum(ours[r['id']] and not r['success'] for r in rows),
                                baseline_only_success=sum(not ours[r['id']] and r['success'] for r in rows))
    return dict(table=table, split='validation', baseline_training_repeated=False,
                interpretation='Descriptive selected-validation comparison of a repaired full pipeline against frozen historical implementations. World/state representations and code differ; the old LeFlow sampler has a known defect. Not a common-world comparison, isolated sampler effect, unbiased test estimate or corrected-SOTA claim.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--record',required=True)
    args=parser.parse_args()
    record=Path(args.record).resolve(); root=record/'campaign'; repo=record/'repo'
    cancellation=json.loads((root/'baseline-queue-cancelled.json').read_text())
    assert cancellation['allowed_fresh_head']==METHOD and not cancellation['baseline_retraining_started']
    parent=cancellation['previous_scheduler_state']['supervisor']
    world_identity=cancellation['previous_scheduler_state']['world_launcher']
    revision='75e0815351eb25c63e495f87458f139bd5062259'
    cpath=repo/'config/flow_metaworld_repair.json';c=json.loads(cpath.read_text())
    plan=json.loads((root/'campaign.json').read_text())
    assert plan['code']==revision and plan['configuration']==c
    assert c['training']['budget_seconds']==7200 and c['training']['planner_steps']==20000
    state=dict(pid=os.getpid(),started_utc=now(),state='waiting_for_world',
               methods_to_train=[METHOD],baseline_retraining=False,final_test_enabled=False,
               execution_revision=revision,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def update(**kw):
        state.update(kw,updated_utc=now());save(root/'ours-only-status.json',state)
    def verify():
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==revision
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
        assert json.loads(cpath.read_text())==c
    lockdir=Path('/tmp')/f'flow-jepa-{os.getuid()}';lockdir.mkdir(exist_ok=True)
    locks=[];child=None;log=None
    try:
        lock=(lockdir/'repaired-20261009-ours-only.lock').open('a+')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
        assert identity(parent['pid'])==parent
        assert Path(f"/proc/{parent['pid']}/stat").read_text().rsplit(')',1)[1].split()[0]=='T'
        update()
        deadline=time.monotonic()+8*3600
        while alive(world_identity['pid']):
            assert identity(world_identity['pid'])==world_identity
            if time.monotonic()>deadline:raise TimeoutError('World did not finish within continuation wait bound')
            time.sleep(10)
        world=root/'runs/world_3072/best.pt'
        assert (world.parent/'complete.json').exists() and world.exists()
        verify()
        # Retire only the cancelled, stopped dispatcher after its world child exits.
        # It must never be resumed because it still contains baseline train jobs.
        checked_signal(parent,signal.SIGKILL)
        for _ in range(50):
            if not alive(parent['pid']):break
            time.sleep(.1)
        assert not alive(parent['pid'])
        wait_idle((0,1,2,3))
        gpu,_=gpu_rows()
        for i in (0,1,2,3):
            lock=(lockdir/(gpu[i][0]+'.lock')).open('a+')
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
        wait_idle((0,1,2,3))
        out=root/'runs'/f'{METHOD}_3072'
        assert not out.exists(), 'Unexpected existing ours run; inspect instead of duplicating'
        cmd=[sys.executable,'-m','torch.distributed.run','--nnodes=1','--rdzv-backend=c10d',
             '--rdzv-endpoint=127.0.0.1:0','--nproc_per_node=4','-m','flow_jepa.train',
             '--config',str(cpath),'--root',str(root),'--run-dir',str(out),'--method',METHOD,
             '--seed','3072','--world',str(world)]
        log=(root/'logs/ours_only_training.log').open('a')
        child=subprocess.Popen(cmd,cwd=repo,env=child_environment((0,1,2,3)),stdin=subprocess.DEVNULL,
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        update(state='training_ours',child_pid=child.pid,command=cmd,world_sha256=hashlib.sha256(world.read_bytes()).hexdigest())
        save(root/'status.json',dict(status='running',stage='joint_flow_consistent_3072',scope='ours_only',command=cmd))
        code=child.wait()
        if code:raise RuntimeError(f'Ours training exited {code}; retain checkpoints and charged ledger for recovery')
        assert (out/'complete.json').exists()
        chosen=selected_validation(out/'validation')
        reports=[dict(chosen,method=METHOD)]
        old=root.parent.parent/'20261007-joint-flow/campaign'
        for method in ['leflow_adapted','hwm_adapted']:
            reports.append(dict(selected_validation(old/'runs'/f'{method}_3072'/'validation'),method=method))
        cem=old/'runs/world_3072/validation/cem_step_0020000.json'
        reports.append(dict(json.loads(cem.read_text()),method='cem_long',source_file=str(cem)))
        review=paired_reference_review(reports)
        review.update(current_code=revision,historical_code='56419ed14cf127d4b7a8ab09a9d68e6681d2edc0',
                      current_world_sha256=state['world_sha256'],
                      historical_world_sha256=hashlib.sha256((old/'runs/world_3072/best.pt').read_bytes()).hexdigest())
        save(root/'historical-validation-comparison.json',review)
        sys.path.insert(0,str(repo))
        from flow_jepa.failure_analysis import analyze
        failure=analyze(c,reports);failure['comparison_scope']=review['interpretation']
        save(root/'validation-failure-analysis.json',failure)
        update(state='completed_validation_review',selected_step=chosen['step'],successes=review['table'][METHOD]['successes'])
        save(root/'status.json',dict(status='completed',stage='ours_only_validation_review',final_test_run=False,
                                    comparison=str(root/'historical-validation-comparison.json')))
    except BaseException as exc:
        update(state='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait()
        if log:log.close()
        for lock in locks:lock.close()
        # Never hand back to the cancelled all-method parent.


if __name__=='__main__':
    def stop(signum,frame):raise RuntimeError(f'Continuation interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    main()
