#!/usr/bin/env python3
"""Checkpoint-boundary migration to equivalent batching and parallel validation."""
import argparse
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import signal
import struct
import subprocess
import sys
import time
import torch
from parallel_flow_campaign import alive, checked_signal, child_environment, gpu_rows, identity, now, save, wait_idle
from ours_only_continuation import selected_validation, paired_reference_review

METHOD='joint_flow_consistent'
BASE='75e0815351eb25c63e495f87458f139bd5062259'


def migrate(record):
    root=record/'campaign';run=root/'runs'/f'{METHOD}_3072';original=record/'repo';runtime=record/'repo-runtime'
    state=json.loads((root/'ours-only-status.json').read_text())
    assert state['state']=='training_ours' and state['methods_to_train']==[METHOD]
    assert not (run/'complete.json').exists()
    parent=identity(state['pid']);child=identity(state['child_pid'])
    assert parent['pid']==807392 and child['pid']==858560
    assert os.getpgid(child['pid'])==child['pid']
    assert subprocess.check_output(['git','status','--porcelain'],cwd=original,text=True).strip()==''
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=runtime,text=True).strip()==BASE
    assert subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=runtime,text=True).strip()==''
    spec=json.loads((root/'throughput-runtime.json').read_text())
    for name,expected in spec['files'].items():
        assert hashlib.sha256((runtime/name).read_bytes()).hexdigest()==expected,name
    bench=json.loads((root/'fused-throughput-benchmark.json').read_text())
    assert bench['cuda_rng_identical'] and bench['gradient_relative_l2']<1e-4
    assert bench['adam_parameter_max_abs']<2e-6 and bench['speedup']>1.2
    assert spec['tests_passed']==27
    # Inotify reacts at the atomic last.pt rename, before the checkpoint barrier.
    libc=ctypes.CDLL(None,use_errno=True);fd=libc.inotify_init1(os.O_NONBLOCK)
    if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
    wd=libc.inotify_add_watch(fd,os.fsencode(run),0x80)
    if wd<0:raise OSError(ctypes.get_errno(),'inotify_add_watch')
    checked_signal(parent,signal.SIGSTOP)
    paused=False;retired=False
    save(root/'throughput-migration-status.json',dict(state='waiting_for_checkpoint',pid=os.getpid(),parent=parent,child=child,started_utc=now()))
    deadline=time.monotonic()+1200
    try:
        while time.monotonic()<deadline:
            if not select.select([fd],[],[],2)[0]:
                assert alive(child['pid']);continue
            data=os.read(fd,65536);offset=0;found=False
            while offset<len(data):
                _,mask,_,length=struct.unpack_from('iIII',data,offset)
                name=data[offset+16:offset+16+length].split(b'\0')[0]
                offset+=16+length
                if mask&0x80 and name==b'last.pt':found=True
            if not found:continue
            assert identity(child['pid'])==child
            os.killpg(child['pid'],signal.SIGSTOP);paused=True;time.sleep(.2)
            saved=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
            ledger=json.loads((run/'compute_usage.json').read_text())
            if ledger['step']!=saved['step']:
                os.killpg(child['pid'],signal.SIGCONT);paused=False;continue
            assert saved['world_size']==4 and saved['seed']==3072 and saved['code']==BASE
            assert saved['step']<20000 and saved['training_seconds']<7200
            validation_files=list((run/'validation').glob('step_*.json'))
            assert len(validation_files)==saved['validation_round']
            archive=root/'throughput-handoff'/f'step_{saved["step"]:07d}';archive.mkdir(parents=True,exist_ok=False)
            for name in ['last.pt','best.pt','compute_usage.json','run.json']:
                if (run/name).exists():shutil.copy2(run/name,archive/name)
            last_hash=hashlib.sha256((run/'last.pt').read_bytes()).hexdigest()
            assert last_hash==hashlib.sha256((archive/'last.pt').read_bytes()).hexdigest()
            # Retain all charged time and conservatively charge any unfinished
            # next update interrupted before its ledger write. No completed update
            # is discarded, no research update allowance is reset.
            ledger['training_seconds']+=1.0
            ledger['training_gpu_hours']=ledger['training_seconds']*4/3600
            ledger['migration_inflight_allowance_seconds']=1.0
            save(run/'compute_usage.json',ledger)
            event=dict(state='checkpoint_preserved',pid=os.getpid(),observed_utc=now(),parent=parent,
                child=child,checkpoint_step=saved['step'],checkpoint_sha256=last_hash,
                completed_updates_discarded=0,conservative_inflight_charge_seconds=1.0,
                archive=str(archive),runtime_manifest_sha256=hashlib.sha256((root/'throughput-runtime.json').read_bytes()).hexdigest(),
                prior_status=state,prior_checkpoint_training_seconds=saved['training_seconds'],ledger=ledger)
            save(root/'throughput-migration.json',event)
            checked_signal(parent,signal.SIGKILL)
            os.killpg(child['pid'],signal.SIGKILL);retired=True;paused=False
            for _ in range(100):
                if not alive(child['pid']) and not alive(parent['pid']):break
                time.sleep(.1)
            assert not alive(child['pid']) and not alive(parent['pid'])
            save(root/'throughput-migration-status.json',dict(state='ready_to_resume',checkpoint_step=saved['step'],pid=os.getpid(),updated_utc=now()))
            return
        raise TimeoutError('No safe checkpoint boundary within20 minutes')
    finally:
        os.close(fd)
        if not retired:
            if paused:os.killpg(child['pid'],signal.SIGCONT)
            if alive(parent['pid']):checked_signal(parent,signal.SIGCONT)


def migrate_detached_workers(record):
    """Complete a handoff when torchrun workers have independent process groups."""
    root=record/'campaign';run=root/'runs'/f'{METHOD}_3072'
    prior=json.loads((root/'throughput-migration.json').read_text())
    assert prior['checkpoint_step']==7000
    for pid in [807392,858560,906402]:assert not alive(pid)
    workers=[]
    for pid in [858566,858567,858568,858569]:
        item=identity(pid)
        assert item['start_ticks']=='984503717' and item['cwd']==str(record/'repo')
        assert 'flow_jepa.train' in item['command'] and str(run) in item['command']
        assert os.getpgid(pid)==pid
        workers.append(item)
    spec=json.loads((root/'throughput-runtime.json').read_text())
    for name,expected in spec['files'].items():
        assert hashlib.sha256((record/'repo-runtime'/name).read_bytes()).hexdigest()==expected,name
    libc=ctypes.CDLL(None,use_errno=True);fd=libc.inotify_init1(os.O_NONBLOCK)
    if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
    wd=libc.inotify_add_watch(fd,os.fsencode(run),0x80)
    if wd<0:raise OSError(ctypes.get_errno(),'inotify_add_watch')
    def group_signal(sig):
        for worker in workers:
            assert identity(worker['pid'])==worker
            os.killpg(worker['pid'],sig)
    stopped=False;retired=False
    save(root/'throughput-migration-status.json',dict(state='waiting_for_worker_checkpoint',pid=os.getpid(),workers=workers,started_utc=now()))
    try:
        deadline=time.monotonic()+1200
        while time.monotonic()<deadline:
            if not select.select([fd],[],[],2)[0]:
                assert all(alive(x['pid']) for x in workers);continue
            data=os.read(fd,65536);offset=0;found=False
            while offset<len(data):
                _,mask,_,length=struct.unpack_from('iIII',data,offset)
                name=data[offset+16:offset+16+length].split(b'\0')[0];offset+=16+length
                if mask&0x80 and name==b'last.pt':found=True
            if not found:continue
            group_signal(signal.SIGSTOP);stopped=True;time.sleep(.2)
            assert all((Path('/proc')/str(w['pid'])/'stat').read_text().rsplit(')',1)[1].split()[0]=='T' for w in workers)
            saved=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
            ledger=json.loads((run/'compute_usage.json').read_text())
            if ledger['step']!=saved['step']:
                group_signal(signal.SIGCONT);stopped=False;continue
            assert saved['step']>=8000 and saved['step']<20000
            assert saved['world_size']==4 and saved['seed']==3072 and saved['code']==BASE
            assert len(list((run/'validation').glob('step_*.json')))==saved['validation_round']
            archive=root/'throughput-handoff'/f'step_{saved["step"]:07d}';archive.mkdir(parents=True,exist_ok=False)
            for name in ['last.pt','best.pt','compute_usage.json','run.json']:
                if (run/name).exists():shutil.copy2(run/name,archive/name)
            sha=hashlib.sha256((run/'last.pt').read_bytes()).hexdigest()
            assert sha==hashlib.sha256((archive/'last.pt').read_bytes()).hexdigest()
            save(root/'throughput-migration-attempt-7000.json',dict(prior,actual_outcome='Only launcher retired; independent training worker groups continued. No optimized child dispatched. All later updates preserved at final handoff.'))
            ledger['training_seconds']+=1.0;ledger['training_gpu_hours']=ledger['training_seconds']*4/3600
            ledger['migration_inflight_allowance_seconds']=1.0;save(run/'compute_usage.json',ledger)
            event=dict(state='checkpoint_preserved_all_workers',pid=os.getpid(),observed_utc=now(),workers=workers,
                checkpoint_step=saved['step'],checkpoint_sha256=sha,completed_updates_discarded=0,
                conservative_inflight_charge_seconds=1.0,archive=str(archive),ledger=ledger,
                runtime_manifest_sha256=hashlib.sha256((root/'throughput-runtime.json').read_bytes()).hexdigest(),
                correction='Explicitly stopped all four independent torchrun worker process groups at a later checkpoint; no completed updates rolled back.')
            save(root/'throughput-migration.json',event)
            group_signal(signal.SIGKILL);retired=True;stopped=False
            for _ in range(100):
                if all(not alive(w['pid']) for w in workers):break
                time.sleep(.1)
            assert all(not alive(w['pid']) for w in workers)
            save(root/'throughput-migration-status.json',dict(state='ready_to_resume_all_workers_retired',pid=os.getpid(),checkpoint_step=saved['step'],updated_utc=now()))
            return
        raise TimeoutError('No safe worker checkpoint boundary')
    finally:
        os.close(fd)
        if stopped and not retired:group_signal(signal.SIGCONT)


def main():
    p=argparse.ArgumentParser();p.add_argument('--record',required=True);p.add_argument('--migrate',action='store_true');p.add_argument('--migrate-orphans',action='store_true');a=p.parse_args()
    record=Path(a.record).resolve();root=record/'campaign';repo=record/'repo-runtime';run=root/'runs'/f'{METHOD}_3072'
    lockdir=Path('/tmp')/f'flow-jepa-{os.getuid()}';locks=[];child=None;log=None
    lock=(lockdir/'repaired-20261009-throughput-migration.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
    state=dict(pid=os.getpid(),state='preparing_throughput',started_utc=now(),methods_to_train=[METHOD],
        baseline_retraining=False,final_test_enabled=False,execution_revision=BASE,
        runtime_manifest_sha256=hashlib.sha256((root/'throughput-runtime.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),training_gpus=[0,1,2,3],validation_gpus=[4,5,6,7],final_validation_gpus=list(range(8)))
    def update(**kw):state.update(kw,updated_utc=now());save(root/'ours-only-status.json',state)
    try:
        if a.migrate:
            raise RuntimeError("Use explicit verified-worker migration; launcher-only mode is retired")
        if a.migrate_orphans:migrate_detached_workers(record)
        assert (root/'throughput-migration.json').exists()
        for pid in [807392,858560,858566,858567,858568,858569]:assert not alive(pid),'Old run still active'
        wait_idle(tuple(range(8)),seconds=90)
        gpu,_=gpu_rows()
        for i in range(8):
            lock=(lockdir/(gpu[i][0]+'.lock')).open('a+')
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
        lock=(lockdir/'repaired-20261009-ours-only.lock').open('a+')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
        cpath=record/'repo/config/flow_metaworld_repair.json';world=root/'runs/world_3072/best.pt'
        c=json.loads(cpath.read_text())
        cmd=[sys.executable,'-m','torch.distributed.run','--nnodes=1','--rdzv-backend=c10d','--rdzv-endpoint=127.0.0.1:0','--nproc_per_node=4','-m','flow_jepa.runtime.train','--config',str(cpath),'--root',str(root),'--run-dir',str(run),'--method',METHOD,'--seed','3072','--world',str(world)]
        log=(root/'logs/ours_throughput_training.log').open('a')
        child=subprocess.Popen(cmd,cwd=repo,env=child_environment((0,1,2,3)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        update(state='training_ours_optimized',child_pid=child.pid,command=cmd,world_sha256=hashlib.sha256(world.read_bytes()).hexdigest())
        save(root/'status.json',dict(status='running',stage='joint_flow_consistent_3072',scope='ours_only_all_gpu_throughput',command=cmd))
        code=child.wait()
        if code:raise RuntimeError(f'Optimized continuation exited{code}; preserve ledger/checkpoints and inspect')
        assert (run/'complete.json').exists()
        reports=[dict(selected_validation(run/'validation'),method=METHOD)]
        old=record.parent/'20261007-joint-flow/campaign'
        for method in ['leflow_adapted','hwm_adapted']:
            reports.append(dict(selected_validation(old/'runs'/f'{method}_3072'/'validation'),method=method))
        cem=old/'runs/world_3072/validation/cem_step_0020000.json'
        reports.append(dict(json.loads(cem.read_text()),method='cem_long',source_file=str(cem)))
        review=paired_reference_review(reports)
        review.update(current_code=BASE,historical_code='56419ed14cf127d4b7a8ab09a9d68e6681d2edc0',current_world_sha256=state['world_sha256'],runtime_manifest_sha256=state['runtime_manifest_sha256'])
        current_cem=json.loads((root/'runs/world_3072/validation/cem_step_0020000.json').read_text())
        review['same_new_world_cem']=dict(successes=sum(r['success'] for r in current_cem['records']),cases=104,step=20000,success=current_cem['metrics']['success_macro'],additional_evaluation_run=False)
        save(root/'historical-validation-comparison.json',review)
        sys.path.insert(0,str(repo));from flow_jepa.failure_analysis import analyze
        failure=analyze(c,reports);failure['comparison_scope']=review['interpretation'];save(root/'validation-failure-analysis.json',failure)
        update(state='completed_validation_review',selected_step=reports[0]['step'],successes=review['table'][METHOD]['successes'])
        save(root/'status.json',dict(status='completed',stage='ours_only_validation_review',final_test_run=False))
    except BaseException as exc:
        # Do not overwrite original coordinator state if migration safely backed out.
        if (root/'throughput-migration.json').exists():update(state='failed',error=f'{type(exc).__name__}: {exc}')
        else:save(root/'throughput-migration-status.json',dict(state='failed_original_resumed',error=f'{type(exc).__name__}: {exc}',updated_utc=now()))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        if log:log.close()
        for lock in reversed(locks):lock.close()

if __name__=='__main__':main()
