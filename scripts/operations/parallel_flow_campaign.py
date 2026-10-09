#!/usr/bin/env python3
"""Schedule unchanged four-GPU jobs on two pools; never change model/data budgets.

Run outside the frozen execution checkout. Suspend ONLY its sequential parent
while its current world-model subprocess continues. Resume the parent after the
registered jobs finish; it verifies and reuses their results and publishes the
comparison. On a handled failure, stop/reap this scheduler's own jobs BEFORE
resuming the original parent for exact-checkpoint recovery under existing caps.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def identity(pid):
    p = Path('/proc') / str(pid)
    fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start_ticks=fields[19], cwd=str((p / 'cwd').resolve()),
                command=(p / 'cmdline').read_bytes().decode().split('\0')[:-1])


def checked_signal(expected, sig):
    if identity(expected['pid']) != expected:
        raise RuntimeError('Process identity changed; refusing signal')
    os.kill(expected['pid'], sig)


def alive(pid):
    try:
        state = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[0]
        return state != 'Z'
    except FileNotFoundError:
        return False


def gpu_rows():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                      '--format=csv,noheader,nounits'], text=True, timeout=20)
    rows = {int(a): (b.strip(), int(c), int(d)) for a, b, c, d in csv.reader(output.splitlines())}
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                      '--format=csv,noheader,nounits'], text=True, timeout=20)
    busy = set(output.splitlines())
    return rows, busy


def idle(pool):
    rows, busy = gpu_rows()
    return all(rows[i][0] not in busy and rows[i][1] < 1024 and rows[i][2] < 5 for i in pool)


def wait_idle(pool, seconds=60):
    deadline = time.monotonic() + seconds
    while not idle(pool):
        if time.monotonic() >= deadline:
            raise RuntimeError(f'GPU pool {pool} is unexpectedly busy')
        time.sleep(2)


def run_queue(jobs, pools, start, finish, changed, *, sleep=time.sleep):
    """A pool is reused only after the previous child has exited and verified."""
    pending, active = list(jobs), {}
    try:
        while pending or active:
            for pool in pools:
                if pool not in active and pending:
                    job = pending.pop(0)
                    active[pool] = (job, start(job, pool))
                    changed('started', job, pool)
            for pool, (job, proc) in list(active.items()):
                code = proc.poll()
                if code is not None:
                    if code != 0:
                        raise RuntimeError(f'{job} exited with {code}')
                    finish(job, pool)
                    del active[pool]
                    changed('finished', job, pool)
            if active:
                sleep(5)
    except BaseException:
        # Caller must drain these processes before allowing the old parent to run.
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--record', required=True)
    p.add_argument('--supervisor', required=True, type=int)
    p.add_argument('--world-launcher', required=True, type=int)
    p.add_argument('--revision', required=True)
    a = p.parse_args()
    record = Path(a.record).resolve()
    root, repo = record / 'campaign', record / 'repo'
    cpath = repo / 'config/flow_metaworld_repair.json'
    c = json.loads(cpath.read_text())
    plan = json.loads((root / 'campaign.json').read_text())
    assert plan['code'] == a.revision and c == plan['configuration']
    assert c['seeds'] == [3072] and plan['max_gpus'] == 4
    assert c['training']['budget_seconds'] == 7200
    assert c['training']['world_steps'] == c['training']['planner_steps'] == 20000
    assert c['training']['global_batch'] == 64 and c['training']['micro_batch'] == 4
    supervisor = identity(a.supervisor)
    world_launcher = identity(a.world_launcher)
    assert 'flow_jepa.campaign' in supervisor['command'] and str(root) in supervisor['command']
    assert 'flow_jepa.train' in world_launcher['command'] and str(root) in world_launcher['command']
    assert (Path('/proc') / str(a.world_launcher) / 'stat').read_text().rsplit(')', 1)[1].split()[1] == str(a.supervisor)
    assert json.loads((root / 'status.json').read_text())['stage'] == 'world_3072'
    assert not any((root / 'runs' / f'{m}_3072').exists() for m in c['methods'] if not m.startswith('cem_'))
    baseline_manifest = hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()
    children, logs, locks = [], [], []
    stopped = False
    status = dict(scheduler_pid=os.getpid(), supervisor=supervisor, world_launcher=world_launcher,
                  started_utc=now(), execution_revision=a.revision, protocol=plan['protocol'],
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  max_simultaneous_gpus=8, gpus_per_job=4, pools=[[0,1,2,3],[4,5,6,7]],
                  unchanged='Seed, models, code, data, updates/time caps, global batch, validation and final-test cases',
                  events=[], state='acquiring_spare_pool')
    status_path = root / 'parallel-status.json'
    def update(**values):
        status.update(values, updated_utc=now())
        save(status_path, status)
    def verify_source():
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip() == a.revision
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
        assert hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest() == baseline_manifest
        assert json.loads(cpath.read_text()) == c
        assert identity(a.supervisor) == supervisor
    update()
    local_locks = Path('/tmp') / f'flow-jepa-{os.getuid()}'
    local_locks.mkdir(exist_ok=True)
    lock = (local_locks / 'repaired-20261009-parallel.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    locks.append(lock)
    try:
        # The already-running parent owns pool 0–3; acquire only idle extra GPUs.
        for _ in range(3):
            if not idle((4,5,6,7)):
                raise RuntimeError('Spare GPUs are no longer idle; no job was displaced')
            time.sleep(30)
        rows, _ = gpu_rows()
        for i in (4,5,6,7):
            lock = (local_locks / (rows[i][0] + '.lock')).open('a+')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(lock)
        if not idle((4,5,6,7)):
            raise RuntimeError('Extra GPUs became busy during acquisition')
        verify_source()
        assert json.loads((root/'status.json').read_text())['stage'] == 'world_3072'
        checked_signal(supervisor, signal.SIGSTOP)
        stopped = True
        update(state='waiting_for_world', supervisor_paused=True)
        world_dir = root / 'runs/world_3072'
        deadline = time.monotonic() + 8*3600
        while alive(a.world_launcher):
            assert identity(a.world_launcher) == world_launcher
            if time.monotonic() > deadline:
                raise TimeoutError('World job exceeded scheduler wait bound')
            time.sleep(10)
        if not (world_dir/'complete.json').exists() or not (world_dir/'best.pt').exists():
            raise RuntimeError('World exited without a selected completed checkpoint')
        world = world_dir/'best.pt'
        pools = [(0,1,2,3),(4,5,6,7)]
        methods = [m for m in c['methods'] if not m.startswith('cem_')]
        for phase, jobs in [('training', methods), ('test', c['methods'])]:
            update(state=phase)
            def start(method, pool):
                verify_source()
                wait_idle(pool)
                cmd = [sys.executable, '-m', 'torch.distributed.run', '--nnodes=1',
                       '--rdzv-backend=c10d', '--rdzv-endpoint=127.0.0.1:0', '--nproc_per_node=4',
                       '-m', 'flow_jepa.train' if phase == 'training' else 'flow_jepa.evaluate',
                       '--config', str(cpath), '--root', str(root), '--method', method,
                       '--seed', '3072', '--world', str(world)]
                if phase == 'training':
                    cmd += ['--run-dir', str(root/'runs'/f'{method}_3072')]
                else:
                    cmd += ['--output', str(root/'test'/f'{method}_3072.json')]
                    if not method.startswith('cem_'):
                        cmd += ['--checkpoint', str(root/'runs'/f'{method}_3072'/'best.pt')]
                log = (root/'logs'/f'parallel_{phase}_{method}_3072.log').open('a')
                logs.append(log)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str,pool)),
                           FLOW_EGL_DEVICES='0,0,0,0', OMP_NUM_THREADS='2', WANDB_MODE='online')
                proc = subprocess.Popen(cmd,cwd=repo,env=env,stdin=subprocess.DEVNULL,
                                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children.append(proc)
                status['last_dispatch'] = dict(phase=phase,method=method,pid=proc.pid,gpus=list(pool),command=cmd)
                return proc
            def finish(method, pool):
                path = root/'runs'/f'{method}_3072'/'complete.json' if phase == 'training' else root/'test'/f'{method}_3072.json'
                report = json.loads(path.read_text())
                if phase == 'training':
                    assert report['step'] <= 20000 and report['training_budget_seconds'] == 7200
                    assert (path.parent/'best.pt').exists()
                else:
                    assert report['wandb_synced'] and len(report['records']) == 3200
                    assert report['code'] == a.revision and report['protocol'] == plan['protocol']
            def changed(event, method, pool):
                status['events'].append(dict(utc=now(),phase=phase,event=event,method=method,gpus=list(pool)))
                update()
            run_queue(jobs,pools,start,finish,changed)
        update(state='completed_jobs', supervisor_paused=True)
    except BaseException as exc:
        update(state='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        # Own process groups only. Never signal an unrelated trainer or the world child.
        for proc in children:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for proc in children:
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
        for log in logs:
            log.close()
        if stopped:
            checked_signal(supervisor, signal.SIGCONT)
            update(supervisor_paused=False, handoff_utc=now(),
                   handoff='Original parent verifies completed reports, skips finished jobs and publishes comparison; on failure resumes incomplete jobs under preserved ledgers.')
        for lock in locks:
            lock.close()


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise RuntimeError(f'Scheduler interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    main()
