import importlib.util
from pathlib import Path
import signal

import pytest

spec = importlib.util.spec_from_file_location('parallel_scheduler', Path(__file__).parents[1]/'scripts/operations/parallel_flow_campaign.py')
scheduler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduler)


def test_child_uses_four_visible_gpus_and_explicit_cuda(monkeypatch):
    monkeypatch.setenv('MUJOCO_GL','egl')
    env = scheduler.child_environment((4,5,6,7))
    assert env['CUDA_VISIBLE_DEVICES'] == '4,5,6,7'
    assert env['FLOW_DEVICE'] == 'cuda'
    assert env['FLOW_EGL_DEVICES'] == '0,0,0,0'
    assert env['MUJOCO_GL'] == 'egl'


def test_two_pools_never_share_a_job_or_overlap_on_one_pool():
    active, events, started = set(), [], []
    class Process:
        def __init__(self, remaining): self.remaining = remaining
        def poll(self):
            self.remaining -= 1
            return 0 if self.remaining <= 0 else None
    def start(job,pool):
        assert pool not in active
        active.add(pool);started.append((job,pool))
        return Process({'ours':4,'leflow':1,'hwm':2}[job])
    def finish(job,pool): active.remove(pool)
    scheduler.run_queue(['ours','leflow','hwm'], [(0,1,2,3),(4,5,6,7)], start,finish,
                        lambda *args:events.append(args),sleep=lambda _:None)
    assert started == [('ours',(0,1,2,3)),('leflow',(4,5,6,7)),('hwm',(4,5,6,7))]
    assert not active and len(events) == 6


def test_failure_does_not_dispatch_the_next_job():
    class Failed:
        def poll(self): return 1
    started=[]
    def start(job,pool):started.append(job);return Failed()
    with pytest.raises(RuntimeError,match='exited with 1'):
        scheduler.run_queue(['bad','never'],[(0,1,2,3)],start,lambda *a:None,lambda *a:None)
    assert started == ['bad']


def test_pid_reuse_cannot_signal_unrelated_process(monkeypatch):
    calls=[]
    monkeypatch.setattr(scheduler,'identity',lambda pid:dict(pid=pid,start_ticks='changed'))
    monkeypatch.setattr(scheduler.os,'kill',lambda *args:calls.append(args))
    with pytest.raises(RuntimeError,match='identity changed'):
        scheduler.checked_signal(dict(pid=123,start_ticks='original'),signal.SIGSTOP)
    assert calls == []
