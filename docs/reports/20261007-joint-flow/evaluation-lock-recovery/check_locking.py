"""CPU-only four-process check of journal locking; runs no evaluation episodes."""
import datetime
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import time

BASE = Path('/home/mtxu/adam/LeFlow-experiments/20261007-joint-flow')
SCRATCH = Path('/tmp/mtxu-flow-jepa-20261007')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(BASE / 'repo-throughput'))
import torch
from flow_jepa.evaluate import EpisodeJournal

torch.set_num_threads(2)
ctx = mp.get_context('fork')
archive = BASE / 'evaluation-lock-recovery-20261009'
archive.mkdir(exist_ok=True)
shared = Path(tempfile.mkdtemp(prefix='probe-', dir=archive))
local = Path(tempfile.mkdtemp(prefix='journal-lock-probe-', dir=SCRATCH))


def lock_worker(path, barrier, queue):
    with open(path, 'a+') as lock:
        barrier.wait()
        start = time.monotonic()
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            acquired = time.monotonic()
            time.sleep(0.15)  # Force contention; not model computation.
            released = time.monotonic()
            fcntl.flock(lock, fcntl.LOCK_UN)
            queue.put(dict(ok=True, acquired=acquired, released=released,
                           waited_seconds=acquired-start))
        except OSError as exc:
            queue.put(dict(ok=False, errno=exc.errno, error=str(exc)))


def run_workers(target, *args):
    barrier, queue = ctx.Barrier(4), ctx.Queue()
    processes = [ctx.Process(target=target, args=(*args, barrier, queue))
                 for _ in range(4)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(20)
        if process.is_alive():
            process.terminate()
            process.join()
            raise RuntimeError('Probe child exceeded timeout')
        assert process.exitcode == 0
    return [queue.get(timeout=2) for _ in processes]


shared_results = run_workers(lock_worker, str(shared / 'direct.lock'))
local_results = run_workers(lock_worker, str(local / 'local.lock'))
assert all(row['ok'] for row in local_results)
intervals = sorted((row['acquired'], row['released']) for row in local_results)
assert all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))

journal_path = shared / 'journal'
journal_path.mkdir()
(journal_path / 'identity.lock').symlink_to(local / 'journal.lock')
identity = dict(method='locking_probe', seed=3072, scope='no_evaluation_episodes')


def journal_worker(path, barrier, queue):
    barrier.wait()
    try:
        for _ in range(10):
            EpisodeJournal(path, identity)
        queue.put(dict(ok=True, initializations=10))
    except Exception as exc:
        queue.put(dict(ok=False, error=repr(exc)))


journal_results = run_workers(journal_worker, str(journal_path))
assert all(row['ok'] for row in journal_results)
assert json.loads((journal_path / 'identity.json').read_text()) == identity
try:
    EpisodeJournal(journal_path, dict(identity, seed=3073))
except ValueError as exc:
    assert str(exc) == 'Evaluation resume identity mismatch'
    mismatch_rejected = True
else:
    raise AssertionError('Identity mismatch was not rejected')

report = dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              shared_probe=str(shared), local_probe=str(local),
              shared_lock_results=shared_results, local_lock_results=local_results,
              local_mutual_exclusion_verified=True,
              unchanged_episode_journal_results=journal_results,
              identity_mismatch_rejected=mismatch_rejected,
              evaluation_episodes_run=0, training_updates_run=0, gpus_used=0)
(archive / 'lock-check.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
