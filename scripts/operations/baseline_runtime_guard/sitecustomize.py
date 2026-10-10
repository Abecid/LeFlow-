"""Expose worker exceptions before NCCL cleanup can hide them.

Install as sitecustomize.py only for an explicitly recorded recovery. This does
not change model computation, random state, data, evaluation, or normal teardown.
"""
import os
import sys
import traceback
import fcntl
import time


def install_journal_lock_guard():
    original = fcntl.flock

    def journal_lock(file, operation):
        fd = file if isinstance(file, int) else file.fileno()
        target = os.readlink(f'/proc/self/fd/{fd}')
        if operation != fcntl.LOCK_EX or not target.endswith('/identity.lock'):
            return original(file, operation)
        deadline = time.monotonic() + 30.0
        while True:
            try:
                return original(file, operation)
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)

    fcntl.flock = journal_lock

if 'flow_jepa.baselines.train' in sys.orig_argv:
    install_journal_lock_guard()
    import torch.distributed as dist

    _destroy_process_group = dist.destroy_process_group

    def _destroy_with_error_report(*args, **kwargs):
        if sys.exc_info()[0] is not None:
            traceback.print_exc()
            sys.stderr.flush()
            sys.stdout.flush()
            # The launcher terminates the sibling workers when this rank exits.
            # Collective teardown cannot succeed if they are awaiting a result.
            os._exit(1)
        return _destroy_process_group(*args, **kwargs)

    dist.destroy_process_group = _destroy_with_error_report
