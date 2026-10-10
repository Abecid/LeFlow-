"""Blocking advisory locks tolerate transient EAGAIN from shared filesystems."""
import fcntl
import time


def exclusive_lock(file, timeout=30.0):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return fcntl.flock(file, fcntl.LOCK_EX)
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)
