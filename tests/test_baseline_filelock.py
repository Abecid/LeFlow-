import errno
import unittest
from unittest.mock import patch

from flow_jepa.filelock import exclusive_lock


class SharedFilesystemLockTests(unittest.TestCase):
    def test_waits_for_actual_lock_after_transient_eagain(self):
        retry = BlockingIOError(errno.EAGAIN, 'shared lock temporarily unavailable')
        with patch('flow_jepa.filelock.fcntl.flock', side_effect=[retry, retry, None]) as lock, \
                patch('flow_jepa.filelock.time.sleep') as sleep:
            exclusive_lock(7)
        self.assertEqual(lock.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_persistent_contention_is_not_silently_ignored(self):
        with patch('flow_jepa.filelock.fcntl.flock', side_effect=BlockingIOError(errno.EAGAIN, 'busy')):
            with self.assertRaises(BlockingIOError):
                exclusive_lock(7, timeout=0)

    def test_other_errors_propagate_without_retry(self):
        with patch('flow_jepa.filelock.fcntl.flock', side_effect=OSError(errno.EIO, 'broken filesystem')) as lock:
            with self.assertRaises(OSError):
                exclusive_lock(7)
        self.assertEqual(lock.call_count, 1)


if __name__ == '__main__':
    unittest.main()
