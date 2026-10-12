"""Ensure split publication evidence is lossless and rejects corruption."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.analysis.result_io import PARTS_FORMAT, read_record


class SavedRecordTest(unittest.TestCase):
    def test_lossless_parts_and_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {'records': [{'id': 'validation/task/00000', 'value': 0.0123456789012345}]}
            raw = json.dumps(expected, indent=2).encode()
            compressed = gzip.compress(raw, mtime=0)
            parts = []
            for index, start in enumerate(range(0, len(compressed), 19)):
                data = compressed[start:start+19]
                name = f'part{index}'
                (root/name).write_bytes(data)
                parts.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
            manifest = dict(format=PARTS_FORMAT, parts=parts,
                compressed_bytes=len(compressed), compressed_sha256=hashlib.sha256(compressed).hexdigest(),
                uncompressed_bytes=len(raw), uncompressed_sha256=hashlib.sha256(raw).hexdigest())
            path = root/'record.json'
            path.write_text(json.dumps(manifest))
            self.assertEqual(read_record(path), expected)
            (root/'part0').write_bytes(b'x'*parts[0]['bytes'])
            with self.assertRaisesRegex(ValueError, 'part size/hash'):
                read_record(path)

    def test_plain_gzip_and_invalid_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {'value': 42}
            path = root/'record.json.gz'
            path.write_bytes(gzip.compress(json.dumps(expected).encode()))
            self.assertEqual(read_record(path), expected)
            for relative in ('../outside', '/absolute'):
                path = root/'bad.json'
                path.write_text(json.dumps(dict(format=PARTS_FORMAT, parts=[dict(path=relative)])))
                with self.assertRaisesRegex(ValueError, 'Invalid'):
                    read_record(path)


if __name__ == '__main__':
    unittest.main()
