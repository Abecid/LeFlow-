"""Read complete saved JSON records, including hash-checked gzip parts."""
import gzip
import hashlib
import json
from pathlib import Path


PARTS_FORMAT = 'leflow.lossless-json-gzip-parts.v1'


def read_record(path):
    path = Path(path)
    raw = path.read_bytes()
    record = json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)
    if not isinstance(record, dict) or record.get('format') != PARTS_FORMAT:
        return record
    compressed = bytearray()
    seen = set()
    for part in record['parts']:
        relative = Path(part['path'])
        if relative.is_absolute() or '..' in relative.parts or str(relative) in seen:
            raise ValueError('Invalid or duplicate evidence part path')
        seen.add(str(relative))
        source = (path.parent / relative).resolve()
        if not source.is_relative_to(path.parent.resolve()):
            raise ValueError('Evidence part escapes its record directory')
        data = source.read_bytes()
        if len(data) != part['bytes'] or hashlib.sha256(data).hexdigest() != part['sha256']:
            raise ValueError('Evidence part size/hash mismatch: ' + str(relative))
        compressed.extend(data)
    if (len(compressed) != record['compressed_bytes'] or
            hashlib.sha256(compressed).hexdigest() != record['compressed_sha256']):
        raise ValueError('Combined gzip size/hash mismatch')
    raw = gzip.decompress(compressed)
    if (len(raw) != record['uncompressed_bytes'] or
            hashlib.sha256(raw).hexdigest() != record['uncompressed_sha256']):
        raise ValueError('Uncompressed record size/hash mismatch')
    return json.loads(raw)
