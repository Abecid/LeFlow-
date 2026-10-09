#!/usr/bin/env python3
"""Read a remote campaign and save compact, credential-free research evidence.

Run from a reporting checkout, never the frozen execution checkout. Commit/push
changed reports after running. Large models/features stay on the server.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

REMOTE = r'''
import collections, hashlib, json, pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1])
files = {}
for name in ('campaign.json', 'status.json', 'queue.json', 'allocation.json',
             'wandb.json', 'online-readback.json', 'gpu_preflight.json',
             'data_protocol.json', 'comparison.json', 'seed-scope-change.json',
             'compute-scope-change.json', 'throughput-scope-change.json',
             'fresh-start.json', 'failure-analysis.json'):
    p = root / name
    if p.exists():
        files[name] = json.loads(p.read_text())
manifest = root / 'manifest.json'
if manifest.exists():
    raw = manifest.read_bytes()
    m = json.loads(raw)
    files['manifest-summary.json'] = {
        'sha256': hashlib.sha256(raw).hexdigest(),
        'protocol': m['protocol'], 'encoder': m['encoder'],
        'fixture': m.get('fixture'), 'statistics_scope': m.get('statistics_scope'),
        'entries': len(m['entries']), 'goal_screening': m.get('goal_screening'),
        'expert_success': m.get('expert_success'),
    }
counts = collections.Counter()
for p in (root / 'episodes').glob('*/*/*.h5'):
    counts['/'.join(p.relative_to(root / 'episodes').parts[:2])] += 1
files['preparation-progress.json'] = {'completed_episode_files': dict(sorted(counts.items())),
                                    'total': sum(counts.values())}
for run in sorted((root / 'runs').glob('*')):
    if not run.is_dir():
        continue
    prefix = 'runs/' + run.name + '/'
    for name in ('run.json', 'complete.json', 'compute_usage.json'):
        p = run / name
        if p.exists():
            files[prefix + name] = json.loads(p.read_text())
    p = run / 'metrics.jsonl'
    if p.exists():
        rows = []
        for line in p.read_text().splitlines():
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: pass  # A concurrent append may be incomplete.
        files[prefix + 'metrics.json'] = rows
    for p in sorted((run / 'validation').glob('*.json')):
        data = json.loads(p.read_text())
        files[prefix + 'validation/' + p.name] = data
    checkpoints = {}
    for p in run.glob('*.pt'):
        s = p.stat()
        checkpoints[p.name] = {'path': str(p), 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}
    if checkpoints:
        files[prefix + 'checkpoints.json'] = checkpoints
for p in sorted((root / 'test').glob('*.json')):
    data = json.loads(p.read_text())
    files['test/' + p.name] = data
print(json.dumps({'root': str(root), 'files': files}, allow_nan=False))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default='target_server_2')
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    result = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', a.host,
         'python3 - ' + shlex.quote(a.root)],
        input=REMOTE, text=True, capture_output=True, check=True, timeout=120,
    )
    snapshot = json.loads(result.stdout)
    output = Path(a.output)
    changed = []
    for name, value in snapshot['files'].items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unsafe remote report path')
        path = output / rel
        text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
        if not path.exists() or path.read_text() != text:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + '.partial')
            tmp.write_text(text)
            tmp.replace(path)
            changed.append(name)
    print(json.dumps({'root': snapshot['root'], 'changed_reports': changed}, indent=2))


if __name__ == '__main__':
    main()
