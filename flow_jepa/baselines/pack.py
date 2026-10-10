"""Losslessly pack registered expert features for repeated baseline reads."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time

import h5py
import numpy as np

from ..common import file_hash, save_json, digest


def copy_one(args):
    root, cache, index, row, shape = args
    source = Path(root) / row['path']
    if file_hash(source) != row['sha256']:
        raise ValueError('Source cache changed: ' + row['id'])
    z = np.memmap(Path(cache) / 'features.f16', mode='r+', dtype=np.float16, shape=shape)
    a = np.memmap(Path(cache) / 'actions.f32', mode='r+', dtype=np.float32,
                  shape=(shape[0], shape[1] - 1, 8))
    with h5py.File(source, 'r') as f:
        features, actions = f['image_goals'][:], f['actions'][:]
        if features.dtype != np.float16 or actions.dtype != np.float32:
            raise ValueError('Unexpected original precision; do not silently quantize')
        if features.shape != shape[1:] or actions.shape != a.shape[1:]:
            raise ValueError('Unexpected episode lengths')
        z[index], a[index] = features, actions
        if not np.array_equal(z[index], features) or not np.array_equal(a[index], actions):
            raise ValueError('Packed data differ from original cache')
    return index


def main(args):
    began = time.perf_counter()
    root, out = Path(args.root), Path(args.output)
    manifest = json.loads((root / 'manifest.json').read_text())
    rows = [r for r in manifest['entries'] if r['split'] in ('train', 'validation')
            and r['mode'] == 'expert' and r['expert_success']]
    signature = digest(rows)
    if (out / 'complete.json').exists():
        old = json.loads((out / 'complete.json').read_text())
        assert old['entries_sha256'] == signature
        print(json.dumps(old)); return
    out.mkdir(parents=True, exist_ok=True)
    shape = (len(rows), 101, 32, 1024)
    for name, dtype, dims in [('features.f16', np.float16, shape),
                              ('actions.f32', np.float32, (len(rows), 100, 8))]:
        with (out / name).open('wb') as f:
            f.truncate(int(np.prod(dims)) * np.dtype(dtype).itemsize)
    jobs = [(str(root), str(out), i, r, shape) for i, r in enumerate(rows)]
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for n, _ in enumerate(pool.map(copy_one, jobs, chunksize=8), 1):
            if n % 500 == 0: print(json.dumps(dict(copied=n, total=len(rows))), flush=True)
    actions = np.memmap(out / 'actions.f32', mode='r', dtype=np.float32, shape=(len(rows), 100, 8))
    indexes = [i for i, r in enumerate(rows) if r['split'] == 'train']
    train_actions = np.asarray(actions[indexes], np.float64)
    mean, std = train_actions.mean((0, 1)), train_actions.std((0, 1))
    result = dict(entries_sha256=signature, rows=rows, feature_shape=shape,
        action_mean=mean.tolist(), action_std=np.maximum(std, 1e-6).tolist(),
        source_manifest_sha256=file_hash(root / 'manifest.json'),
        source_files_and_values_verified=True, test_files_read=0,
        feature_bytes=(out / 'features.f16').stat().st_size,
        seconds=time.perf_counter() - began)
    save_json(out / 'complete.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True); p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=8)
    main(p.parse_args())
