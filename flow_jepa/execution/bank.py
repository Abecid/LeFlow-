"""Train-only observed-route memory; no oracle task or query duration."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from ..common import checkpoint, config, digest, file_hash, save_json
from ..models import distance


def reuse_bank(source, output, c, root):
    """Rebind metadata after verifying the unchanged training inventory.

    The source stays untouched; all tensors and recorded actions are reused
    exactly, avoiding another pass through thousands of compressed episodes.
    """
    started = time.perf_counter()
    source, output, root = Path(source), Path(output), Path(root)
    if source.resolve() == output.resolve():
        raise ValueError('Cannot overwrite a preserved route bank')
    metadata = json.loads(Path(str(source)+'.json').read_text())
    source_hash = file_hash(source)
    if source_hash != metadata['sha256']:
        raise ValueError('Source route-bank hash mismatch')
    data = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    manifest = json.loads((root/'manifest.json').read_text())
    rows = [r for r in manifest['entries'] if r['split']=='train'
            and r['mode']=='expert' and r['expert_success'] and r['steps']>=60]
    if (data['episode_ids'] != [r['id'] for r in rows]
            or data['episode_hashes'] != [r['sha256'] for r in rows]):
        raise ValueError('Source route-bank training inventory differs')
    for key in ('mean', 'std'):
        if not torch.equal(data[key], torch.tensor(manifest[key])):
            raise ValueError('Source route-bank normalization differs')
    if data['manifest'] != metadata['manifest'] or data['protocol'] != metadata['protocol']:
        raise ValueError('Source route-bank metadata differs')
    data['manifest'], data['protocol'] = file_hash(root/'manifest.json'), digest(c)
    checkpoint(output, data)
    save_json(str(output)+'.json', dict(states=len(data['raw']), routes=len(data['routes']),
        episodes=len(rows), manifest=data['manifest'], protocol=data['protocol'],
        sha256=file_hash(output), preparation_cpu_seconds=time.perf_counter()-started,
        bytes=output.stat().st_size, train_only=True, source_bank=str(source),
        source_bank_sha256=source_hash, transformation='identity tensors; protocol/manifest metadata only'))


def read_episode(path):
    # HDF5's global lock serializes threads. Independent processes decompress
    # episodes concurrently; dense reading avoids expensive strided selection.
    with h5py.File(path, 'r') as f:
        return f['image_goals'][:][::5].copy(), f['actions'][:]


def build(c, root, output):
    start = time.perf_counter()
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['protocol'] == digest(c) and not manifest.get('fixture', True)
    rows = [r for r in manifest['entries'] if r['split'] == 'train'
            and r['mode'] == 'expert' and r['expert_success'] and r['steps'] >= 60]
    assert len(rows) == 6222

    raw_states, chunks, routes, episodes, times = [], [], [], [], []
    offset = 0
    with ProcessPoolExecutor(max_workers=16) as pool:
        loaded = pool.map(read_episode, [str(root / row['path']) for row in rows], chunksize=4)
        for episode, (row, (raw, actions)) in enumerate(zip(rows, loaded)):
            n = len(raw)
            raw_states.append(raw)
            chunk = np.zeros((n, 5, 8), np.float32)
            chunk[:len(actions)//5] = actions[:len(actions)//5*5].reshape(-1, 5, 8)
            chunks.append(chunk)
            episodes.extend([episode] * n)
            times.extend(range(0, n * 5, 5))
            success = row['first_success_action'] // c['data']['action_repeat']
            for s in range(n - 1):
                if s * 5 > success:
                    continue
                for span in (1, 2, 4, 8, 12):
                    if s + span < n:
                        routes.append((offset+s, offset+s+1, offset+s+span, span*5))
            offset += n
            if (episode+1) % 500 == 0:
                print(json.dumps(dict(event='bank_progress',episodes=episode+1,total=len(rows))),flush=True)
    payload = dict(raw=torch.from_numpy(np.concatenate(raw_states)),
                   chunks=torch.from_numpy(np.concatenate(chunks)),
                   routes=torch.tensor(routes, dtype=torch.int64),
                   episode=torch.tensor(episodes), time=torch.tensor(times),
                   episode_ids=[r['id'] for r in rows],
                   episode_hashes=[r['sha256'] for r in rows],
                   mean=torch.tensor(manifest['mean']), std=torch.tensor(manifest['std']),
                   manifest=file_hash(root / 'manifest.json'), protocol=digest(c))
    checkpoint(output, payload)
    save_json(str(output)+'.json', dict(states=offset, routes=len(routes), episodes=len(rows),
              manifest=payload['manifest'], protocol=payload['protocol'],
              sha256=file_hash(output), preparation_cpu_seconds=time.perf_counter()-start,
              bytes=Path(output).stat().st_size, train_only=True))


class RouteBank:
    def __init__(self, path, c, manifest_hash, device):
        data = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        if data['protocol'] != digest(c) or data['manifest'] != manifest_hash:
            raise ValueError('Route bank identity mismatch')
        if not all(x.startswith('train/') for x in data['episode_ids']):
            raise ValueError('Non-training trajectory in route bank')
        self.raw = data['raw'].to(device)
        self.mean, self.std = data['mean'].to(device), data['std'].to(device)
        self.routes, self.chunks = data['routes'].to(device), data['chunks'].to(device)
        self.episode, self.time, self.ids = data['episode'], data['time'], data['episode_ids']
        rng = torch.Generator().manual_seed(3072)
        self.projection = (torch.randn(c['encoder']['dim'], 8, generator=rng) / np.sqrt(8)).to(device)
        self.keys = torch.cat([self.key(self.states(torch.arange(i, min(i+1024, len(self.raw)), device=device)))
                              for i in range(0, len(self.raw), 1024)])
        self.norms = self.keys.square().sum(-1)

    def states(self, ids):
        return (self.raw[ids].float() - self.mean) / self.std

    def key(self, z):
        return (F.normalize(z.float(), dim=-1, eps=1e-8) @ self.projection).flatten(1) / np.sqrt(z.shape[-2])

    @torch.no_grad()
    def query(self, z, goal, count=7):
        keys = self.key(torch.cat((z, goal)))
        approx = self.norms[:, None] + keys.square().sum(-1)[None] - 2 * self.keys @ keys.T
        scores = approx[self.routes[:, 0], 0] + approx[self.routes[:, 2], 1]
        shortlist = scores.topk(min(256, len(scores)), largest=False).indices
        routes = self.routes[shortlist]
        exact = distance(self.states(routes[:, 0]), z) + distance(self.states(routes[:, 2]), goal)
        order = exact.argsort()
        # Different spans may share the same first anchor: do not spend separate
        # controller slots on duplicate targets.
        ordered = routes[order].cpu().tolist()
        selected, seen = [], set()
        for i, route in enumerate(ordered):
            if route[1] not in seen:
                selected.append(i); seen.add(route[1])
            if len(selected) == count:
                break
        if len(selected) != count:
            raise ValueError('Insufficient distinct retrieved anchors')
        chosen = order[torch.tensor(selected, device=z.device)]
        r = routes[chosen]
        info = [dict(episode=self.ids[int(self.episode[s])], start=int(self.time[s]),
                     span=int(h)) for s, _, _, h in r.cpu().tolist()]
        return self.states(r[:, 1]), self.chunks[r[:, 0]], exact[chosen], info


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args(); build(config(a.config), a.root, a.output)
