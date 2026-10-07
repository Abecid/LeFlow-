from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .common import barrier, config, digest, distributed, episode_seed, file_hash, save_json
from .environment import collect_episode
from .vision import Encoder, history_clip


def episode_plan(c):
    rows = []
    for split in ('train', 'validation', 'test'):
        tasks = c['training_tasks'] + (c['heldout_tasks'] if split == 'test' else [])
        count = c['data'][split + '_episodes_per_task']
        for task in tasks:
            for i in range(count):
                mode = 'random' if split == 'train' and i < int(count*c['data']['random_fraction']) else 'expert'
                rows.append(dict(id=f'{split}/{task}/{i:05d}', task=task, split=split,
                                 seed=episode_seed(task, split, i), mode=mode, index=i))
    return rows


def _encode_list(encoder, clips, batch):
    result = []
    for start in range(0, len(clips), batch):
        result.append(encoder(np.stack(clips[start:start+batch])).cpu().numpy().astype(np.float16))
    return np.concatenate(result)


def prepare(c, root, *, encoder_factory=Encoder):
    rank, size, device = distributed()
    root = Path(root)
    plan = episode_plan(c)
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        lock = root / 'data_protocol.json'
        if lock.exists() and json.loads(lock.read_text()) != c:
            raise ValueError('Data directory belongs to another protocol; use a new directory')
        save_json(lock, c)
        encoder = encoder_factory(c['encoder'], root, device)
    barrier()
    if rank != 0:
        encoder = encoder_factory(c['encoder'], root, device)
    for pos in range(rank, len(plan), size):
        row = plan[pos]
        path = root / 'episodes' / (row['id'] + '.h5')
        if path.exists():
            with h5py.File(path, 'r') as f:
                if f.attrs.get('protocol') != digest(c) or f.attrs.get('encoder') != encoder.fingerprint:
                    raise ValueError(f'Mismatched completed cache: {path}')
            continue
        episode = collect_episode(row['task'], row['seed'], c['data'], row['mode'])
        frames = episode.pop('frames')
        history = c['data']['history_frames']
        repeat = c['data']['action_repeat']
        indexes = list(range(0, len(frames), repeat))
        goal_image = frames[episode['goal_index']]
        goal = encoder(np.repeat(goal_image[None, None], history, axis=1)).cpu().numpy()[0]
        # One RGB final goal only; no expert motion history is exposed at evaluation.
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.partial')
        with h5py.File(tmp, 'w') as f:
            f.attrs.update(**row, protocol=digest(c), encoder=encoder.fingerprint,
                           expert_success=episode['expert_success'], goal_index=episode['goal_index'])
            f.create_dataset('initial_rgb', data=frames[0], compression='gzip')
            f.create_dataset('goal_rgb', data=goal_image, compression='gzip')
            f.create_dataset('goal', data=goal.astype(np.float16))
            f.create_dataset('success', data=episode['successes'])
            if row['split'] != 'test':
                clips = [history_clip(frames, i, history) for i in indexes]
                z = _encode_list(encoder, clips, c['encoder']['batch_size'])
                # Hindsight goals are also single images, encoded identically to deployment.
                static = [np.repeat(frames[i][None], history, axis=0) for i in indexes]
                gz = _encode_list(encoder, static, c['encoder']['batch_size'])
                n = len(indexes)-1
                f.create_dataset('z', data=z, compression='lzf')
                f.create_dataset('image_goals', data=gz, compression='lzf')
                f.create_dataset('actions', data=episode['actions'][:n*repeat].reshape(n, repeat*4))
            f.flush()
        tmp.replace(path)
        print(json.dumps(dict(event='cached_episode', rank=rank, episode=row['id'], success=episode['expert_success'])), flush=True)
    barrier()
    if rank == 0:
        entries, total, squared, count = [], None, None, 0
        expert = {}
        for row in plan:
            path = root / 'episodes' / (row['id'] + '.h5')
            with h5py.File(path, 'r') as f:
                item = {**row, 'path': str(path.relative_to(root)), 'sha256': file_hash(path),
                        'expert_success': bool(f.attrs['expert_success']),
                        'steps': len(f['actions']) if 'actions' in f else 0}
                entries.append(item)
                if row['mode'] == 'expert':
                    expert.setdefault(row['split'] + '/' + row['task'], []).append(item['expert_success'])
                if row['split'] == 'train':
                    z = f['z'][:].astype(np.float64).reshape(-1, c['encoder']['dim'])
                    if not np.isfinite(z).all():
                        raise ValueError('Nonfinite training features')
                    total = z.sum(0) if total is None else total + z.sum(0)
                    squared = np.square(z).sum(0) if squared is None else squared + np.square(z).sum(0)
                    count += len(z)
        mean = total/count
        std = np.sqrt(np.maximum(squared/count-mean**2, 1e-6))
        save_json(root / 'manifest.json', dict(protocol=digest(c), encoder=encoder.fingerprint,
                  mean=mean.tolist(), std=std.tolist(), entries=entries,
                  expert_success={k: float(np.mean(v)) for k, v in expert.items()},
                  statistics_scope='training_episodes_only', fixture=False))
    barrier()


class Segments(Dataset):
    """Stateless sampling makes the data sequence reproducible across resume."""
    def __init__(self, root, c, stage, seed, length, split='train'):
        self.root, self.c, self.stage, self.seed, self.length = Path(root), c, stage, seed, length
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        if self.manifest['protocol'] != digest(c):
            raise ValueError('Configuration differs from prepared data')
        self.mean = np.asarray(self.manifest['mean'], np.float32)
        self.std = np.asarray(self.manifest['std'], np.float32)
        k, m = c['model']['chunk_steps'], c['model']['segments']
        self.horizon = k if stage in ('world', 'hwm_adapted') else k*m
        self.tasks = {}
        for row in self.manifest['entries']:
            if row['split'] != split or row['steps'] < self.horizon:
                continue
            if stage != 'world' and (row['mode'] != 'expert' or not row['expert_success']):
                continue
            self.tasks.setdefault(row['task'], []).append(row)
        if set(self.tasks) != set(c['training_tasks']):
            raise ValueError('Some training tasks have no eligible data; inspect expert success and horizon')
        self.names = sorted(self.tasks)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        rows = self.tasks[self.names[int(rng.integers(len(self.names)))]]
        row = rows[int(rng.integers(len(rows)))]
        start = int(rng.integers(row['steps']-self.horizon+1))
        k, m = self.c['model']['chunk_steps'], self.c['model']['segments']
        with h5py.File(self.root/row['path'], 'r') as f:
            if self.stage in ('world', 'hwm_adapted'):
                z = (f['z'][start:start+k+1].astype(np.float32)-self.mean)/self.std
                a = f['actions'][start:start+k].astype(np.float32)
                return {'z': torch.from_numpy(z), 'a': torch.from_numpy(a)}
            z = (f['z'][start:start+self.horizon+1:k].astype(np.float32)-self.mean)/self.std
            goal = (f['image_goals'][start+self.horizon].astype(np.float32)-self.mean)/self.std
            z[-1] = goal
            a = f['actions'][start:start+self.horizon].astype(np.float32).reshape(m, k, -1)
            # Observed one-chunk targets for the separate LeFlow inverse control.
            local = (f['z'][start:start+k+1:k].astype(np.float32)-self.mean)/self.std
            return {'z': torch.from_numpy(z), 'a': torch.from_numpy(a), 'local': torch.from_numpy(local)}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config/flow_metaworld.json')
    p.add_argument('--root', required=True)
    args = p.parse_args()
    prepare(config(args.config), args.root)
