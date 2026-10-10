"""Baseline-only offline sampling over the registered episode inventory."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ..common import digest


class BaselineSegments(Dataset):
    def __init__(self, root, c, method, length, split='train', stage='planner'):
        self.root, self.c, self.method = Path(root), c, method
        self.length, self.split, self.stage = length, split, stage
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if self.manifest['protocol'] != digest(c):
            raise ValueError('Baseline manifest differs from its registered configuration')
        self.mean = np.asarray(self.manifest['mean'], np.float32)
        self.std = np.asarray(self.manifest['std'], np.float32)
        self.tasks = {t: [] for t in sorted(c['training_tasks'])}
        for row in self.manifest['entries']:
            if row['split'] != split:
                continue
            if row['mode'] == 'expert' and row['expert_success']:
                self.tasks[row['task']].append(row)
        self.names = list(self.tasks)
        assert all(self.tasks.values())
        self.packed = None
        cache = c['baseline'].get('packed_cache')
        if cache:
            meta = json.loads((Path(cache) / 'complete.json').read_text())
            registered = [r for r in self.manifest['entries'] if r['split'] in ('train', 'validation')
                          and r['mode'] == 'expert' and r['expert_success']]
            if digest(registered) != meta['entries_sha256']:
                raise ValueError('Packed features differ from registered episodes')
            self.packed = (Path(cache), tuple(meta['feature_shape']))
            self.row_index = {r['id']: i for i, r in enumerate(meta['rows'])}
        self._features = self._actions = None

    def read(self, row, selection):
        if self.packed:
            if self._features is None:
                cache, shape = self.packed
                self._features = np.memmap(cache / 'features.f16', mode='r', dtype=np.float16, shape=shape)
                self._actions = np.memmap(cache / 'actions.f32', mode='r', dtype=np.float32,
                                           shape=(shape[0], shape[1] - 1, 8))
            i = self.row_index[row['id']]
            return self._features[i, selection].astype(np.float32), self._actions[i]
        with h5py.File(self.root / row['path'], 'r') as f:
            return f['image_goals'][selection].astype(np.float32), f['actions'][:].astype(np.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Task/episode uniform, as in the shared benchmark. The same pre-success
        # start rule removes collection padding for ALL methods. No goal-offset
        # curriculum or recovery-specific weighting is imported from our policy.
        rng = np.random.default_rng(np.random.SeedSequence([3072, int(index),
                                      0 if self.split == 'train' else 919]))
        task = self.names[int(index) % len(self.names)] if self.split != 'train' else self.names[int(rng.integers(len(self.names)))]
        rows = self.tasks[task]
        row = rows[int(rng.integers(len(rows)))]
        if self.stage == 'adapter':
            ids = np.array([int(rng.integers(row['steps'] + 1))])
            z, _ = self.read(row, ids)
            z = (z - self.mean) / self.std
            return {'z': torch.from_numpy(z.copy())}
        if self.method == 'leflow_release':
            h = self.c['baseline']['leflow']['horizon']
            limit = row['steps'] - h
            if row.get('first_success_action') is not None:
                limit = min(limit, row['first_success_action'] // self.c['data']['action_repeat'])
            start = int(rng.integers(limit + 1))
            z, a = self.read(row, slice(start, start+h+1))
            z = (z - self.mean) / self.std
            actions = a[start:start+h]
            return dict(z=torch.from_numpy(z.copy()), a=torch.from_numpy(actions.copy()))
        raise ValueError(self.method)
