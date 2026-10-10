"""Unchanged LeFlow query windows plus causal, recorded macro-transition history."""
import numpy as np
import torch
from ..baselines.data import BaselineSegments


class Segments(BaselineSegments):
    def __init__(self, root, c, method, length, split='train', stage='planner'):
        super().__init__(root, c, 'leflow_release', length, split, stage)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if self.stage == 'adapter':
            return item
        # Replay the exact baseline draw, without consuming its query RNG.
        rng = np.random.default_rng(np.random.SeedSequence([3072, int(index),
                                      0 if self.split == 'train' else 919]))
        task = self.names[int(index) % len(self.names)] if self.split != 'train' else self.names[int(rng.integers(len(self.names)))]
        rows = self.tasks[task]; row = rows[int(rng.integers(len(rows)))]
        h = self.c['baseline']['leflow']['horizon']
        limit = row['steps']-h
        if row.get('first_success_action') is not None:
            limit = min(limit, row['first_success_action']//self.c['data']['action_repeat'])
        start = int(rng.integers(limit+1))
        count = self.c['leflow_ttt']['history_chunks']
        hs = np.zeros((count, *item['z'].shape[1:]), np.float32)
        hn = np.zeros_like(hs); ha = np.zeros((count,h,8),np.float32)
        mask = np.zeros(count, bool)
        for j in range(count):
            left = start-(count-j)*h; right = left+h
            if left < 0: continue
            z,a = self.read(row, np.array([left,right]))
            hs[j],hn[j] = (z-self.mean)/self.std
            ha[j] = a[left:right]; mask[j] = True
        return dict(**item, history_start=torch.from_numpy(hs), history_next=torch.from_numpy(hn),
                    history_actions=torch.from_numpy(ha), history_mask=torch.from_numpy(mask),
                    query_start=torch.tensor(start))
