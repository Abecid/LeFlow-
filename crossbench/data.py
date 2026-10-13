"""Episode-disjoint cached windows and TRAIN-only retrieval, without test reads."""
from __future__ import annotations
import json
import os
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F


class Windows(Dataset):
    def __init__(self, root, spec, length=None, split='train'):
        self.root,self.s=Path(root),spec
        self.meta=json.loads((self.root/'manifest.json').read_text())
        if split not in ('train','val'):
            raise ValueError('Final-test data is not exposed by the training reader')
        self.rows=[r for r in self.meta['episodes'] if r['split']==split
                   and r['length'] >= spec.horizon*spec.action_block+1]
        if not self.rows:
            raise ValueError('No eligible episodes')
        if self.meta['dim'] != spec.dim or self.meta['action_block'] != spec.action_block:
            raise ValueError('Model/cache dimensions differ')
        self.length=length or spec.updates*spec.global_batch
        self.handle=None;self.pid=None

    def __len__(self):
        return self.length

    def file(self):
        if self.pid != os.getpid():
            self.handle=h5py.File(self.root/'cache.h5','r')
            self.pid=os.getpid()
        return self.handle

    def __getitem__(self, index):
        s=self.s;h=s.horizon;k=s.action_block
        rng=np.random.default_rng(np.random.SeedSequence([s.seed,int(index)]))
        row=self.rows[int(rng.integers(len(self.rows)))]
        start=int(rng.integers(row['length']-h*k))
        maxblocks=(row['length']-1-start)//k
        offsets=[v for v in (5,10,20,40,60) if v<=maxblocks]
        offset=int(rng.choice(offsets))
        duration=int(rng.choice([v for v in (5,10,20,40,60) if v<=offset]))
        f=self.file();base=row['cache_offset']
        def z(t):
            return f['z'][base+t].astype(np.float32)
        def a(t,count):
            return f['a'][base+t:base+t+count*k].astype(np.float32).reshape(count,s.ad)
        states=np.stack([z(start+j*k) for j in range(h+1)])
        hs=np.zeros((s.history,s.patches,s.dim),np.float32)
        hn=np.zeros_like(hs);ha=np.zeros((s.history,s.ad),np.float32)
        mask=np.zeros(s.history,bool)
        count=min(s.history,start//k)
        for j in range(count):
            t=start-(count-j)*k;slot=s.history-count+j
            hs[slot],hn[slot],ha[slot]=z(t),z(t+k),a(t,1)[0]
            mask[slot]=True
        values=dict(z=states,a=a(start,h),goal=z(start+offset*k),
                    history_start=hs,history_next=hn,history_action=ha,history_mask=mask,
                    subgoal=z(start+duration*k))
        out={name:torch.from_numpy(value.copy()) for name,value in values.items()}
        out.update(episode=torch.tensor(row['id']),index=torch.tensor(index),
                   duration=torch.tensor(duration),start=torch.tensor(start))
        return out


class RouteBank:
    """Compact-state exact search; only training-episode observed routes.

    Each route's local endpoint is H blocks ahead, with a possibly farther
    same-episode final endpoint. Own-episode actions are excluded during training.
    """
    def __init__(self, root, spec, device):
        self.s=spec;root=Path(root)
        meta=json.loads((root/'manifest.json').read_text())
        starts=[];locals_=[];ends=[];chunks=[];episode=[];local_id=[]
        with h5py.File(root/'cache.h5','r') as f:
            for row in meta['episodes']:
                if row['split']!='train':continue
                o,n=row['cache_offset'],row['length'];k=spec.action_block
                z=f['z'][o:o+n].astype(np.float32)
                a=f['a'][o:o+n].astype(np.float32)
                for t in range(0,n-spec.horizon*k,spec.horizon*k):
                    for span in (5,10,20,40,60):
                        if t+span*k>=n:continue
                        starts.append(z[t]);locals_.append(z[t+spec.horizon*k]);ends.append(z[t+span*k])
                        chunks.append(a[t:t+spec.horizon*k].reshape(spec.horizon,spec.ad))
                        episode.append(row['id']);local_id.append(o+t)
        if len(set(episode))<2:
            raise ValueError('Retrieval needs two distinct TRAIN episodes')
        self.start=torch.tensor(np.asarray(starts),device=device)
        self.local=torch.tensor(np.asarray(locals_),device=device)
        self.end=torch.tensor(np.asarray(ends),device=device)
        self.chunks=torch.tensor(np.asarray(chunks),device=device)
        self.episode=torch.tensor(episode,device=device)
        self.local_id=local_id
        self.start_keys=F.normalize(self.start[:,0],dim=-1)
        self.local_keys=F.normalize(self.local[:,0],dim=-1)
        self.end_keys=F.normalize(self.end[:,0],dim=-1)

    def _cost(self, z, q, end_keys):
        return 2-F.normalize(z[:,0],dim=-1)@self.start_keys.T-F.normalize(q[:,0],dim=-1)@end_keys.T

    @torch.no_grad()
    def query(self,z,goal,count=7):
        if len(z)!=1:raise ValueError('Deployment retrieval expects one environment')
        cost=self._cost(z,goal,self.end_keys)[0]
        # Suppress identical local endpoints across alternative route durations.
        order=cost.topk(min(256,len(cost)),largest=False).indices.tolist();selected=[];seen=set()
        for i in order:
            signature=self.local_id[i]
            if signature in seen:continue
            seen.add(signature);selected.append(i)
            if len(selected)==count:break
        if len(selected)!=count:raise ValueError('Too few distinct TRAIN routes')
        ids=torch.tensor(selected,device=z.device)
        return self.local[ids],self.chunks[ids],cost[ids]

    @torch.no_grad()
    def training_query(self,z,target,excluded):
        cost=self._cost(z,target,self.local_keys)
        cost.masked_fill_(self.episode[None]==excluded[:,None],float('inf'))
        values,ids=cost.min(1)
        if not torch.isfinite(values).all():raise ValueError('No cross-episode route exists')
        return self.chunks[ids].detach()
