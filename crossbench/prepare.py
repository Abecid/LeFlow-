"""Encode only TRAIN/validation frames, or reuse a compatible existing LeWM cache.

Run scripts/prepare_data.py first; source.json is the existing episode-split
contract. Test images and actions are never read by this cache builder.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import h5py
import numpy as np
import torch
from flow_jepa.common import file_hash, save_json, digest
from .world import ActionUnits, LeWMWorld
from .environment import NativeEnv


def prepare(task, source_json, output, device='cuda', batch_size=512, latent_manifest=None, gpus=1):
    from .environment import configure_renderer_device
    configure_renderer_device()
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'prepare.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source_json=Path(source_json).resolve();meta=json.loads(source_json.read_text())
    groups=[set(meta['split'][k]) for k in ('train','val','test')]
    if any(groups[i]&groups[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('Source episode splits overlap')
    if len(set.union(*groups)) != meta['source']['episodes']:
        raise ValueError('Incomplete source episode split')
    wanted=dict(source_metadata_sha256=file_hash(source_json),task=task,
                checkpoint_sha256=meta['checkpoint_sha256'],schema=2)
    if (root/'manifest.json').exists():
        old=json.loads((root/'manifest.json').read_text())
        if any(old[k]!=v for k,v in wanted.items()) or file_hash(root/'cache.h5')!=old['cache_sha256']:
            raise ValueError('Existing cache identity mismatch')
        return old
    from latent_planner import load_lewm
    import hdf5plugin  # registers source compression
    if file_hash(meta['lewm_checkpoint'])!=meta['checkpoint_sha256']:
        raise ValueError('Source checkpoint changed')
    env=NativeEnv(task)
    try:low,high=env.action_space.low.copy(),env.action_space.high.copy()
    finally:env.close()
    lewm=load_lewm(meta['lewm_checkpoint']).to(device).eval().requires_grad_(False)
    ad=int(lewm.action_encoder.patch_embed.weight.shape[1]);pd=meta['source']['action_dim']
    if ad%pd or ad//pd!=5:raise ValueError('Expected released five-primitive action blocks')
    units=ActionUnits(low,high,meta['source']['action_mean'],meta['source']['action_std'],5).to(device)
    world=LeWMWorld(lewm,units)
    reuse=None;handles={}
    if latent_manifest:
        lp=Path(latent_manifest);reuse=json.loads(lp.read_text());reuse['_root']=lp.parent
        if reuse['checkpoint_sha256']!=meta['checkpoint_sha256'] or reuse['split']!=meta['split']:
            raise ValueError('Reusable latents have different encoder or splits')
        if reuse['source']['signature']!=meta['source']['signature']:
            raise ValueError('Reusable latents belong to a different source')
    if gpus not in (1,4):raise ValueError('Preparation uses one or four assigned GPUs')
    worlds=[world]
    if not reuse:
        for rank in range(1,gpus):
            other=load_lewm(meta['lewm_checkpoint']).to(f'cuda:{rank}').eval().requires_grad_(False)
            other_units=ActionUnits(low,high,meta['source']['action_mean'],meta['source']['action_std'],5).to(f'cuda:{rank}')
            worlds.append(LeWMWorld(other,other_units))
    pools=[ThreadPoolExecutor(max_workers=1) for _ in worlds]
    def encode(rank,pixels):
        if torch.device(device).type=='cuda':torch.cuda.set_device(rank)
        return worlds[rank].encode(pixels).cpu().numpy()
    rows=[];offset=0;started=time.perf_counter();temp=root/'cache.h5.partial'
    unsupported=0;outgoing=0
    pending=deque();submitted=0
    with h5py.File(meta['source']['path'],'r') as src,h5py.File(temp,'w') as out:
        out.create_dataset('a',shape=(0,pd),maxshape=(None,pd),dtype='f4',chunks=True,compression='lzf')
        out.create_dataset('valid_action',shape=(0,),maxshape=(None,),dtype='?',chunks=True,compression='lzf')
        def write_z(start,z):
            if not np.isfinite(z).all():raise FloatingPointError('Nonfinite cached features')
            if 'z' not in out:
                out.create_dataset('z',shape=(0,*z.shape[1:]),maxshape=(None,*z.shape[1:]),dtype='f4',chunks=True,compression='lzf')
            out['z'].resize(start+len(z),axis=0);out['z'][start:start+len(z)]=z
        def drain():
            start,future=pending.popleft();write_z(start,future.result())
        for split in ('train','val'):
            for ep in meta['split'][split]:
                o,n=meta['source']['offsets'][ep],meta['source']['lengths'][ep]
                if n<26:continue
                actions=src['action'][o:o+n].astype(np.float32)
                # The final row has no outgoing transition and may be NaN.
                if not np.isfinite(actions[:-1]).all():
                    raise ValueError(f'Nonfinite outgoing actions in episode {ep}; do not guess an action-time shift')
                actions[-1]=np.where(np.isfinite(actions[-1]),actions[-1],low)
                valid=((actions>=low)&(actions<=high)).all(-1)
                unsupported+=int((~valid[:-1]).sum());outgoing+=n-1
                valid[-1]=False  # No factual transition follows the terminal row.
                # Preserve raw expert commands. PushT's step does not clip them,
                # even though a few exceed its declared policy box. Clipping
                # would attach the old outcome to a different physical action.
                a=units.unit(torch.tensor(actions,device=device),clip=False).cpu().numpy()
                if reuse:
                    shard=reuse['episode_shards'][str(ep)]
                    if shard not in handles:handles[shard]=h5py.File(reuse['_root']/shard,'r')
                    hf=handles[shard]
                    if not hf.attrs.get('complete',False):raise ValueError('Incomplete reusable latent shard')
                    latent=hf[f'episodes/{ep}/z']
                    if len(latent)!=n:raise ValueError('Latent episode length mismatch')
                for t in range(0,n,batch_size):
                    size=min(batch_size,n-t)
                    if reuse:write_z(offset+t,latent[t:t+size].astype(np.float32)[:,None])
                    else:
                        rank=submitted%len(worlds);submitted+=1
                        pixels=src['pixels'][o+t:o+t+size]
                        pending.append((offset+t,pools[rank].submit(encode,rank,pixels)))
                        if len(pending)>=2*len(worlds):drain()
                out['a'].resize(offset+n,axis=0);out['a'][offset:offset+n]=a
                out['valid_action'].resize(offset+n,axis=0);out['valid_action'][offset:offset+n]=valid
                rows.append(dict(id=ep,split=split,source_offset=o,length=n,cache_offset=offset))
                offset+=n
                if len(rows)%100==0:print(json.dumps(dict(event='cache',episodes=len(rows),frames=offset)),flush=True)
        while pending:drain()
        dim=out['z'].shape[-1];out.attrs['complete']=True
    for pool in pools:pool.shutdown(wait=True)
    for hf in handles.values():hf.close()
    temp.replace(root/'cache.h5')
    result={**wanted, 'source_path':str(Path(meta['source']['path']).resolve()),
        'source_signature':meta['source']['signature'],'source_json':str(source_json),
        'lewm_checkpoint':str(Path(meta['lewm_checkpoint']).resolve()),
        'action_mean':meta['source']['action_mean'],'action_std':meta['source']['action_std'],
        'action_low':low.tolist(),'action_high':high.tolist(),'action_block':5,'primitive_dim':pd,
        'dim':dim,'patches':1,'episodes':rows,'test_episodes_read':0,'fixture':False,
        'latent_reuse':str(latent_manifest) if reuse else None,
        'cache_sha256':file_hash(root/'cache.h5'),'preparation_seconds':time.perf_counter()-started,
        'encoder_gpus':len(worlds),'encoder_batch_size':batch_size,'encoder_dtype':'float32',
        'ordered_bounded_prefetch':True,'unsupported_outgoing_actions':unsupported,
        'outgoing_actions':outgoing,'raw_actions_clipped':False,
        'training_eligibility':'bounded factual history and complete goal prefix; evaluation case plan unchanged'}
    save_json(root/'manifest.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=['pusht','cube'],required=True)
    p.add_argument('--source-json',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--gpus',type=int,choices=[1,4],default=1)
    p.add_argument('--latent-manifest')
    a=p.parse_args();prepare(a.task,a.source_json,a.output,a.device,a.batch_size,a.latent_manifest,a.gpus)


if __name__=='__main__':main()
