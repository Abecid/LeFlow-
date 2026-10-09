"""Evaluate immutable checkpoints on a separate four-GPU pool."""
import argparse
from datetime import timedelta
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch
import torch.distributed as dist
from flow_jepa.common import (checkpoint, config, digest, distributed, file_hash,
                              git_revision, save_json)
from flow_jepa.evaluate import evaluate, load_models
from flow_jepa.vision import Encoder


def child_env(devices):
    excluded={'RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK',
              'ROLE_RANK','ROLE_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'}
    env={k:v for k,v in os.environ.items() if k not in excluded and not k.startswith('TORCHELASTIC_')}
    env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,devices)), FLOW_DEVICE='cuda',
               FLOW_EGL_DEVICES=','.join('0' for _ in devices), OMP_NUM_THREADS='2', CUDA_MODULE_LOADING='LAZY')
    return env


class EvaluationPool:
    def __init__(self, root, run_dir, config_path, world_path, seed, devices=(4,5,6,7)):
        self.root, self.run_dir=Path(root),Path(run_dir)
        self.directory=self.run_dir/'async_validation';self.directory.mkdir(exist_ok=True)
        self.config_path=str(Path(config_path).resolve());self.world_path=str(Path(world_path).resolve())
        self.seed,self.devices=seed,devices
        self.proc,self.active,self.log=None,None,None
        self.started=time.perf_counter()

    def submit(self, saved, *, final=False):
        step=saved['step'];path=self.directory/f'step_{step:07d}.pt'
        jobpath=path.with_suffix('.json')
        if jobpath.exists():
            raise ValueError('Duplicate validation submission')
        checkpoint(path,saved)
        save_json(jobpath,dict(step=step,checkpoint=str(path.resolve()),
             checkpoint_sha256=file_hash(path),root=str(self.root.resolve()),
             output=str((self.run_dir/'validation'/f'step_{step:07d}.json').resolve()),
             config=self.config_path,world=self.world_path,seed=self.seed,
             method=saved['method'],protocol=saved['protocol'],manifest=saved['manifest'],
             model_code=saved['code'],world_sha256=saved['world_hash'],devices=list(range(8)) if final else list(self.devices)))

    def poll(self):
        if self.proc is not None:
            code=self.proc.poll()
            if code is None:return []
            self.log.close();self.log=None
            active=self.active;self.proc=None;self.active=None
            if code:raise RuntimeError(f'Async validation exited {code}: {active}')
            job=json.loads(active.read_text())
            if not Path(job['output']).exists():raise RuntimeError('Evaluator exited without report')
        ready=[];pending=[]
        for path in sorted(self.directory.glob('step_*.json')):
            if path.name.endswith('.consumed.json'):continue
            job=json.loads(path.read_text());result=Path(job['output'])
            if result.exists():
                if not path.with_suffix('.consumed.json').exists():ready.append((path,job,json.loads(result.read_text())))
            else:pending.append(path)
        if pending:
            self.active=pending[0]
            job=json.loads(self.active.read_text())
            devices=job['devices']
            self.log=self.active.with_suffix('.log').open('a')
            cmd=[sys.executable,'-m','torch.distributed.run','--nnodes=1',
                 '--rdzv-backend=c10d','--rdzv-endpoint=127.0.0.1:0',
                 f'--nproc_per_node={len(devices)}','-m','flow_jepa.runtime.async_eval','--job',str(self.active)]
            self.proc=subprocess.Popen(cmd,env=child_env(devices),stdin=subprocess.DEVNULL,
                stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
            save_json(self.directory/'worker.json',dict(pid=self.proc.pid,job=str(self.active),command=cmd,devices=list(devices)))
        return ready

    def acknowledge(self,path):save_json(path.with_suffix('.consumed.json'),dict(consumed=True))

    def outstanding(self):
        return any(not p.with_suffix('.consumed.json').exists() for p in self.directory.glob('step_*.json') if not p.name.endswith('.consumed.json'))

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            os.killpg(self.proc.pid,signal.SIGTERM)
            try:self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(self.proc.pid,signal.SIGKILL);self.proc.wait()
        if self.log is not None:self.log.close()


def worker(job_path):
    # This worker alone supports eight-way episode sharding. Training stays at
    # four ranks and its checkpoint/RNG topology is unchanged.
    rank,size=int(os.environ['RANK']),int(os.environ['WORLD_SIZE'])
    local=int(os.environ['LOCAL_RANK']);assert size in (4,8)
    device=torch.device(f'cuda:{local}');torch.cuda.set_device(device)
    os.environ.setdefault('MUJOCO_GL','egl')
    os.environ['MUJOCO_EGL_DEVICE_ID']=os.environ['FLOW_EGL_DEVICES'].split(',')[local]
    dist.init_process_group('nccl',timeout=timedelta(hours=1))
    importlib.import_module('flow_jepa.evaluate').distributed=lambda: (rank,size,device)
    job=json.loads(Path(job_path).read_text());c=config(job['config'])
    assert job['protocol']==digest(c) and job['model_code']==git_revision()
    assert file_hash(job['checkpoint'])==job['checkpoint_sha256']
    assert file_hash(job['world'])==job['world_sha256']
    assert file_hash(Path(job['root'])/'manifest.json')==job['manifest']
    started=time.perf_counter()
    model,world,saved=load_models(c,job['world'],job['method'],job['checkpoint'],device)
    assert saved['step']==job['step'] and saved['seed']==job['seed']
    encoder=Encoder(c['encoder'],job['root'],device)
    records,summary=evaluate(model,world,c,job['root'],job['method'],job['seed'],
                             'validation',encoder,device,c['training']['validation_episodes_per_task'])
    torch.cuda.synchronize();elapsed=torch.tensor(time.perf_counter()-started,device=device,dtype=torch.float64)
    dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
    if rank==0:
        save_json(job['output'],dict(step=job['step'],metrics=summary,records=records,
            execution=dict(mode='async_episode_sharding',gpu_count=size,devices=job['devices'],
                           seconds=float(elapsed),gpu_hours=float(elapsed)*size/3600,
                           checkpoint_sha256=job['checkpoint_sha256'],controller_unchanged=True)))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);worker(p.parse_args().job)
