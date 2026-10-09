"""Eight-GPU single-candidate training; fixed aggregate compute and validation."""
import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from ..common import checkpoint, config, digest, file_hash, git_revision, require_execution, save_json, seed_all
from ..models import System
from ..evaluate import evaluate, EpisodeJournal
from ..vision import Encoder
from .bank import RouteBank
from .controller import ExecutionController
from .model import ChunkPolicy, ExecutionSegments


def load_world(c, path, device):
    if file_hash(path) != c['controller_grounded']['world_sha256']:
        raise ValueError('Frozen world hash mismatch')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['method'] != 'world' or saved['seed'] != 3072 or saved['fixture']:
        raise ValueError('Invalid shared-world source')
    model = System(c, 'world')
    model.load_state_dict(saved['model'])
    return model.world.to(device).eval().requires_grad_(False), saved


def initialize():
    rank, size, local = (int(os.environ.get(k, d)) for k, d in
                         [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')])
    assert size in (1, 4, 8)
    device = torch.device(f'cuda:{local}')
    torch.cuda.set_device(device)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ['MUJOCO_EGL_DEVICE_ID'] = '0'
    if size > 1:
        dist.init_process_group('nccl', timeout=timedelta(hours=2))
    return rank, size, device


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())


def restore_rng(value):
    random.setstate(value['python']); np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch']); torch.cuda.set_rng_state(value['cuda'])


def train(args):
    c = config(args.config)
    root, run_dir = Path(args.root), Path(args.run_dir)
    require_execution(c, root, 'training')
    assert c['seeds'] == [3072] and c['methods'] == ['controller_grounded']
    assert not c['execution']['test_enabled']
    rank, size, device = initialize()
    torch.set_num_threads(2)
    manifest = json.loads((root / 'manifest.json').read_text())
    manifest_hash = file_hash(root / 'manifest.json')
    if manifest.get('fixture', True) or manifest['protocol'] != digest(c):
        raise ValueError('Wrong real-data manifest')
    tc = c['training']; steps, batch = tc['planner_steps'], tc['global_batch']
    assert steps == 20000 and batch == 64 and batch % size == 0
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_all(3072)
    model = ChunkPolicy(c).to(device)
    world, source = load_world(c, args.world, device)
    reuse = json.loads((root / 'reuse.json').read_text())
    assert source['manifest'] == reuse['world_source_manifest']
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc['learning_rate'],
                                  weight_decay=tc['weight_decay'])
    used, start, best, best_step, validation_gpu_seconds, run_id = 0.0, 0, -1.0, None, 0.0, None
    validation_steps = []
    saved = None
    last = run_dir / 'last.pt'
    identity = dict(protocol=digest(c), manifest=manifest_hash, world_hash=file_hash(args.world),
                    bank_hash=json.loads(Path(args.bank+'.json').read_text())['sha256'],
                    code=git_revision(), method='controller_grounded', seed=3072, world_size=size)
    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=False)
        for k, v in identity.items():
            if saved[k] != v:
                raise ValueError(f'Resume identity mismatch: {k}')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        start, used = saved['step'], saved['optimization_gpu_seconds']
        best, best_step, validation_steps = saved['best'], saved['best_step'], saved['validation_steps']
        validation_gpu_seconds = saved['validation_gpu_seconds']; run_id = saved['wandb_id']
        ledger = json.loads((run_dir / 'compute_usage.json').read_text())
        used = max(used, ledger['optimization_gpu_seconds'])
    elif (run_dir / 'run.json').exists():
        raise RuntimeError('Existing run without recoverable checkpoint')
    module = DDP(model, device_ids=[device.index], broadcast_buffers=False) if size > 1 else model
    seed_all(3072 + 997 * rank)
    data = ExecutionSegments(root, c, 'controller_grounded', 3072, steps * batch)
    data = Subset(data, range(start * batch, len(data)))
    sampler = DistributedSampler(data, size, rank, shuffle=False, drop_last=True) if size > 1 else None
    loader = DataLoader(data, batch_size=batch//size, sampler=sampler, shuffle=False,
                        drop_last=True, num_workers=4, prefetch_factor=4, pin_memory=True,
                        persistent_workers=True, generator=torch.Generator().manual_seed(3072))
    if saved:
        restore_rng(saved['rng'][rank])
    iterator = iter(loader)
    wandb_run = None
    if rank == 0:
        import wandb
        wandb_run = wandb.init(project=c['wandb']['project'], mode='online',
                              name='controller_grounded_3072', group=c['name'], dir=str(run_dir),
                              id=run_id, resume='must' if run_id else None,
                              config={**c, **identity, 'parameters':sum(p.numel() for p in model.parameters())})
        run_id = wandb_run.id
        wandb_run.define_metric('validation/checkpoint_step')
        wandb_run.define_metric('validation/*', step_metric='validation/checkpoint_step')
        save_json(run_dir/'run.json', dict(id=run_id, wandb_url=wandb_run.url, **identity))
    if size > 1:
        value = [run_id]; dist.broadcast_object_list(value, src=0); run_id=value[0]
    encoder, bank = None, None
    began = time.perf_counter(); step = start
    cap = c['controller_grounded']['optimization_gpu_seconds']

    def maximum(seconds):
        value = torch.tensor(seconds, device=device, dtype=torch.float64)
        if size > 1: dist.all_reduce(value, op=dist.ReduceOp.MAX)
        return float(value)

    def save_usage():
        if rank == 0:
            save_json(run_dir/'compute_usage.json', dict(step=step, optimization_gpu_seconds=used,
                optimization_gpu_hours=used/3600, optimization_cap_gpu_seconds=cap,
                validation_gpu_seconds=validation_gpu_seconds, validation_gpu_hours=validation_gpu_seconds/3600,
                gpu_count=size, global_batch=64, validation_steps=validation_steps,
                stop_check='optimizer boundary; final-update overrun reported',
                last_update_overrun_gpu_seconds=max(0.0, used-cap)))

    def snapshot():
        rngs = [None] * size
        if size > 1: dist.all_gather_object(rngs, rng_state())
        else: rngs = [rng_state()]
        value = dict(**identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     step=step, optimization_gpu_seconds=used, validation_gpu_seconds=validation_gpu_seconds,
                     best=best, best_step=best_step, validation_steps=list(validation_steps),
                     rng=rngs, wandb_id=run_id, fixture=False)
        if rank == 0: checkpoint(last, value)
        if size > 1: dist.barrier()
        return value

    try:
        while (step < steps and used < cap) or (step > 0 and step % 5000 == 0 and step not in validation_steps):
            pending_validation = step > 0 and step % 5000 == 0 and step not in validation_steps
            if not pending_validation:
                torch.cuda.synchronize(); started = time.perf_counter()
                step += 1
                module.train(); optimizer.zero_grad(set_to_none=True)
                progress = max((step-1)/steps, used/cap)
                lr_factor = min(1.0, step/1000) * (0.01 + 0.99*(1+math.cos(math.pi*progress))/2)
                for group in optimizer.param_groups: group['lr']=tc['learning_rate']*lr_factor
                weight = 0.1 * min(1.0, max(0.0, (step-1000)/1000))
                item = {k:v.to(device, non_blocking=True) for k,v in next(iterator).items()}
                loss, parts = module(item, world, weight=weight)
                if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss at {step}')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize(); update_seconds = maximum(time.perf_counter()-started)
                used += update_seconds * size
                save_usage()
                if step == 1 or step % 50 == 0:
                    metrics = {'loss':loss.detach(), **parts}
                    names = sorted(metrics); vals = torch.stack([metrics[k] for k in names])
                    if size > 1: dist.all_reduce(vals); vals /= size
                    log = {f'train/{k}':float(v) for k,v in zip(names, vals)}
                    log.update(step=step, **{'train/lr':tc['learning_rate']*lr_factor,
                        'train/execution_weight':weight, 'train/grad_norm':float(norm),
                        'budget/optimization_gpu_hours':used/3600,
                        'system/update_seconds':update_seconds, 'system/update_examples_per_second':64/update_seconds,
                        'system/peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30})
                    if rank == 0:
                        wandb_run.log(log)
                        with (run_dir/'metrics.jsonl').open('a') as f: f.write(json.dumps(log)+'\n')
                        print(json.dumps(log), flush=True)
                if step == 1 or step % 1000 == 0 or used >= cap:
                    state = snapshot()
            evaluate_now = step > 0 and step % 5000 == 0 and step not in validation_steps
            if evaluate_now:
                before_rng = rng_state()
                torch.cuda.synchronize(); eval_start = time.perf_counter()
                if encoder is None:
                    encoder = Encoder(c['encoder'], root, device)
                    bank = RouteBank(args.bank, c, manifest_hash, device)
                model.eval()
                controller = ExecutionController(model, world, c, bank)
                journal = EpisodeJournal(Path(args.bank).parent/'journals'/f'step_{step:07d}',
                    {**identity, 'step':step, 'split':'validation'})
                records, summary = evaluate(model, world, c, root, 'controller_grounded', 3072,
                    'validation', encoder, device, count=8, controller=controller, journal=journal,
                    trajectory_dir=run_dir/'trajectories'/f'step_{step:07d}',
                    distributed_context=(rank,size,device))
                torch.cuda.synchronize(); seconds = maximum(time.perf_counter()-eval_start)
                validation_gpu_seconds += seconds * size
                validation_steps.append(step)
                score = summary['success_macro']; improved = score > best
                if improved: best, best_step = score, step
                restore_rng(before_rng)
                state = snapshot(); save_usage()
                if rank == 0:
                    path = run_dir/'validation'/f'step_{step:07d}.json'
                    save_json(path, dict(**identity, step=step, metrics=summary, records=records,
                        execution=dict(gpu_count=size, seconds=seconds, gpu_hours=seconds*size/3600)))
                    checkpoint(run_dir/'checkpoints'/f'step_{step:07d}.pt',state)
                    if improved: checkpoint(run_dir/'best.pt',state)
                    log = {'validation/'+k:v for k,v in summary.items()}
                    log['validation/checkpoint_step']=step
                    wandb_run.log(log)
                    with (run_dir/'metrics.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
                    print(json.dumps(dict(event='validation_complete',step=step,successes=sum(r['success'] for r in records),metrics=summary)),flush=True)
                if size > 1: dist.barrier()
        snapshot(); save_usage()
        if rank == 0:
            save_json(run_dir/'complete.json',dict(step=step,best=best,best_step=best_step,
                validation_steps=validation_steps,optimization_gpu_hours=used/3600,
                validation_gpu_hours=validation_gpu_seconds/3600,
                wall_seconds=time.perf_counter()-began,
                stop_reason='update_limit' if step==steps else 'compute_cap'))
    finally:
        if wandb_run is not None: wandb_run.finish()
        if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ('config','root','run-dir','world','bank'):p.add_argument('--'+name,required=True)
    train(p.parse_args())
