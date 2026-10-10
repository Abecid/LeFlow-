"""Budgeted LeFlow-TTT training with an exactly reused, charged common adapter."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from ..common import config, digest, file_hash, git_revision, checkpoint, save_json, seed_all, require_execution
from ..models import System
from ..evaluate import evaluate, EpisodeJournal
from ..vision import Encoder
from .data import Segments as BaselineSegments
from .model import AdaptiveLeFlow
from .controller import AdaptiveController
from .common import load_common_adapter


def components(c):
    assert c['primary_method']=='leflow_ttt'
    return AdaptiveLeFlow, AdaptiveController


def load_world(c, path, device):
    if file_hash(path) != c['baseline']['world_sha256']:
        raise ValueError('Frozen world hash changed')
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['method'] != 'world' or state['seed'] != 3072 or state['fixture']:
        raise ValueError('Invalid frozen-world source')
    model = System(c, 'world')
    model.load_state_dict(state['model'])
    return model.world.to(device).eval().requires_grad_(False)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state(state['cuda'])


def train(a):
    # Shared filesystem occasionally returns EAGAIN for a blocking flock.
    # This recorded runtime repair only retries episode-journal identity locks.
    from scripts.operations.baseline_runtime_guard.sitecustomize import install_journal_lock_guard
    install_journal_lock_guard()
    c = config(a.config); method = c['primary_method']
    Model, Controller = components(c)
    root, out = Path(a.root), Path(a.run_dir)
    require_execution(c, root, 'training')
    assert c['seeds'] == [3072] and not c['execution']['test_enabled']
    rank, size, local = (int(os.getenv(k, d)) for k, d in
                        [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')])
    assert size in (1, 4, 8)
    torch.cuda.set_device(local); torch.set_num_threads(2)
    device = torch.device('cuda', local)
    if size > 1:
        dist.init_process_group('nccl', timeout=timedelta(hours=2))
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['protocol'] == digest(c) and not manifest.get('fixture', True)
    out.mkdir(parents=True, exist_ok=True)
    tc, bc = c['training'], c['baseline']
    steps, batch, cap = tc['planner_steps'], tc['global_batch'], bc['optimization_gpu_seconds']
    assert steps == 20000 and batch == 64 and cap == 28800 and batch % size == 0
    adapter_steps = c['leflow_ttt']['common_adapter_steps']
    seed_all(3072)
    model = Model(c).to(device)
    common_adapter=load_common_adapter(model,c)
    world = load_world(c, a.world, device)
    packed = json.loads((Path(bc['packed_cache']) / 'complete.json').read_text())
    if hasattr(model,'action_mean'):
        model.action_mean.copy_(torch.tensor(packed['action_mean'], device=device))
        model.action_std.copy_(torch.tensor(packed['action_std'], device=device))
    identity = dict(code=git_revision(), protocol=digest(c), manifest=file_hash(root / 'manifest.json'),
                    world_hash=file_hash(a.world), packed_entries=packed['entries_sha256'],
                    method=method, seed=3072, world_size=size, common_adapter=common_adapter)
    step, used, validation_seconds = adapter_steps, common_adapter['adapter_optimization_gpu_seconds'], 0.
    best, best_step, eval_steps, run_id = -1., None, [], None
    saved = None; stage = 'planner'
    if (out / 'last.pt').exists():
        saved = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        for k, v in identity.items():
            if saved[k] != v: raise ValueError('Resume identity mismatch: ' + k)
        model.load_state_dict(saved['model'])
        step, used, validation_seconds = saved['step'], saved['optimization_gpu_seconds'], saved['validation_gpu_seconds']
        best, best_step, eval_steps, run_id, stage = (saved[k] for k in
            ['best', 'best_step', 'validation_steps', 'wandb_id', 'stage'])
        ledger = json.loads((out / 'compute_usage.json').read_text())
        used = max(used, ledger['optimization_gpu_seconds'])
        validation_seconds = max(validation_seconds, ledger['validation_gpu_seconds'])
    elif (out / 'run.json').exists():
        raise RuntimeError('Existing run lacks recoverable checkpoint')

    def setup_stage(stage_name, start):
        model.set_stage(False)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                               lr=tc['learning_rate'], weight_decay=tc['weight_decay'])
        wrapped = DDP(model, device_ids=[local], broadcast_buffers=False) if size > 1 else model
        dataset = BaselineSegments(root, c, method, steps * batch, stage=stage_name)
        subset = Subset(dataset, range(start * batch, len(dataset)))
        sampler = DistributedSampler(subset, size, rank, shuffle=False, drop_last=True) if size > 1 else None
        loader = DataLoader(subset, batch_size=batch // size, sampler=sampler, shuffle=False,
                            num_workers=4, prefetch_factor=4, pin_memory=True, persistent_workers=True,
                            drop_last=True, generator=torch.Generator().manual_seed(3072))
        return opt, wrapped, loader, iter(loader)

    optimizer, module, loader, iterator = setup_stage(stage, step)
    seed_all(3072 + 997 * rank)
    if saved:
        optimizer.load_state_dict(saved['optimizer']); restore_rng(saved['rng'][rank])
    wandb_run = None
    if rank == 0:
        import wandb
        wandb_run = wandb.init(project=c['wandb']['project'], name=method + '_3072',
            mode='online', group=c['name'], dir=str(out), id=run_id,
            resume='must' if run_id else None, config={**c, **identity,
                'parameters': sum(p.numel() for p in model.parameters()), 'new_optimizer_updates': steps-adapter_steps})
        run_id = wandb_run.id
        wandb_run.define_metric('validation/checkpoint_step')
        wandb_run.define_metric('validation/*', step_metric='validation/checkpoint_step')
        save_json(out / 'run.json', dict(**identity, id=run_id, wandb_url=wandb_run.url))
    if size > 1:
        value = [run_id]; dist.broadcast_object_list(value, src=0); run_id = value[0]
    encoder = None

    def maximum(seconds):
        v = torch.tensor(seconds, device=device, dtype=torch.float64)
        if size > 1: dist.all_reduce(v, op=dist.ReduceOp.MAX)
        return float(v)

    def usage():
        if rank == 0:
            save_json(out / 'compute_usage.json', dict(step=step, stage=stage,
                optimization_gpu_seconds=used, optimization_gpu_hours=used / 3600,
                optimization_cap_gpu_seconds=cap, validation_gpu_seconds=validation_seconds,
                validation_gpu_hours=validation_seconds / 3600, validation_steps=eval_steps,
                gpu_count=size, global_batch=batch, last_update_overrun_gpu_seconds=max(0., used-cap),
                adapter_optimization_included=True, common_adapter=common_adapter))

    def snapshot(commit_last=True):
        rngs = [None] * size
        if size > 1: dist.all_gather_object(rngs, rng_state())
        else: rngs = [rng_state()]
        state = dict(**identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
            step=step, stage=stage, optimization_gpu_seconds=used,
            validation_gpu_seconds=validation_seconds, best=best, best_step=best_step,
            validation_steps=list(eval_steps), rng=rngs, wandb_id=run_id, fixture=False)
        if rank == 0 and commit_last: checkpoint(out / 'last.pt', state)
        if size > 1: dist.barrier()
        return state

    def log(values):
        if rank == 0:
            wandb_run.log(values)
            with (out / 'metrics.jsonl').open('a') as f: f.write(json.dumps(values) + '\n')
            print(json.dumps(values), flush=True)

    def heldout_loss():
        before = rng_state(); seed_all(80317 + rank)
        model.inner_steps=c['leflow_ttt']['inner_steps']
        ds = BaselineSegments(root, c, method, 256, split='validation', stage=stage)
        dl = DataLoader(Subset(ds, list(range(rank, len(ds), size))), batch_size=batch // size,
                        num_workers=0)
        totals, count = {}, 0
        model.eval()
        with torch.no_grad():
            for item in dl:
                item = {k: v.to(device) for k, v in item.items()}
                loss, parts = model(item, world)
                n = len(item['z']); count += n
                for k, v in dict(loss=loss, **parts).items():
                    totals[k] = totals.get(k, 0.) + float(v) * n
        names = sorted(totals)
        values = torch.tensor([totals[k] for k in names] + [count], device=device, dtype=torch.float64)
        if size > 1: dist.all_reduce(values)
        restore_rng(before)
        return {f'heldout/{stage}/{k}': float(v / values[-1]) for k, v in zip(names, values[:-1])}

    def pending_evaluation():
        return (step > 0 and stage == 'planner' and len(eval_steps) < 4
                and step not in eval_steps
                and max(step / steps, used / cap) >= (len(eval_steps) + 1) / 4)

    def validate_checkpoint():
        nonlocal encoder, validation_seconds, best, best_step
        before = rng_state(); torch.cuda.synchronize(); started = time.perf_counter()
        model.eval()
        if encoder is None: encoder = Encoder(c['encoder'], root, device)
        model.inner_steps=c['leflow_ttt']['inner_steps']
        controller = Controller(model, world, c)
        journal = EpisodeJournal(out / 'journals' / f'step_{step:07d}',
                                 {**identity, 'step': step, 'split': 'validation'})
        records, summary = evaluate(model, world, c, root, method, 3072, 'validation',
            encoder, device, count=8, controller=controller, journal=journal,
            trajectory_dir=out / 'trajectories' / f'step_{step:07d}',
            distributed_context=(rank, size, device))
        torch.cuda.synchronize(); seconds = maximum(time.perf_counter()-started)
        validation_seconds += seconds * size; eval_steps.append(step)
        score = summary['success_macro']; improved = score > best
        if improved: best, best_step = score, step
        restore_rng(before); state = snapshot(commit_last=False)
        if rank == 0:
            save_json(out / 'validation' / f'step_{step:07d}.json', dict(**identity,
                step=step, metrics=summary, records=records,
                execution=dict(gpu_count=size, seconds=seconds, gpu_hours=seconds*size/3600)))
            checkpoint(out / 'checkpoints' / f'step_{step:07d}.pt', state)
            if improved: checkpoint(out / 'best.pt', state)
            # Commit resume state only after the report and selected checkpoint
            # are durable. A crash earlier safely reuses the episode journal.
            checkpoint(out / 'last.pt', state)
        if size > 1: dist.barrier()
        usage()
        log({**{'validation/' + k: v for k, v in summary.items()}, 'validation/checkpoint_step': step})

    began = time.perf_counter()
    try:
        while (step < steps and used < cap) or pending_evaluation():
            if pending_evaluation():
                validate_checkpoint()
                continue
            if stage == 'adapter' and step >= adapter_steps:
                del module, iterator, loader
                stage = 'planner'
                optimizer, module, loader, iterator = setup_stage(stage, step)
                snapshot()
            torch.cuda.synchronize(); started = time.perf_counter()
            step += 1; model.train(); optimizer.zero_grad(set_to_none=True)
            schedule=c['leflow_ttt']['train_inner_steps']
            model.inner_steps=schedule[(step-adapter_steps-1)%len(schedule)]
            item = {k: v.to(device, non_blocking=True) for k, v in next(iterator).items()}
            loss, parts = module(item, world)
            if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss at {step}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                tc.get('gradient_clip') or float('inf'),error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize(); seconds = maximum(time.perf_counter() - started)
            used += seconds * size; usage()
            if step == adapter_steps+1 or step % 50 == 0:
                metrics = dict(loss=loss.detach(), **parts)
                names = sorted(metrics); values = torch.stack([metrics[k] for k in names])
                if size > 1: dist.all_reduce(values); values /= size
                log({**{f'train/{stage}/{k}': float(v) for k, v in zip(names, values)},
                    'step': step, 'train/lr': tc['learning_rate'], 'train/grad_norm': float(norm),
                    'budget/optimization_gpu_hours': used / 3600, 'system/update_seconds': seconds,
                    'system/update_examples_per_second': batch / seconds,
                    'system/peak_gpu_gib': torch.cuda.max_memory_allocated() / 2**30})
            if step % 1000 == 0 or step == adapter_steps+1 or used >= cap:
                snapshot()
                torch.cuda.synchronize(); loss_started = time.perf_counter()
                diagnostics = heldout_loss()
                torch.cuda.synchronize(); validation_seconds += maximum(time.perf_counter()-loss_started) * size
                log(dict(step=step, **diagnostics)); usage()
        snapshot(); usage()
        if stage != 'planner' or len(eval_steps) != 4:
            raise RuntimeError('Baseline budget ended before all four registered validations')
        if rank == 0:
            save_json(out / 'complete.json', dict(step=step, stage=stage, best=best, best_step=best_step,
                validation_steps=eval_steps, optimization_gpu_hours=used/3600,
                validation_gpu_hours=validation_seconds/3600, wall_seconds=time.perf_counter()-began,
                stop_reason='update_limit' if step == steps else 'compute_cap',
                convergence_established=False,common_adapter=common_adapter,new_optimizer_updates=step-adapter_steps))
    except BaseException:
        # NCCL teardown can wait forever if peers are still evaluating. Report
        # the original error first and let torchrun terminate sibling workers.
        import sys
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(1)
    finally:
        if wandb_run is not None: wandb_run.finish()
        if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for k in ('config', 'root', 'run-dir', 'world'): p.add_argument('--' + k, required=True)
    train(p.parse_args())
