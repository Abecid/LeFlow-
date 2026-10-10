"""Bounded training-data checks before freezing and launching a baseline."""
import argparse
import json
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..common import config, save_json, seed_all, distributed, digest
from ..vision import Encoder
from ..budget import PlanningBudget
from .data import BaselineSegments
from .train import components, load_world


def main(a):
    began = time.perf_counter()
    rank, size, device = distributed()
    torch.set_num_threads(2); seed_all(3072)
    c = config(a.config); method = c['primary_method']
    Model, Controller = components(c)
    model = Model(c).to(device); world = load_world(c, a.world, device)
    before = {k: v.cpu().clone() for k, v in world.state_dict().items()}
    meta = json.loads((Path(c['baseline']['packed_cache']) / 'complete.json').read_text())
    if hasattr(model,'action_mean'):
        model.action_mean.copy_(torch.tensor(meta['action_mean'], device=device))
        model.action_std.copy_(torch.tensor(meta['action_std'], device=device))
    details = []
    for stage in (['adapter', 'planner'] if method == 'leflow_release' else ['planner']):
        if method == 'leflow_release': model.set_stage(stage == 'adapter')
        ds = BaselineSegments(a.root, c, method, 64, stage=stage)
        item = next(iter(DataLoader(torch.utils.data.Subset(ds,range(rank,64,size)), batch_size=64//size)))
        item = {k: v.to(device) for k, v in item.items()}
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        module = DDP(model, device_ids=[device.index], broadcast_buffers=False) if size > 1 else model
        for i in range(2):
            model.train(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); started = time.perf_counter()
            loss, metrics = module(item, world)
            assert torch.isfinite(loss)
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.requires_grad]
            assert all(g is not None and torch.isfinite(g).all() for g in grads)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step(); torch.cuda.synchronize()
            details.append(dict(stage=stage, update=i, loss=float(loss),
                seconds=time.perf_counter()-started, metrics={k: float(v) for k, v in metrics.items()}))
        del module
    assert all(p.grad is None for p in world.parameters())
    assert all(torch.equal(v.cpu(), before[k]) for k, v in world.state_dict().items())
    model.eval()
    manifest = json.loads((Path(a.root) / 'manifest.json').read_text())
    row = next(r for r in manifest['entries'] if r['split'] == 'train' and r['mode'] == 'expert' and r['expert_success'])
    with h5py.File(Path(a.root) / row['path'], 'r') as f:
        rgb, goal = f['initial_rgb'][:], f['goal'][:]
    from ..environment import make_env
    env = make_env(row['task'], row['seed'], c['data'])
    try:
        env.reset(); assert np.array_equal(env.render(), rgb), 'TRAIN reset RGB mismatch'
    finally: env.close()
    encoder = Encoder(c['encoder'], a.root, device)
    mean, std = (torch.tensor(manifest[k], device=device) for k in ('mean', 'std'))
    goal = (torch.tensor(goal, device=device, dtype=torch.float32) - mean) / std
    controller = Controller(model, world, c)
    if hasattr(controller, 'begin_episode'): controller.begin_episode()
    timings = []
    for _ in range(4):
        torch.cuda.synchronize(); started = time.perf_counter()
        budget = PlanningBudget(30.)
        z = (encoder(np.repeat(rgb[None, None], 16, axis=1)).float()-mean)/std
        action, _, score, _ = controller.plan(z, goal[None], budget=budget)
        torch.cuda.synchronize(); timings.append(time.perf_counter()-started)
        assert action.shape == (8 * getattr(controller, 'execution_blocks', 1),)
        assert torch.isfinite(action).all() and np.isfinite(score)
    elapsed = torch.tensor(time.perf_counter()-began,device=device)
    if size > 1: dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
    if rank == 0: save_json(a.output, dict(method=method, configuration=c, protocol=digest(c), training_checks=details,
        world_unchanged=True, finite_gradients=True, native_full_dimension=512 if method=='leflow_release' else None,
        controller_seconds=timings, controller_warm_mean_ms=1000*float(np.mean(timings[1:])),
        peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
        gpu_hours=float(elapsed)*size/3600, gpu_count=size, train_reset_rgb_exact=True,
        data_case=row['id'], simulator_calls=0, test_files_read=0,
        weights_discarded=True))
    if rank == 0: print(Path(a.output).read_text(), flush=True)
    if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for k in ('config', 'root', 'world', 'output'): p.add_argument('--'+k, required=True)
    main(p.parse_args())
