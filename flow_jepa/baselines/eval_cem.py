"""One fixed validation of released CEM; it has no extra learned head."""
import argparse
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

from ..common import config, digest, distributed, file_hash, git_revision, save_json, seed_all
from ..evaluate import evaluate, EpisodeJournal
from ..vision import Encoder
from .train import load_world
from .cem import ReleasedCEMController, verify_solver


def main(a):
    c = config(a.config)
    assert c['primary_method'] == 'cem_release' and not c['execution']['test_enabled']
    rank, size, device = distributed(); torch.set_num_threads(2); seed_all(3072)
    out, root = Path(a.run_dir), Path(a.root); out.mkdir(parents=True, exist_ok=True)
    identity = dict(code=git_revision(), protocol=digest(c), manifest=file_hash(root/'manifest.json'),
        world_hash=file_hash(a.world), method='cem_release', seed=3072, split='validation',
        solver=verify_solver())
    manifest = json.loads((root/'manifest.json').read_text())
    assert not manifest['fixture'] and manifest['protocol'] == identity['protocol']
    if (out/'validation.json').exists():
        old = json.loads((out/'validation.json').read_text())
        assert all(old[k] == v for k,v in identity.items())
        if old.get('wandb_synced'): return
    torch.cuda.synchronize(); started = time.perf_counter()
    world = load_world(c, a.world, device)
    controller = ReleasedCEMController(None, world, c)
    encoder = Encoder(c['encoder'], root, device)
    journal = EpisodeJournal(out/'journal', identity)
    records, summary = evaluate(None, world, c, root, 'cem_release', 3072, 'validation',
        encoder, device, count=8, controller=controller, journal=journal,
        trajectory_dir=out/'trajectories', distributed_context=(rank,size,device))
    torch.cuda.synchronize(); seconds = torch.tensor(time.perf_counter()-started,device=device)
    if dist.is_initialized(): dist.all_reduce(seconds,op=dist.ReduceOp.MAX)
    if rank == 0:
        report = dict(**identity, metrics=summary, records=records, fixture=False,
            optimization_gpu_hours=0., validation_gpu_hours=float(seconds)*size/3600,
            fixed_world_reused=True, wandb_synced=False)
        save_json(out/'validation.json',report)
        import wandb
        run = wandb.init(project=c['wandb']['project'],name='cem_release_3072',
            group=c['name'],mode='online',dir=str(out),config={**c,**identity})
        run.log({'validation/'+k:v for k,v in summary.items()})
        report.update(wandb_id=run.id,wandb_url=run.url)
        run.finish(); report['wandb_synced']=True
        save_json(out/'validation.json',report); save_json(out/'complete.json',report)
    if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for k in ('config','root','world','run-dir'): p.add_argument('--'+k,required=True)
    main(p.parse_args())
