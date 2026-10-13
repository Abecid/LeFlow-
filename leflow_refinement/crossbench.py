"""Evaluate frozen refinement against the existing shared-execution baselines.

No checkpoint training or baseline code changes. The old training identity is
verified, and additional inference source is recorded separately.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from .core import PlannerOps, RefinementConfig, sample, select
from .runtime import variants


class RefinementController:
    def __init__(self, model, world, spec, cfg):
        if spec.method != 'leflow' or cfg.horizon != spec.horizon:
            raise ValueError('Refinement requires a matching LeFlow checkpoint')
        self.model = model.eval().requires_grad_(False)
        self.world, self.s, self.cfg = world, spec, cfg
        self.begin_episode()

    def begin_episode(self):
        self.trace = []
        self.decisions = 0

    def observe(self, actual):
        pass  # Frozen inference only; no TTT or hidden adaptation.

    def plan(self, z, goal, *, budget=None):
        if z.shape[1] != 1 or goal.shape != z.shape:
            raise ValueError('Compact [B,1,D] LeWM features required')
        def decode(paths):
            raw = self.model.inverse(paths[:, :-1].flatten(0, 1), paths[:, 1:].flatten(0, 1))
            raw = raw.reshape(len(paths), self.s.horizon, self.s.ad)
            return self.world.from_world(raw)
        def rollout(start, actions):
            pred = self.world.rollout(start[:, None], actions, budget=budget)[:, :, 0]
            return torch.cat((start[:, None], pred), 1)
        ops = PlannerOps(self.model.flow, decode, rollout, budget.check if budget else None)
        # Match legacy LeFlow's GLOBAL initial-noise stream exactly. Dedicated
        # perturbation stream cannot consume the next decision's initial pool.
        perturb = torch.Generator(device=z.device).manual_seed(torch.initial_seed()+1000000007+self.decisions)
        paths = sample(z[:, 0], goal[:, 0], ops, self.cfg, perturb_generator=perturb)
        out = select(paths, ops)
        self.trace.append(dict(**ops.compute.as_dict(), local_cost=float(out['costs'][0]),
                               selected_index=int(out['selected_indices'][0]),
                               refinement=asdict(self.cfg)))
        self.decisions += 1
        return out['actions'][0, 0]


def source_hash():
    from flow_jepa.common import file_hash, digest
    root = Path(__file__).parent
    return digest({p.name: file_hash(p) for p in sorted(root.glob('*.py'))})


def evaluate(cache, checkpoint, output, cfg, *, device='cuda:0'):
    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np
    from flow_jepa.common import file_hash, save_json, digest
    from crossbench.spec import Spec
    from crossbench.models import build
    from crossbench.world import load_world
    from crossbench.train import code_hash
    from crossbench.evaluate import cases, episode, summarize
    from crossbench.environment import NativeEnv
    root, out = Path(cache), Path(output)
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU substitution')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    identity = payload['identity']
    spec = Spec(**identity['spec'])
    meta = json.loads((root/'manifest.json').read_text())
    if identity.get('fixture') or meta.get('fixture'):
        raise ValueError('Synthetic fixtures cannot produce benchmark scores')
    if identity['source'] != code_hash():
        raise ValueError('Baseline runtime changed since training')
    if (identity['cache'] != file_hash(root/'cache.h5') or
            identity['world'] != meta['checkpoint_sha256']):
        raise ValueError('Training cache/world identity mismatch')
    model = build(spec).to(device).eval().requires_grad_(False)
    model.load_state_dict(payload['model'], strict=True)
    world = load_world(meta, device)
    plan = cases(meta, spec)
    run_id = dict(checkpoint=file_hash(checkpoint), cache=meta['cache_sha256'],
        world=meta['checkpoint_sha256'], baseline_source=identity['source'],
        refinement_source=source_hash(), specification=asdict(spec),
        refinement=asdict(cfg), case_hash=digest(plan), device=str(device),
        protocol='crossbench-shared-execution-refinement-v1')
    # Canonical JSON representation avoids tuple/list resume mismatches.
    run_id = json.loads(json.dumps(run_id))
    out.mkdir(parents=True, exist_ok=True)
    ip = out/'identity.json'
    if ip.exists() and json.loads(ip.read_text()) != run_id:
        raise ValueError('Refusing to overwrite a different evaluation')
    save_json(ip, run_id)
    controller = RefinementController(model, world, spec, cfg)
    records = []
    env = NativeEnv(spec.dataset, spec.action_budget)
    try:
        with h5py.File(meta['source_path'], 'r') as src:
            for idx, row in enumerate(plan):
                path = out/'cases'/f'{idx:05d}.json'
                if path.exists():
                    record = json.loads(path.read_text())
                    if record.get('identity') != run_id or record['id'] != row['id']:
                        raise ValueError('Resumed case identity mismatch')
                else:
                    record, images = episode(env, world, controller, src, row, spec, device)
                    record['identity'] = run_id
                    image_path = out/'trajectories'/f'{idx:05d}.npz'
                    image_path.parent.mkdir(exist_ok=True)
                    np.savez_compressed(image_path, **images)
                    save_json(path, record)
                records.append(record)
                save_json(out/'progress.json', dict(completed=len(records), expected=len(plan),
                                                     metrics=summarize(records)))
    finally:
        env.close()
    if len(records) != len(plan) or len({r['id'] for r in records}) != len(plan):
        raise ValueError('Incomplete/duplicate evaluations')
    report = dict(identity=run_id, metrics=summarize(records), records=records)
    save_json(out/'report.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'checkpoint', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--variant', choices=tuple(variants()), default='world_pnp')
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    print(json.dumps(evaluate(a.cache, a.checkpoint, a.output, variants()[a.variant],
                             device=a.device)['metrics'], indent=2))


if __name__ == '__main__':
    main()
