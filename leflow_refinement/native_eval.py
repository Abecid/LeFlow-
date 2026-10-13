"""Resumable paired evaluation with the original SWM dataset-evaluation protocol.

No optimizer is created. Every variant reuses the SAME native LeFlow checkpoint.
This native 25-action execution protocol, restricted to validation episodes, is separate from crossbench's 5-action
execution protocol. Never aggregate their scores into one comparison table.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from .runtime import NativeRuntime, variants, sha256, synchronize


def jsonable(value):
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [jsonable(v) for v in value]
    return value


def case_plan(source, count, seed, offset, allowed_episodes):
    """Native frame-weighted sampling restricted to registered validation episodes."""
    ep = np.asarray(source['episode_idx' if 'episode_idx' in source else 'ep_idx'])
    step = np.asarray(source['step_idx'])
    if ep.ndim != 1 or step.shape != ep.shape or offset < 0:
        raise ValueError('Invalid episode/step columns or goal offset')
    unique, inverse = np.unique(ep, return_inverse=True)
    lengths = np.zeros(len(unique), dtype=np.int64)
    np.maximum.at(lengths, inverse, step.astype(np.int64)+1)
    if not allowed_episodes:
        raise ValueError('An explicit nonempty validation episode split is required')
    valid = np.flatnonzero((step <= lengths[inverse]-offset-1)
                           & np.isin(ep, list(allowed_episodes)))
    if count < 1 or len(valid) < count:
        raise ValueError('Insufficient native starting points')
    indices = np.sort(np.random.default_rng(seed).choice(valid, size=count, replace=False))
    return [dict(id=f'{int(ep[i])}/{int(step[i])}', episode=int(ep[i]),
                 step=int(step[i]), frame=int(i)) for i in indices]


class Solver:
    def __init__(self, engine, cfg, seed):
        self.engine, self.cfg, self.seed = engine, cfg, seed
        self.trace = []
        self.decisions = 0
        self.rollout_count = 0

    def configure(self, *, action_space, n_envs, config):
        self._n_envs, self._config = n_envs, config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        if config.horizon != self.cfg.horizon or config.action_block != self.engine.model.action_block:
            raise ValueError('Planner checkpoint and execution configuration disagree')

    @property
    def n_envs(self): return self._n_envs
    @property
    def action_dim(self): return self._action_dim*self._config.action_block
    @property
    def horizon(self): return self.cfg.horizon

    def __call__(self, *args, **kwargs): return self.solve(*args, **kwargs)

    def solve(self, info_dict, init_action=None):
        synchronize(self.engine.device); started = time.perf_counter()
        z, g = self.engine.model.encode_current_and_goal(info_dict)
        result = self.engine.plan(z, g, self.cfg, seed=self.seed+self.decisions*100003)
        synchronize(self.engine.device)
        telemetry = result['telemetry']
        telemetry['encoding_and_planning_seconds'] = time.perf_counter()-started
        telemetry['selected_cost'] = float(result['costs'][0])
        self.trace.append(telemetry); self.decisions += 1
        self.rollout_count += telemetry['world_candidate_transitions']//self.cfg.horizon
        return dict(actions=result['actions'].detach().cpu(),
                    costs=result['costs'].detach().cpu().tolist(),
                    goal_costs=result['costs'].detach().cpu(),rollout_count=self.rollout_count)


def run(args):
    os.environ.setdefault('MUJOCO_GL', 'egl' if args.device.startswith('cuda') else 'osmesa')
    import h5py
    import hdf5plugin  # noqa: F401
    import stable_worldmodel as swm
    from omegaconf import OmegaConf
    from sklearn.preprocessing import StandardScaler
    from btm_jepa.env import World
    from btm_jepa.data import image_transform
    from crossbench.environment import verify_install
    from flow_jepa.common import save_json, digest
    from .crossbench import source_hash
    verify_install()
    torch.set_num_threads(args.cpu_threads)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA device is unavailable')
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1]/'config/eval'/f'{args.task}.yaml')
    cfg.eval.num_eval = 1
    cfg.world.max_episode_steps = 2*cfg.eval.eval_budget
    source_path = Path(args.source).resolve(strict=True)
    out = Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=True)
    cache = out/'dataset-link'
    target = cache/'datasets'/(str(cfg.eval.dataset_name)+'.h5')
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if target.resolve() != source_path: raise ValueError('Dataset link points elsewhere')
    else: target.symlink_to(source_path)
    split_payload=json.loads(Path(args.split_file).read_text())
    split=split_payload.get('split',split_payload)
    allowed=split.get('val_episode_ids')
    if allowed is None:
        raise ValueError('Provide a prepared source.json with explicit val_episode_ids; sealed tests are excluded')
    if set(allowed) & set(split.get('test_episode_ids',[])):
        raise ValueError('Validation and sealed test episodes overlap')
    with h5py.File(source_path,'r') as src:
        plan = case_plan(src,args.cases,args.seed,int(cfg.eval.goal_offset_steps),allowed)
        processors = {}
        for col in cfg.dataset.keys_to_cache:
            if col == 'pixels': continue
            processor = StandardScaler()
            for start in range(0,len(src[col]),65536):
                values = np.asarray(src[col][start:start+65536])
                values = values[np.isfinite(values).all(axis=1)]
                if len(values): processor.partial_fit(values)
            if not hasattr(processor,'scale_'): raise ValueError('No finite normalization data')
            processors[col] = processor
            if col != 'action': processors['goal_'+col] = processor
    dataset = swm.data.HDF5Dataset(str(cfg.eval.dataset_name),
        keys_to_cache=list(cfg.dataset.keys_to_cache),cache_dir=cache)
    source_identity = dict(path=str(source_path),bytes=source_path.stat().st_size,sha256=sha256(source_path))
    transform = dict(pixels=image_transform(),goal=image_transform())
    world = World(**OmegaConf.to_container(cfg.world,resolve=True),image_shape=(224,224))
    try:
        engine = NativeRuntime.load(args.checkpoint,world_checkpoint=args.world_checkpoint,device=args.device)
        ac = processors['action']; block = engine.model.action_block
        space = world.envs.envs[0].action_space
        lower = np.tile((space.low-ac.mean_)/ac.scale_,block)
        upper = np.tile((space.high-ac.mean_)/ac.scale_,block)
        if not args.legacy_unbounded:
            engine.lower=torch.as_tensor(lower,device=engine.device,dtype=torch.float32)
            engine.upper=torch.as_tensor(upper,device=engine.device,dtype=torch.float32)
            engine.provenance['action_clipping']='normalized_actuator_bounds'
        names = list(variants()) if args.variant == 'all' else [args.variant]
        reports=[]
        for name in names:
            refinement = variants()[name]
            identity = jsonable(dict(protocol='native-swm-validation-refinement-v1',task=args.task,
                split_file_sha256=sha256(args.split_file),validation_episode_ids=allowed,
                native_execution=OmegaConf.to_container(cfg.plan_config,resolve=True),
                action_budget=int(cfg.eval.eval_budget),goal_offset=int(cfg.eval.goal_offset_steps),
                source=source_identity,model=engine.provenance,refinement=asdict(refinement),
                refinement_source=source_hash(),cases=plan,seed=args.seed,
                normalized_action_bounds=None if args.legacy_unbounded else [lower.tolist(),upper.tolist()],
                normalization='full-source StandardScaler ddof=0; native eval convention',
                device=args.device,cpu_threads=args.cpu_threads))
            dest=out/name;dest.mkdir(exist_ok=True)
            ip=dest/'identity.json'
            if ip.exists() and json.loads(ip.read_text())!=identity:
                raise ValueError('Different identity in existing evaluation directory')
            save_json(ip,identity)
            records=[]
            for i,row in enumerate(plan):
                cp=dest/'cases'/f'{i:05d}.json'
                if cp.exists():
                    record=json.loads(cp.read_text())
                    if record['identity_hash']!=digest(identity) or record['case']!=row:
                        raise ValueError('Resumed native case mismatch')
                else:
                    solver=Solver(engine,refinement,args.seed+i*10000019)
                    policy=swm.policy.WorldModelPolicy(solver=solver,
                        config=swm.PlanConfig(**cfg.plan_config),process=processors,transform=transform)
                    world.set_policy(policy)
                    metrics=world.evaluate_from_dataset(dataset=dataset,
                        episodes_idx=[row['episode']],start_steps=[row['step']],
                        goal_offset_steps=int(cfg.eval.goal_offset_steps),eval_budget=int(cfg.eval.eval_budget),
                        callables=OmegaConf.to_container(cfg.eval.callables,resolve=True),
                        save_video=args.video,video_path=str(dest/'videos'/f'{i:05d}'))
                    record=jsonable(dict(case=row,identity_hash=digest(identity),metrics=metrics,
                        success=bool(metrics['episode_successes'][0]),trace=solver.trace))
                    save_json(cp,record)
                records.append(record)
                print(json.dumps(dict(task=args.task,variant=name,completed=len(records),
                    expected=len(plan),successes=sum(r['success'] for r in records))),flush=True)
            latency=[t['encoding_and_planning_seconds']*1000 for r in records for t in r['trace']]
            wins=sum(r['success'] for r in records)
            report=dict(identity=identity,metrics=dict(successes=wins,episodes=len(records),
                success_rate=wins/len(records),mean_decision_ms=float(np.mean(latency)),
                p95_decision_ms=float(np.percentile(latency,95))),records=records)
            save_json(dest/'report.json',report);reports.append(report)
        save_json(out/'comparison.json',[dict(variant=n,**r['metrics']) for n,r in zip(names,reports)])
        return reports
    finally:
        world.envs.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=('pusht','cube','tworoom','reacher'),required=True)
    for k in ('source','split-file','checkpoint','world-checkpoint','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--variant',choices=('all',*variants()),default='all')
    p.add_argument('--cases',type=int,default=50);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda:0');p.add_argument('--cpu-threads',type=int,default=1)
    p.add_argument('--legacy-unbounded',action='store_true',help='Explicit native legacy unclipped scoring control')
    p.add_argument('--video',action='store_true')
    run(p.parse_args())


if __name__=='__main__':main()
