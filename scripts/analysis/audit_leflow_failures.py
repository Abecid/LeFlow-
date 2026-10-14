"""Read-only forensic audit of existing LeFlow pilot evidence.

No training, simulation, or checkpoint selection is performed. The gradient
probe uses the exact current LeFlowPolicy.loss statements with a small synthetic
frozen world; it establishes graph connectivity, NOT physical model quality.
"""
from __future__ import annotations
import argparse
import ast
import copy
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import sys
import textwrap
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read_json(path: Path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def median(values):
    return statistics.median(values) if values else None


def audit_reports(root: Path) -> list[dict]:
    output = []
    for campaign in ('20261014-self-refine', '20261014-noise-cem'):
        for path in sorted((root/'docs/reports'/campaign/'jobs').glob('*/report.json.gz')):
            report = read_json(path)
            records = report['records']
            if len(records) != 50 or len({r['id'] for r in records}) != 50:
                raise ValueError(f'Incomplete/duplicate cases: {path}')
            if any(not isinstance(r['success'], bool) for r in records):
                raise ValueError(f'Nonboolean success: {path}')
            successes = sum(r['success'] for r in records)
            if successes != report['metrics']['successes']:
                raise ValueError(f'Aggregate mismatch: {path}')
            if report['frozen_before'] != report['frozen_after']:
                raise ValueError(f'Changed frozen state: {path}')
            trace = [t for r in records for t in r['trace']]
            output.append(dict(
                path=str(path.relative_to(root)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                variant=path.parent.name,
                successes=successes, episodes=len(records),
                checkpoint_step=report['identity']['checkpoint_step'],
                checkpoint_sha256=report['identity']['checkpoint_sha256'],
                timeouts=sum(r['timeout'] for r in records),
                failure_reasons={reason:sum(r['failure_reason']==reason for r in records)
                                 for reason in sorted({r['failure_reason'] for r in records})},
                mean_decision_ms=statistics.mean(v for r in records for v in r['decision_latency_ms']),
                median_selected_terminal_mse=median([t['local_cost'] for t in trace if 'local_cost' in t]),
                median_observed_prefix_prediction_mse=median([t['observed_prefix_prediction_mse']
                    for t in trace if 'observed_prefix_prediction_mse' in t]),
                optimizer_updates=report['optimizer_updates'],
            ))
    if len(output) != 24:
        raise ValueError(f'Expected 24 existing inference reports, found {len(output)}')
    return output


def audit_training(root: Path) -> list[dict]:
    output = []
    for task in ('pusht', 'cube'):
        base = root/'docs/reports/20261013-crossbench-baselines/runs'/task/'leflow'
        identity, best, complete = [read_json(base/f'{name}.json') for name in ('identity','best','complete')]
        records=[json.loads(line) for line in gzip.decompress((base/'metrics.jsonl.gz').read_bytes()).decode().splitlines()]
        spec=identity['spec']; count=identity['window_sampling']['eligible_starts']
        windows=best['step']*spec['global_batch']
        output.append(dict(
            dataset=task, selected_step=best['step'], final_step=complete['step'],
            selected_successes=best['metrics']['successes'], eligible_starts=count,
            selected_training_draws=windows,
            selected_draws_per_eligible_start=windows/count,
            final_training_draws=complete['step']*spec['global_batch'],
            final_draws_per_eligible_start=complete['step']*spec['global_batch']/count,
            interpretation='Draw-count ratio, not distinct coverage or an epoch: sampling is episode-uniform with replacement.',
            optimization_gpu_hours=complete['optimization_gpu_hours'],
            milestones=[r for r in records if r['step'] in (5000,10000,15000,20000)],
            last_100_logs_median={key:median([r[key] for r in records[-100:]])
                for key in ('flow_loss','inverse_loss','consistency')},
        ))
    return output


def audit_consistency_graph() -> dict:
    import torch
    from torch import nn
    import crossbench.models as models
    from crossbench.spec import Spec
    torch.set_num_threads(1)
    torch.manual_seed(9041)
    spec=Spec('pusht','leflow')
    model=models.LeFlowPolicy(spec).eval()

    class FrozenProbeWorld(nn.Module):
        def __init__(self):
            super().__init__()
            self.map=nn.Linear(spec.ad,spec.dim,bias=False).requires_grad_(False)
        def to_world(self,a): return a
        def from_world(self,a): return a.clamp(-1,1)
        def forward(self,z,a): return z+0.1*self.map(a)[:,None]

    world=FrozenProbeWorld().eval()
    source=textwrap.dedent(inspect.getsource(models.LeFlowPolicy.loss))
    tree=ast.parse(source); fn=tree.body[0]
    if not isinstance(fn,ast.FunctionDef) or not isinstance(fn.body[-1],ast.Return):
        raise ValueError('Unexpected loss layout; inspect the current method before auditing')
    # Expose the three existing graph tensors; do not rewrite their computations.
    fn.name='exposed_components'
    fn.body[-1].value=ast.Tuple(elts=[ast.Name(id=n,ctx=ast.Load()) for n in
                                    ('fm','inverse','consistency')],ctx=ast.Load())
    ns=dict(vars(models));exec(compile(ast.fix_missing_locations(tree),'<exact-loss-graph-probe>','exec'),ns)
    probe=types.MethodType(ns['exposed_components'],model)
    batch={'z':torch.randn(2,6,1,spec.dim), 'a':0.1*torch.randn(2,5,spec.ad)}
    fm, inverse, consistency=probe(batch,world)
    flow_params=tuple(model.flow.parameters()); inverse_params=tuple(model.inverse.parameters())
    gradients=torch.autograd.grad(consistency,flow_params+inverse_params,allow_unused=True)
    nf=len(flow_params);fg,ig=gradients[:nf],gradients[nf:]
    return dict(
        source='crossbench/models.py::LeFlowPolicy.loss',
        source_sha256=hashlib.sha256((ROOT/'crossbench/models.py').read_bytes()).hexdigest(),
        scope='Exact loss graph, synthetic frozen dynamics; no learned checkpoint or physical trial.',
        flow_parameter_tensors=len(fg),
        flow_tensors_connected_to_consistency=sum(g is not None for g in fg),
        inverse_parameter_tensors=len(ig),
        inverse_tensors_connected_to_consistency=sum(g is not None for g in ig),
        inverse_nonzero_consistency_gradient_tensors=sum(g is not None and bool(g.abs().sum()) for g in ig),
        frozen_world_has_parameter_gradients=any(p.grad is not None for p in world.parameters()),
        forward_context_in_consistency='One encoded state per independent transition, rather than the three-state history used by inference after warmup.',
        conclusion='This loss regularizes the inverse decoder on recorded paths. It does not train the flow generator with dynamics feedback.',
    )


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=dict(
        audited_commit='9f2e440f0b0fe2e6e5783a6c84ca2ef1f5c34dc4',
        optimizer_updates=0, simulator_steps=0,
        reports=audit_reports(args.evidence_root), training=audit_training(args.evidence_root),
        consistency_graph=audit_consistency_graph(),
        limitations=[
            'No dataset/checkpoint replay: trained policy weights and source HDF5 data were not in the exported repository evidence.',
            'Across-trajectory cost/error medians are descriptive, not matched counterfactual accuracy measurements.',
            'Absence of a flow consistency gradient is also present in the inspected upstream LeFlow training code; not an isolated explanation of our reproduction gap.',
        ],
    )
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'verified_reports':len(result['reports']),
        'verified_physical_cases':sum(r['episodes'] for r in result['reports']),
        'training':[{k:r[k] for k in ('dataset','selected_step','selected_training_draws',
            'eligible_starts','selected_draws_per_eligible_start','final_draws_per_eligible_start')}
            for r in result['training']], 'consistency_graph':result['consistency_graph']},indent=2))


if __name__=='__main__':main()
