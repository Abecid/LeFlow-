"""Analyze saved progress-objective evaluation records; no model/simulator calls."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import statistics


def load(path):
    path=Path(path);raw=path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix=='.gz' else raw)


def summarize(records):
    decisions=[d for r in records for d in r['controller_diagnostics']['decisions']]
    observed=[d for d in decisions if 'goal_progress_observed' in d]
    mean=lambda values:statistics.mean(values) if values else None
    result=dict(cases=len(records),successes=sum(r['success'] for r in records),
        timeouts=sum(r['controller_budget_exhausted'] for r in records),
        decisions=len(decisions),observed_prefixes=len(observed),
        raw_progress_mae=mean([abs(d['goal_progress_predicted']-d['goal_progress_observed']) for d in observed]),
        applied_progress_mae=mean([abs(d['goal_progress_corrected']-d['goal_progress_observed']) for d in observed]),
        optimism_penalized_decisions=sum(d['goal_progress_error_correction']>0 for d in decisions),
        selected_stage_penalty_mean=mean([d['candidate_stage_penalties'][d['selected_anchor']] for d in decisions]),
        incumbent_retention_rate=mean([int(d['selected_is_unchanged_incumbent']) for d in decisions]))
    tails=[]
    for record in records:
        if record['success']:continue
        ds=record['controller_diagnostics']['decisions'][-20:]
        routes=Counter('direct_goal' if d['anchor'].get('direct_goal') else d['anchor']['episode'].split('/')[1] for d in ds)
        tails.append(dict(id=record['id'],task=record['task'],steps=record['steps'],
            last20_routes=dict(routes),matching_task_routes=routes[record['task']],
            predicted_goal_progress_mean=mean([d['goal_progress_predicted'] for d in ds]),
            observed_goal_progress_mean=mean([d['goal_progress_observed'] for d in ds if 'goal_progress_observed' in d])))
    result['failure_tails']=tails
    return result


def main(a):
    report=load(a.evaluation);records=report['records']
    assert len(records)==104 and all(r['id'].startswith('validation/') for r in records)
    result=dict(step=report['step'],overall=summarize(records),
        per_task={t:summarize([r for r in records if r['task']==t]) for t in sorted({r['task'] for r in records})},
        limitations=['Readout progress is a weak learned expert-time proxy, not physical completion.',
                     'Only executed prefixes have observed outcomes; rejected candidates have no physical labels.',
                     'Route task identities are diagnostic, never policy inputs or causal failure labels.'],
        model_calls=0,simulator_calls=0,final_tests_read=False)
    if a.baseline:
        baseline={r['id']:r for r in load(a.baseline)['records']}
        wins=[];regressions=[]
        for r in records:
            old=baseline[r['id']]
            assert all(r[k]==old[k] for k in ('episode_sha256','reset_seed','model_seed','task'))
            if r['success'] and not old['success']:wins.append(r['id'])
            if old['success'] and not r['success']:regressions.append(r['id'])
        result['paired_reference']=dict(successes=sum(r['success'] for r in baseline.values()),wins=wins,regressions=regressions)
    Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('overall','per_task')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('evaluation','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--baseline')
    main(p.parse_args())
