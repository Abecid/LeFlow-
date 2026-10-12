"""Read-only route provenance audit; task names are diagnostic labels only."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


def source_task(decision):
    anchor=decision['anchor']
    if anchor.get('direct_goal'):
        assert anchor.get('episode') is None
        return 'direct_goal'
    episode=anchor['episode']
    assert episode.startswith('train/')
    return episode.split('/')[1]


def summarize(records, task):
    rows=[r for r in records if r['task']==task]
    cases=[]
    for row in rows:
        decisions=row['controller_diagnostics']['decisions']
        tail=decisions[-20:]
        first=source_task(decisions[0])
        cases.append(dict(id=row['id'],success=bool(row['success']),steps=row['steps'],
            timeout=bool(row['controller_budget_exhausted']),decisions=len(decisions),
            initial_route_task=first,all_sources=dict(Counter(map(source_task,decisions))),
            tail_sources=dict(Counter(map(source_task,tail))),tail_decisions=len(tail),
            tail_has_matching_task_route=any(source_task(d)==task for d in tail)))
    return dict(successes=sum(r['success'] for r in cases),cases=cases,
        failed_cases_without_matching_route_in_tail=sum(not r['success'] and not r['tail_has_matching_task_route'] for r in cases))


def main(a):
    repo=Path(a.repo); ledger=json.loads((repo/'docs/results-ledger.json').read_text());out={}
    identity=None
    for method in ['execution_revision','repair_flow_ttt','retrieval_repair_ttt','shifted_repair_ttt']:
        run=next(r for r in ledger['completed_runs'] if r['method']==method)
        relative=next(c['artifact'] for c in run['curve'] if c['step']==run['selected_step'])
        p=repo/relative;raw=p.read_bytes();report=json.loads(gzip.decompress(raw) if p.suffix=='.gz' else raw)
        records=report['records']; assert len(records)==104
        current=sorted(tuple(r[k] for k in ('id','task','episode_sha256','reset_seed','model_seed')) for r in records)
        if identity is None:identity=current
        else:assert identity==current
        assert all(r['id'].startswith('validation/') for r in records)
        out[method]=dict(source=relative,sha256=hashlib.sha256(raw).hexdigest(),step=report['step'],
            successes=sum(bool(r['success']) for r in records),
            tasks={t:summarize(records,t) for t in sorted({r['task'] for r in records})})
    result=dict(methods=out,paired_case_identities_verified=True,training_updates=0,model_calls=0,
        simulator_calls=0,final_tests_read=False,
        interpretation=['Selected-route provenance, not provenance of every candidate action.',
            'Cross-task routes can be useful; their task label is not an error label.',
            'Direct-goal choices can be valid; neither category alone proves a physical failure cause.',
            'Association on selected, repeatedly reused development cases; no causal intervention.',
            'Task names are never fed to a policy or used to filter its routes.'])
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({m:{'successes':v['successes'],'pick_place':v['tasks']['pick-place']['successes'],
        'failed_pick_place_without_matching_tail':v['tasks']['pick-place']['failed_cases_without_matching_route_in_tail']} for m,v in out.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',default='.');p.add_argument('--output',required=True)
    main(p.parse_args())
