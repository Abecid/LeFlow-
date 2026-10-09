"""Offline paired validation review. Never reads a test directory or runs models."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def load(path):
    path=Path(path)
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validation_paths(run):
    paths=sorted((run/'validation').glob('step_*.json*'))
    stems=[p.name.removesuffix('.gz') for p in paths]
    if len(set(stems))!=len(stems):
        raise ValueError('Duplicate compressed/uncompressed validation report')
    return paths


def selected(paths):
    reports=[(p,load(p)) for p in sorted(paths)]
    return max(reports,key=lambda item:(sum(x['success'] for x in item[1]['records']),-item[1]['step']))


def summary(report):
    records=report['records']
    cases=[dict((k,r[k]) for k in ('id','task','reset_seed','episode_sha256','model_seed','success',
           'steps','controller_budget_exhausted','first_success_primitive','return')) for r in records]
    return dict(step=report['step'],successes=sum(r['success'] for r in records),cases=len(records),
        per_task={t:sum(r['success'] for r in records if r['task']==t) for t in sorted({r['task'] for r in records})},
        failures_timeout=sum(not r['success'] and r['controller_budget_exhausted'] for r in records),
        failures_action_cap=sum(not r['success'] and r['steps']==200 for r in records),
        successful_case_ids=[r['id'] for r in records if r['success']],
        records=cases,metrics=report['metrics'])


def trace_summary(records):
    decisions=[d for r in records for d in r.get('controller_diagnostics',{}).get('decisions',[])]
    observed=[d for d in decisions if 'actual_prefix_target_progress' in d]
    if not decisions:
        return {'decisions':0,'observed_prefixes':0}
    pred=np.array([d['predicted_prefix_target_progress'] for d in observed])
    actual=np.array([d['actual_prefix_target_progress'] for d in observed])
    def corr(x,y):
        return float(np.corrcoef(x,y)[0,1]) if len(x)>1 and np.std(x)>0 and np.std(y)>0 else None
    transitions=switches=same_episode=backward=0
    for r in records:
        anchors=[d['anchor'] for d in r.get('controller_diagnostics',{}).get('decisions',[])]
        for a,b in zip(anchors,anchors[1:]):
            transitions+=1
            switches+=a.get('episode')!=b.get('episode')
            if a.get('episode') and a.get('episode')==b.get('episode'):
                same_episode+=1
                backward+=b['start']<a['start']
    chosen_response=[d['candidate_response_costs'][d['selected_anchor']] for d in decisions]
    return dict(decisions=len(decisions),observed_prefixes=len(observed),
        actual_prefix_progress_mean=float(actual.mean()) if len(actual) else None,
        predicted_prefix_progress_mean=float(pred.mean()) if len(pred) else None,
        optimism_gap_mean=float((pred-actual).mean()) if len(actual) else None,
        predicted_actual_progress_pearson=corr(pred,actual),
        selected_response_cost_actual_progress_pearson=corr([d['local_cost'] for d in observed],actual),
        predicted_positive_count=int((pred>0).sum()),
        predicted_positive_actual_nonpositive_count=int(((pred>0)&(actual<=0)).sum()),
        observed_actual_positive_count=int((actual>0).sum()),
        episode_switch_rate=switches/transitions if transitions else None,
        same_episode_backtracking_rate=backward/same_episode if same_episode else None,
        same_episode_transitions=same_episode,
        direct_goal_rate=float(np.mean([d['anchor'].get('direct_goal',False) for d in decisions])),
        response_changes_retrieval_rate=float(np.mean([d['response_changed_retrieval_choice'] for d in decisions])),
        proposal_diversity_mean=float(np.mean([d['proposal_diversity'] for d in decisions])),
        chosen_response_cost_mean=float(np.mean(chosen_response)),
        chosen_route_cost_mean=float(np.mean([d['route_cost'] for d in decisions])),
        per_decision_route_cost_range_mean=float(np.mean([np.ptp(d['candidate_route_costs']) for d in decisions])),
        per_decision_response_cost_range_mean=float(np.mean([np.ptp(d['candidate_response_costs']) for d in decisions])))


def failed_case_tails(records):
    tails=[]
    for r in records:
        if r['success']:continue
        trace=r.get('controller_diagnostics',{}).get('decisions',[])
        if not trace:continue
        tail=trace[-20:]
        row=dict(id=r['id'],task=r['task'],decisions_in_tail=len(tail),
            final_goal_distance_start_tail=tail[0]['candidate_route_costs'][-1],
            final_goal_distance_end=tail[-1]['candidate_route_costs'][-1],
            last_target=tail[-1]['anchor'])
        row.update(trace_summary([dict(controller_diagnostics=dict(decisions=tail))]))
        tails.append(row)
    return tails


def analyze(args):
    repo=Path(args.repo);new=Path(args.run_dir)
    old=repo/'docs/reports/20261007-joint-flow/runs'
    repaired=repo/'docs/reports/20261009-repaired-comparison/runs'
    paths=validation_paths(new)
    p,ours=selected(paths)
    refs={
        'same_world_cem':(repaired/'world_3072/validation/cem_step_0020000.json',22),
        'repaired_flow':(repaired/'joint_flow_consistent_3072/validation/step_0005000.json',17),
        'historical_cem':(old/'world_3072/validation/cem_step_0020000.json',7),
    }
    for name,method,expected in [('historical_leflow','leflow_adapted',27),('historical_hwm','hwm_adapted',8),('historical_flow','joint_flow_consistent',23)]:
        path,_=selected((old/f'{method}_3072/validation').glob('step_*.json'));refs[name]=(path,expected)
    byid={r['id']:r for r in ours['records']}
    assert len(byid)==104 and all(i.startswith('validation/') for i in byid)
    comparison={};rng=np.random.default_rng(40107)
    for name,(path,expected) in refs.items():
        ref=load(path);other={r['id']:r for r in ref['records']}
        assert set(other)==set(byid)
        assert sum(r['success'] for r in other.values())==expected,(name,expected)
        for i,r in byid.items():
            for k in ('reset_seed','episode_sha256','model_seed'):
                assert r[k]==other[i][k],(i,k)
        ours_only=[i for i,r in byid.items() if r['success'] and not other[i]['success']]
        ref_only=[i for i,r in byid.items() if not r['success'] and other[i]['success']]
        differences=[]
        for t in sorted({r['task'] for r in byid.values()}):
            d=np.array([int(r['success'])-int(other[i]['success']) for i,r in byid.items() if r['task']==t])
            differences.append(d[rng.integers(0,len(d),(10000,len(d)))].mean(1))
        interval=np.quantile(np.mean(differences,axis=0)*100,[.025,.975]).tolist()
        comparison[name]=dict(successes=expected,ours_only=ours_only,reference_only=ref_only,
            difference_pp=100*(len(ours_only)-len(ref_only))/104,
            descriptive_paired_reset_interval95_pp=interval,source=str(path.relative_to(repo)),sha256=sha(path))
    decisions=[d for r in ours['records'] for d in r.get('controller_diagnostics',{}).get('decisions',[])]
    observed=[d for d in decisions if 'actual_prefix_target_progress' in d]
    pred=np.array([d['predicted_prefix_target_progress'] for d in observed])
    actual=np.array([d['actual_prefix_target_progress'] for d in observed])
    correlation=float(np.corrcoef(pred,actual)[0,1]) if len(actual)>1 and pred.std()>0 and actual.std()>0 else None
    cross_task=0;retrieved=0;switches=0;transitions=0
    for r in ours['records']:
        trace=r.get('controller_diagnostics',{}).get('decisions',[])
        previous=None
        for d in trace:
            anchor=d['anchor'];episode=anchor.get('episode')
            if episode:
                retrieved+=1;cross_task+=episode.split('/')[1]!=r['task']
            key=(episode,anchor.get('start'),anchor.get('span'))
            if previous is not None:transitions+=1;switches+=key!=previous
            previous=key
    result=dict(selected_step=ours['step'],source_report=str(p),source_sha256=sha(p),
        rounds=[summary(load(p)) for p in paths],
        selected=summary(ours),comparison=comparison,case_pairing_verified=True,
        diagnostics=dict(decisions=len(decisions),observed_prefixes=len(observed),
            response_changes_retrieval_rate=float(np.mean([d['response_changed_retrieval_choice'] for d in decisions])) if decisions else None,
            direct_goal_rate=float(np.mean([d['anchor'].get('direct_goal',False) for d in decisions])) if decisions else None,
            predicted_actual_prefix_progress_pearson=correlation,
            predicted_positive_actual_nonpositive_rate=float(np.mean(actual[pred>0]<=0)) if (pred>0).any() else None,
            actual_prefix_prediction_error_mean=float(np.mean([d['actual_prefix_prediction_error'] for d in observed])) if observed else None,
            cross_task_retrieval_rate=cross_task/retrieved if retrieved else None,
            route_switch_rate=switches/transitions if transitions else None,
            chosen_refinement_round_counts=dict(Counter(str(d['selected_after_refinement_round']) for d in decisions))),
        trace_review=trace_summary(ours['records']),
        trace_by_task={t:trace_summary([r for r in ours['records'] if r['task']==t])
                       for t in sorted({r['task'] for r in ours['records']})},
        trace_by_outcome={label:trace_summary([r for r in ours['records'] if bool(r['success'])==success])
                          for label,success in [('success',True),('failure',False)]},
        failed_case_last_20_decisions=failed_case_tails(ours['records']),
        limits=['Selected reused validation; not an unbiased final-test estimate.',
                'One model seed. Bootstrap covers descriptive paired reset variation only.',
                'Historical reference world/representation/sampler differences remain.',
                'Prefix progress uses latent distance, not full-chunk execution or counterfactual rank calibration.',
                'Trace correlations pool dependent decisions; no independence or causal inference is claimed.',
                'Cross-task retrieval is a warning to inspect, not by itself proof of bad targets.'],
        training_complete=(new/'complete.json').exists(),final_tests_read=False,
        compute=load(new/'compute_usage.json'),run=load(new/'run.json'))
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(selected_step=ours['step'],successes=result['selected']['successes'],diagnostics=result['diagnostics'])))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('repo','run-dir','output'):p.add_argument('--'+name,required=True)
    analyze(p.parse_args())
