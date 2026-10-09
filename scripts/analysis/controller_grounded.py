"""Offline paired validation review. Never reads a test directory or runs models."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def load(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def analyze(args):
    repo=Path(args.repo);new=Path(args.run_dir)
    old=repo/'docs/reports/20261007-joint-flow/runs'
    repaired=repo/'docs/reports/20261009-repaired-comparison/runs'
    p,ours=selected((new/'validation').glob('step_*.json'))
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
        rounds=[summary(load(p)) for p in sorted((new/'validation').glob('step_*.json'))],
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
        limits=['Selected reused validation; not an unbiased final-test estimate.',
                'One model seed. Bootstrap covers descriptive paired reset variation only.',
                'Historical reference world/representation/sampler differences remain.',
                'Prefix progress uses latent distance, not full-chunk execution or counterfactual rank calibration.',
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
