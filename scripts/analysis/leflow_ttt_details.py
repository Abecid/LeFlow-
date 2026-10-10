"""Analyze saved LeFlow-TTT records without model or simulator calls."""
import argparse
import gzip
import json
from pathlib import Path
from statistics import mean


def load(path):
    p=Path(path)
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_text())


def average(xs):
    xs=list(xs)
    return mean(xs) if xs else None


def summarize(r):
    rows=r['records'];assert len(rows)==104
    decisions=[d for row in rows for d in row['controller_diagnostics']['decisions']]
    observed=[d for d in decisions if 'actual_progress' in d]
    adapted_observed=[d for d in observed if d['history_chunks']>0]
    for row in rows:
        ds=row['controller_diagnostics']['decisions']
        assert row['steps']<=200 and row['controller_budget_seconds']==10
        for i,d in enumerate(ds):
            assert d['history_chunks']==min(i,4)
            assert d['candidates']==64 and d['flow_steps']==16 and d['inner_steps']==4
            assert d['execution_blocks']==5 and d['world_transitions']==325
            if i==0:assert d['fast_weight_delta']==0
        if not row['controller_budget_exhausted']:
            assert row['world_predictions']==325*len(ds)
    count=sum(bool(x['success']) for x in rows)
    assert abs(r['metrics']['success_macro']-count/104)<1e-7
    groups={}
    for name,items in [('all',observed),('with_history',adapted_observed)]:
        groups[name]={k:average(d[k] for d in items) for k in
            ('raw_outcome_mse','prior_outcome_mse','adapted_outcome_mse','actual_progress')}
        groups[name]['count']=len(items)
        groups[name]['adaptation_improves_prior_fraction']=average(d['adapted_outcome_mse']<d['prior_outcome_mse'] for d in items)
    failures=[]
    for row in rows:
        if row['success']:continue
        ds=row['controller_diagnostics']['decisions']
        observed_tail=[d for d in ds[-5:] if 'actual_progress' in d]
        failures.append(dict(id=row['id'],steps=row['steps'],timeout=row['controller_budget_exhausted'],
            tail_actual_progress=average(d['actual_progress'] for d in observed_tail),
            tail_raw_progress=average(d['predicted_progress'] for d in observed_tail),
            tail_corrected_progress=average(d['corrected_progress'] for d in observed_tail),
            tail_outcome_mse=average(d['adapted_outcome_mse'] for d in observed_tail),
            selected_clipping=average(d['selected_action_clipping_fraction'] for d in ds)))
    return dict(step=r['step'],successes=count,
        tasks={t:sum(bool(x['success']) for x in rows if x['task']==t) for t in sorted({x['task'] for x in rows})},
        timeouts=sum(bool(x['controller_budget_exhausted']) for x in rows),metrics=r['metrics'],
        decisions=len(decisions),outcomes=groups,
        decision_means={k:average(d[k] for d in decisions) for k in
            ('support_loss_before','support_loss_after','fast_weight_delta','adaptation_changed_selection',
             'mean_correction_norm','action_outside_bounds','selected_action_clipping_fraction',
             'latent_path_diversity','action_diversity')},
        first_decision_means={k:average(x['controller_diagnostics']['decisions'][0][k] for x in rows) for k in
            ('latent_path_diversity','action_diversity','selected_action_clipping_fraction')},
        failures=failures)


def paired(a,b):
    aa={x['id']:x for x in a['records']};bb={x['id']:x for x in b['records']}
    assert set(aa)==set(bb)
    for key in aa:
        for k in ('episode_sha256','reset_seed','model_seed','task'):
            assert aa[key][k]==bb[key][k],(key,k)
    wins=[k for k in aa if aa[k]['success'] and not bb[k]['success']]
    losses=[k for k in aa if not aa[k]['success'] and bb[k]['success']]
    return dict(candidate_step=a['step'],reference_step=b['step'],wins=wins,regressions=losses,
        both_success=sum(aa[k]['success'] and bb[k]['success'] for k in aa),
        both_failure=sum(not aa[k]['success'] and not bb[k]['success'] for k in aa),
        difference_pp=100*(len(wins)-len(losses))/104,case_pairing_verified=True)


def main(a):
    root=Path(a.run_dir);repo=Path(a.repo)
    reports=[load(p) for p in sorted((root/'validation').glob('step_*.json*'))]
    assert 0<len(reports)<=4
    selected=max(reports,key=lambda r:(r['metrics']['success_macro'],-r['step']))
    base_root=repo/'docs/reports/20261010-release-baselines/leflow/validation'
    baselines=[load(p) for p in sorted(base_root.glob('step_*.json*'))]
    best=max(baselines,key=lambda r:(r['metrics']['success_macro'],-r['step']))
    assert sum(x['success'] for x in best['records'])==21
    rounds=[summarize(r) for r in reports]
    for r in reports:paired(r,selected)
    matched=[]
    for r in reports:
        other=next((b for b in baselines if b['step']==r['step']),None)
        if other:matched.append(paired(r,other))
    result=dict(selected_step=selected['step'],rounds=rounds,comparison_to_selected_leflow=paired(selected,best),
        matched_milestones=matched,selected_to_last=paired(reports[-1],selected),
        complete=load(root/'complete.json') if (root/'complete.json').exists() else None,
        compute=load(root/'compute_usage.json'),run=load(root/'run.json'),
        model_calls=0,simulator_calls=0,final_tests_read=False,
        limits=['One seed; repeatedly used development cases; checkpoint selection.',
            'Fixed-budget released-code MetaWorld port; not native LeFlow benchmark or convergence claim.',
            'Observed prediction errors cover executed actions only; no counterfactual action labels.',
            'Raw-versus-corrected argmin includes the learned prior, not isolated online adaptation.',
            'No ablation or inference-depth sweep establishes a causal TTT/scaling benefit.'])
    Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(selected_step=selected['step'],scores=[(x['step'],x['successes']) for x in rounds],
                         comparison=result['comparison_to_selected_leflow'])))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('run-dir','repo','output'):p.add_argument('--'+k,required=True)
    main(p.parse_args())
