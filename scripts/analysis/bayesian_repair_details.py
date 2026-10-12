"""Read-only posterior comparison audit and factual calibration summaries."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

from scripts.analysis.repair_flow_ttt_details import load, summarize


def check_decision(d):
    assert d['candidate_world_transitions']==280 and d['cached_world_transitions']==40
    assert d['posterior_observation_unit']=='one_executed_transition'
    beta=d['acceptance_beta'];assert beta==1.644854
    rounds=d['posterior_comparisons'];assert len(rounds)==2
    for index,entry in enumerate(rounds):
        for route in range(8):
            costs=entry['costs'][route];selected=entry['selected'][route]
            assert len(costs)==4 and len(entry['actions'][route])==4
            for candidate in range(4):
                signed=entry['signed_corrections'][route][candidate]
                expected=entry['raw_costs'][route][candidate]+.5*(entry['applied_prefix'][route][candidate]+max(0,signed[1]))
                assert abs(expected-costs[candidate])<2e-7
                assert entry['predictive_variance'][route][candidate]>=entry['epistemic_variance'][route][candidate]>=0
                assert all(abs(a)<=1 for block in entry['actions'][route][candidate] for a in block)
            if not index:
                assert costs[selected]<=min(costs)+1e-7
                continue
            previous=rounds[0];old=previous['selected'][route]
            assert entry['actions'][route][0]==previous['actions'][route][old]
            assert abs(costs[0]-previous['costs'][route][old])<1e-7
            for candidate in range(4):
                gain=entry['gain'][route][candidate];uncertainty=entry['uncertainty'][route][candidate]
                lower=entry['lower'][route][candidate];eligible=entry['eligible'][route][candidate]
                assert abs(gain-(costs[0]-costs[candidate]))<1e-7
                assert uncertainty>=0 and abs(lower-(gain-beta*uncertainty))<1e-7
                assert eligible==(candidate==0 or lower>0)
                assert max(abs(x-y) for bx,by in zip(entry['actions'][route][candidate],entry['actions'][route][0]) for x,y in zip(bx,by))<=.250001
            valid=[cost for cost,eligible in zip(costs,entry['eligible'][route]) if eligible]
            assert entry['eligible'][route][selected] and costs[selected]<=min(valid)+1e-7
            if selected:assert entry['lower'][route][selected]>0
    if 'observed_prefix_cost_error' in d:
        observed=d['observed_prefix_cost_error']
        assert abs(observed-(d['predicted_prefix_target_progress']-d['actual_prefix_target_progress']))<1e-7
        assert abs(d['applied_prefix_prediction_error']-(observed-d['selected_applied_prefix_correction']))<1e-7
        assert abs(d['corrected_prefix_target_progress']-(d['predicted_prefix_target_progress']-d['selected_applied_prefix_correction']))<1e-7
        expected=(observed-d['selected_unbounded_prefix_mean'])**2/max(d['selected_prefix_predictive_variance'],1e-12)
        assert abs(d['posterior_standardized_error']-expected)<1e-6


def analyze(report):
    summary=summarize(report);decisions=[d for r in report['records'] for d in r['controller_diagnostics']['decisions']]
    counters=Counter();gain=[];uncertainty=[];observed=[]
    for d in decisions:
        check_decision(d)
        if 'posterior_standardized_error' in d:observed.append(d)
        entry=d['posterior_comparisons'][1]
        for route in range(8):
            counters['route_decisions']+=1
            selected=entry['selected'][route]
            counters['accepted_repairs']+=selected!=0
            counters['incumbents_retained']+=selected==0
            for candidate in range(1,4):
                g=entry['gain'][route][candidate];u=entry['uncertainty'][route][candidate]
                gain.append(g);uncertainty.append(u)
                counters['repair_candidates']+=1
                counters['positive_predicted_gain']+=g>0
                counters['positive_gain_rejected_by_margin']+=g>0 and not entry['eligible'][route][candidate]
    summary.update(posterior_decisions=dict(counters),mean_candidate_gain=mean(gain),
        mean_comparison_uncertainty=mean(uncertainty),
        observed_factual_central_90_coverage=mean(d['posterior_standardized_error']<=1.644854**2 for d in observed) if observed else None,
        observed_factual_standardized_error=mean(d['posterior_standardized_error'] for d in observed) if observed else None,
        comparison_and_incumbent_audit_passed=True,
        limitations=['Unselected alternatives have no observed physical outcomes.',
            'Acceptance/rejection groups are observational, not causal ablations.',
            'Posterior covers prefix-error conditional mean only; calibration need not imply better task success.'])
    per_task = {}
    failure_routes = []
    for task in sorted({r['task'] for r in report['records']}):
        records = [r for r in report['records'] if r['task']==task]
        factual = [d for r in records for d in r['controller_diagnostics']['decisions']
                   if 'observed_prefix_cost_error' in d]
        per_task[task] = dict(successes=sum(bool(r['success']) for r in records),
            timeouts=sum(bool(r['controller_budget_exhausted']) for r in records),
            observed_prefixes=len(factual),
            raw_prefix_mae=mean(abs(d['observed_prefix_cost_error']) for d in factual),
            applied_prefix_mae=mean(abs(d['applied_prefix_prediction_error']) for d in factual),
            unbounded_posterior_mean_mae=mean(abs(d['observed_prefix_cost_error']-d['selected_unbounded_prefix_mean']) for d in factual),
            factual_90_coverage=mean(d['posterior_standardized_error']<=1.644854**2 for d in factual))
        for record in records:
            if record['success']: continue
            tail=record['controller_diagnostics']['decisions'][-20:]
            routes=Counter('direct_goal' if d['anchor'].get('direct_goal')
                           else d['anchor']['episode'].split('/')[1] for d in tail)
            failure_routes.append(dict(id=record['id'],steps=record['steps'],
                timeout=record['controller_budget_exhausted'],last20_route_provenance=dict(routes),
                matching_task_routes=routes[task]))
    summary.update(per_task_factual_calibration=per_task, failure_route_tails=failure_routes,
        observed_applied_prefix_mae=mean(abs(d['applied_prefix_prediction_error']) for d in observed),
        observed_unbounded_posterior_mean_mae=mean(abs(d['observed_prefix_cost_error']-d['selected_unbounded_prefix_mean']) for d in observed))
    summary['limitations'].append('Factual prefix error compares encoded observed frames with world predictions; it is not physical task progress. Route task names are reporting provenance, never an oracle policy filter.')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=analyze(load(a.report));Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('step','successes','timeouts','posterior_decisions')}))
