"""Analyze saved validation decisions only; no model or environment calls."""
import argparse
import json
from pathlib import Path

import numpy as np

from .controller_grounded import load, validation_paths


def summarize(report):
    ds=[d for r in report['records'] for d in r['controller_diagnostics']['decisions']]
    observed=[d for d in ds if 'actual_prefix_target_progress' in d]
    def mean(values):
        return float(np.mean(values)) if len(values) else None
    def group(decisions):
        return dict(decisions=len(decisions),
            warm_start_rate=mean([d['warm_start_available'] for d in decisions]),
            selected_shifted_prefix_rate=mean([d['selected_plan_matches_shifted_prefix'] for d in decisions]),
            stall_changed_anchor_count=sum(d['stall_changed_pool_anchor'] for d in decisions),
            any_candidate_stall_penalty_count=sum(max(d['candidate_stall_penalties'])>0 for d in decisions),
            selected_stall_penalty_mean=mean([d['applied_stall_penalty'] for d in decisions]),
            correction_changed_action_batches=sum(d['correction_changed_action_batches'] for d in decisions),
            evaluated_action_batches=sum(d['evaluated_action_batches'] for d in decisions),
            correction_changed_anchor_count=sum(d['correction_changed_raw_pool_anchor'] for d in decisions),
            applied_prefix_penalty_mean=mean([d['applied_prefix_penalty'] for d in decisions]),
            applied_terminal_penalty_mean=mean([d['applied_terminal_penalty'] for d in decisions]))
    tails=[]
    for record in report['records']:
        if record['success']:continue
        tail=record['controller_diagnostics']['decisions'][-20:]
        obs=[d for d in tail if 'actual_prefix_target_progress' in d]
        tails.append(dict(id=record['id'],
            actual=mean([d['actual_prefix_target_progress'] for d in obs]),
            predicted=mean([d['predicted_prefix_target_progress'] for d in obs]),
            corrected=mean([d['corrected_prefix_target_progress'] for d in obs]),
            final_goal_cost_change=tail[-1]['candidate_route_costs'][-1]-tail[0]['candidate_route_costs'][-1],
            **group(tail)))
    raw=np.array([d['predicted_prefix_target_progress'] for d in observed])
    corrected=np.array([d['corrected_prefix_target_progress'] for d in observed])
    actual=np.array([d['actual_prefix_target_progress'] for d in observed])
    return dict(step=report['step'],successes=sum(r['success'] for r in report['records']),
        **group(ds),observed_prefixes=len(observed),raw_mae=mean(abs(raw-actual)),corrected_mae=mean(abs(corrected-actual)),
        raw_positive_count=int((raw>0).sum()),raw_false_positive_count=int(((raw>0)&(actual<=0)).sum()),
        corrected_positive_count=int((corrected>0).sum()),corrected_false_positive_count=int(((corrected>0)&(actual<=0)).sum()),
        failures_with_raw_optimistic_nonprogressing_tail=sum(t['predicted']>0 and t['actual']<=0 for t in tails),
        failures_with_corrected_optimistic_nonprogressing_tail=sum(t['corrected']>0 and t['actual']<=0 for t in tails),
        failures_with_non_decreasing_final_goal_cost_in_tail=sum(t['final_goal_cost_change']>=0 for t in tails),
        by_task={t:group([d for r in report['records'] if r['task']==t for d in r['controller_diagnostics']['decisions']])
                 for t in sorted({r['task'] for r in report['records']})},tails=tails)


def main(args):
    root=Path(args.run_dir)
    rows=[summarize(load(p)) for p in validation_paths(root)]
    logs=[json.loads(line) for line in (root/'metrics.jsonl').read_text().splitlines()]
    windows=[]
    for row in rows:
        window=[x for x in logs if row['step']-500<x.get('step',0)<=row['step']]
        keys=['train/action_nll','train/calibration_loss','train/recorded_raw_prefix_mae',
              'train/recorded_corrected_prefix_mae','train/late_window_fraction','train/sample_start_mean',
              'train/goal_offset_mean','system/update_examples_per_second']
        windows.append(dict(ending_step=row['step'],entries=len(window),
            averages={k:float(np.mean([x[k] for x in window])) for k in keys}))
    result=dict(rounds=rows,training_last_500_update_windows=windows,model_calls=0,simulator_calls=0,
        interpretation='Same-pool decision changes are diagnostics, not a causal ablation. True outcomes exist only for chosen prefixes. No final tests read.')
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps([dict(step=r['step'],successes=r['successes']) for r in rows]))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True);parser.add_argument('--output',required=True)
    main(parser.parse_args())
