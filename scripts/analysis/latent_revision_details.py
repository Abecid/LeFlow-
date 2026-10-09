"""Summarize existing records; performs no model or simulator calls."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo',default='.')
parser.add_argument('--raw-run-dir',help='Optional live exported run; otherwise use published compressed records.')
args=parser.parse_args()
repo=Path(args.repo)
base=repo/'docs/reports/20261009-latent-revision'
review=json.loads((base/'validation-review.json').read_text())
root=Path(args.raw_run_dir) if args.raw_run_dir else base/'run'
def load_text(path):
    return gzip.decompress(path.read_bytes()).decode() if path.suffix=='.gz' else path.read_text()
rows=[]
for path in sorted((root/'validation').glob('step_*.json*')):
    report=json.loads(load_text(path))
    ds=[d for r in report['records'] for d in r['controller_diagnostics']['decisions']]
    obs=[d for d in ds if 'actual_prefix_target_progress' in d]
    tails=[]
    for r in report['records']:
        if r['success']:continue
        tail=r['controller_diagnostics']['decisions'][-20:]
        observed=[d for d in tail if 'actual_prefix_target_progress' in d]
        actual=np.mean([d['actual_prefix_target_progress'] for d in observed])
        predicted=np.mean([d['predicted_prefix_target_progress'] for d in observed])
        corrected=np.mean([d['corrected_prefix_target_progress'] for d in observed])
        goal_change=tail[-1]['candidate_route_costs'][-1]-tail[0]['candidate_route_costs'][-1]
        tails.append(dict(id=r['id'],actual_target_progress=float(actual),predicted_target_progress=float(predicted),
            corrected_target_progress=float(corrected),final_goal_cost_change=float(goal_change),
            direct_goal_rate=float(np.mean([d['anchor'].get('direct_goal',False) for d in tail])),
            final_anchor=tail[-1]['anchor']))
    row=dict(step=report['step'],successes=sum(r['success'] for r in report['records']),
        mean_selected_prefix_correction=float(np.mean([d['selected_prefix_cost_correction'] for d in ds])),
        mean_selected_terminal_correction=float(np.mean([d['selected_terminal_cost_correction'] for d in ds])),
        positive_terminal_penalty_count=sum(d['applied_terminal_penalty']>0 for d in ds),
        changed_raw_pool_anchor_count=sum(d['correction_changed_raw_pool_anchor'] for d in ds),
        raw_positive_count=sum(d['predicted_prefix_target_progress']>0 for d in obs),
        raw_false_positive_count=sum(d['predicted_prefix_target_progress']>0 and d['actual_prefix_target_progress']<=0 for d in obs),
        corrected_false_positive_count=sum(d['corrected_prefix_target_progress']>0 and d['actual_prefix_target_progress']<=0 for d in obs),
        failures_with_raw_optimistic_nonprogressing_tail=sum(t['predicted_target_progress']>0 and t['actual_target_progress']<=0 for t in tails),
        failures_with_corrected_optimistic_nonprogressing_tail=sum(t['corrected_target_progress']>0 and t['actual_target_progress']<=0 for t in tails),
        failures_with_non_decreasing_final_goal_cost_in_tail=sum(t['final_goal_cost_change']>=0 for t in tails),
        failures_with_positive_local_progress_but_non_decreasing_final_goal_cost=sum(t['actual_target_progress']>0 and t['final_goal_cost_change']>=0 for t in tails),
        tails=tails)
    rows.append(row)
metric_path=root/'metrics.jsonl'
if not metric_path.exists():metric_path=root/'metrics.jsonl.gz'
logs=[json.loads(l) for l in load_text(metric_path).splitlines()]
training=[]
for row in rows:
    step=row['step'];window=[x for x in logs if step-500<x.get('step',0)<=step]
    keys=['train/action_nll','train/calibration_loss','train/shallow_calibration_loss','train/paired_regression',
          'train/recorded_raw_prefix_mae','train/recorded_corrected_prefix_mae',
          'train/recorded_raw_terminal_mae','train/recorded_corrected_terminal_mae','system/update_examples_per_second']
    training.append(dict(ending_step=step,entries=len(window),averages={k:float(np.mean([x[k] for x in window])) for k in keys}))
result=dict(selected_step=review['selected_step'],rounds=rows,training_last_500_update_windows=training,
    interpretation='Chosen-action traces only; no rejected-candidate physical outcomes. Goal-cost tail compares first/last pre-decision observations in the last 20 decisions, not terminal simulator reward. A cosine trend alone does not establish task progress.',
    model_calls=0,simulator_calls=0,final_tests_read=False)
(base/'extra-diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(selected_step=result['selected_step'],rounds=[dict(step=r['step'],successes=r['successes']) for r in rows],model_calls=0,simulator_calls=0)))
