"""Build a result matrix from actual case records; never invent missing scores."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from flow_jepa.common import save_json, file_hash
from .spec import METHODS
from .evaluate import summarize


def collect(workspace,repo=None):
    workspace=Path(workspace);repo=Path(repo or Path(__file__).resolve().parents[1])
    rows=[];pairs=[];selected={}
    ledger=json.loads((repo/'docs/results-ledger.json').read_text())
    mapping={'leflow':'leflow_release'}
    for method in METHODS:
        old=next((r for r in ledger['completed_runs'] if r['method']==mapping.get(method,method)),None)
        rows.append(dict(dataset='metaworld',method=method,status='reused' if old else 'unavailable',
            successes=old['selected_successes'] if old else None,episodes=104 if old else None,
            selected_step=old['selected_step'] if old else None,source_report=old['report'] if old else None))
    for task in ('pusht','cube'):
        for method in METHODS:
            root=workspace/'runs'/task/method
            complete=root/'complete.json';best=root/'best.json'
            row=dict(dataset=task,method=method,status='not_started',successes=None,episodes=None,selected_step=None)
            if (root/'usage.json').exists():
                row.update(status='incomplete',last_update=json.loads((root/'usage.json').read_text())['step'])
            if complete.exists():
                c=json.loads(complete.read_text());b=json.loads(best.read_text())
                if c['identity'].get('fixture'):raise ValueError('Refusing fixture results in comparison table')
                path=root/'validation'/f'step_{b["step"]:07d}'/'report.json'
                report=json.loads(path.read_text());records=report['records'];metrics=summarize(records)
                if metrics['successes']!=b['metrics']['successes']:raise ValueError('Reported success disagrees with case records')
                if file_hash(root/'best.pt')!=b['checkpoint_sha256']:raise ValueError('Selected checkpoint hash changed')
                row.update(status='completed',selected_step=b['step'],**metrics,
                    optimization_gpu_hours=c['optimization_gpu_hours'],evaluation_gpu_hours=c['evaluation_gpu_hours'])
                selected[task,method]={r['id']:r for r in records}
                curve=[]
                for rp in sorted((root/'validation').glob('*/report.json')):
                    result=json.loads(rp.read_text());metric=summarize(result['records'])
                    curve.append(dict(checkpoint=rp.parent.name,successes=metric['successes'],timeouts=metric['timeouts']))
                row['curve']=curve
            rows.append(row)
        base=selected.get((task,'vanilla_action_flow'))
        if base:
            for method in METHODS:
                current=selected.get((task,method))
                if current is None:continue
                if set(base)!=set(current):raise ValueError('Paired comparison case IDs differ')
                for key in base:
                    fields=('episode','source_offset','start','goal_offset','seed')
                    if any(base[key][f]!=current[key][f] for f in fields):raise ValueError('Paired reset identity changed')
                pairs.append(dict(dataset=task,method=method,reference='vanilla_action_flow',
                    wins=[k for k in base if current[k]['success'] and not base[k]['success']],
                    regressions=[k for k in base if base[k]['success'] and not current[k]['success']]))
    result=dict(rows=rows,paired_comparisons=pairs,scope='development; native success; not cross-dataset aggregate SOTA')
    out=workspace/'analysis';out.mkdir(parents=True,exist_ok=True)
    save_json(out/'matrix.json',result)
    keys=['dataset','method','status','successes','episodes','selected_step','timeouts','mean_decision_ms','optimization_gpu_hours']
    with (out/'matrix.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);a=p.parse_args();collect(a.workspace)
