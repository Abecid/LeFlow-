"""Read-only analysis of saved development records; no model/simulator calls."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean


def load(path):
    p = Path(path)
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix == '.gz' else p.read_text())


def average(values):
    values = list(values)
    return mean(values) if values else None


def initial_diversity(reports):
    first = {r['id']:r for r in reports[0]['records']}
    rows = []
    for report in reports:
        current = {r['id']:r for r in report['records']}
        row = dict(step=report['step'])
        for key in first:
            old = first[key]['controller_diagnostics']['decisions'][0]
            new = current[key]['controller_diagnostics']['decisions'][0]
            assert new['history_valid_steps']==0 and not new['warm_start_available']
            assert all(abs(a-b)<1e-6 for a,b in zip(old['candidate_route_costs'],new['candidate_route_costs']))
        for index in range(3):
            before = [first[k]['controller_diagnostics']['decisions'][0]['round_proposal_diversity'][index] for k in first]
            after = [current[k]['controller_diagnostics']['decisions'][0]['round_proposal_diversity'][index] for k in first]
            row[str(index)] = dict(mean=mean(after),relative_change=mean(after)/mean(before)-1,
                                  decreased_cases=sum(b<a for a,b in zip(before,after)))
        rows.append(row)
    return dict(rounds=rows,paired_initial_routes_verified=True,history_empty=True,
        interpretation='Same reset/goal and per-case seed. Round zero includes recorded proposals; later rounds are flow proposals. Diversity is not likelihood entropy or a causal attribution.')


def summarize(report):
    records = report['records']
    assert len(records) == 104 and all(r['id'].startswith('validation/') for r in records)
    decisions = [d for r in records for d in r['controller_diagnostics']['decisions']]
    observed = [d for d in decisions if 'actual_prefix_target_progress' in d]
    for d in decisions:
        assert d['candidate_world_transitions'] == 480 and d['reasoning_depth'] == 3
        assert d['scoring_context_fixed'] and d['flow_evaluations'] == 24
        costs = d['round_best_scores']
        assert len(costs) == 3 and all(b <= a for a, b in zip(costs, costs[1:]))
    tails = []
    for r in records:
        assert r['steps'] <= 200
        ds = r['controller_diagnostics']['decisions']
        if not r['controller_budget_exhausted']:
            assert r['world_predictions'] == 480*len(ds)
            assert r['steps'] in (2*len(ds), 2*len(ds)-1)
        if r['success']:
            continue
        tail = ds[-20:]; obs = [d for d in tail if 'actual_prefix_target_progress' in d]
        tails.append(dict(id=r['id'], timeout=r['controller_budget_exhausted'],
            actual=average(d['actual_prefix_target_progress'] for d in obs),
            predicted=average(d['predicted_prefix_target_progress'] for d in obs),
            corrected=average(d['corrected_prefix_target_progress'] for d in obs),
            goal_cost_change=tail[-1]['candidate_route_costs'][-1]-tail[0]['candidate_route_costs'][-1] if tail else None))
    successes = sum(bool(r['success']) for r in records)
    assert abs(report['metrics']['success_macro']-successes/104) < 1e-7
    return dict(step=report['step'], successes=successes,
        per_task={t:sum(bool(r['success']) for r in records if r['task']==t) for t in sorted({r['task'] for r in records})},
        timeouts=sum(bool(r['controller_budget_exhausted']) for r in records),
        metrics=report['metrics'], decisions=len(decisions), observed_prefixes=len(observed),
        selected_round_counts=dict(Counter(str(d['selected_after_refinement_round']) for d in decisions)),
        mean_round_candidate_changes=[average(d['round_action_changes'][k] for d in decisions) for k in range(3)],
        mean_round_proposal_diversity=[average(d['round_proposal_diversity'][k] for d in decisions) for k in range(3)],
        mean_predicted_cost_improvement=average(d['round_best_scores'][0]-d['round_best_scores'][-1] for d in decisions),
        raw_prefix_mae=average(abs(d['predicted_prefix_target_progress']-d['actual_prefix_target_progress']) for d in observed),
        corrected_prefix_mae=average(abs(d['corrected_prefix_target_progress']-d['actual_prefix_target_progress']) for d in observed),
        optimistic_stall_failures=sum(t['predicted'] is not None and t['predicted']>0 and t['actual']<=0 for t in tails),
        corrected_optimistic_stall_failures=sum(t['corrected'] is not None and t['corrected']>0 and t['actual']<=0 for t in tails),
        failure_tails=tails)


def main(args):
    root = Path(args.run_dir); repo = Path(args.repo)
    paths = sorted((root/'validation').glob('step_*.json*'))
    assert paths and len(paths) <= 4
    reports = [load(p) for p in paths]
    chosen = max(reports, key=lambda r:(r['metrics']['success_macro'], -r['step']))
    byid = {r['id']:r for r in chosen['records']}
    refs = {
        'execution_revision':('20261009-execution-revision/run/validation/step_0005000.json.gz', 80),
        'controller_grounded':('20261009-controller-grounded/run/validation/step_0005000.json.gz', 79),
        'leflow_release':('20261010-release-baselines/leflow/validation/step_0010000.json.gz', 21),
        'hwm_paper_port':('20261010-release-baselines/hwm/validation/step_0015000.json.gz', 5),
        'cem_release':('20261010-release-baselines/cem/validation.json', 24),
    }
    comparisons = {}
    for name, (relative, expected) in refs.items():
        path = repo/'docs/reports'/relative
        other = {r['id']:r for r in load(path)['records']}
        assert set(other) == set(byid)
        assert sum(bool(r['success']) for r in other.values()) == expected
        for key in byid:
            for identity in ('episode_sha256', 'reset_seed', 'model_seed', 'task'):
                assert byid[key][identity] == other[key][identity]
        wins = [k for k in byid if byid[k]['success'] and not other[k]['success']]
        losses = [k for k in byid if other[k]['success'] and not byid[k]['success']]
        comparisons[name] = dict(successes=expected, wins=wins, regressions=losses,
            difference_pp=100*(len(wins)-len(losses))/104, source=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    for r in reports:
        assert {x['id'] for x in r['records']} == set(byid)
        for x in r['records']:
            for k in ('episode_sha256', 'reset_seed', 'model_seed'):
                assert x[k] == byid[x['id']][k]
    logs_path = root/'metrics.jsonl'
    text = logs_path.read_text() if logs_path.exists() else gzip.decompress(Path(str(logs_path)+'.gz').read_bytes()).decode()
    logs = [json.loads(line) for line in text.splitlines()]
    train_logs = [r for r in logs if 'step' in r]
    assert len({r['step'] for r in train_logs}) == len(train_logs)
    keys = ['train/flow_loss', 'train/paired_flow_improvement', 'train/paired_flow_regression',
            'train/calibration_loss', 'system/update_examples_per_second']
    windows = []
    for report in reports:
        rows = [r for r in train_logs if report['step']-500 < r['step'] <= report['step']]
        windows.append(dict(step=report['step'], entries=len(rows),
            metrics={k:average(r[k] for r in rows) for k in keys},
            by_depth={str(d):{k:average(r[k] for r in rows if r['train/reasoning_depth']==d) for k in keys}
                      for d in (1, 2, 3)}))
    result = dict(selected_step=chosen['step'], rounds=[summarize(r) for r in reports],
        initial_proposal_diversity=initial_diversity(reports),
        comparisons=comparisons, case_pairing_verified=True, training_windows=windows,
        compute=load(root/'compute_usage.json'), run=load(root/'run.json'),
        model_calls=0, simulator_calls=0, final_tests_read=False,
        limits=['Repeatedly used development cases; selected checkpoint; one model seed.',
                'Round score improvement is a selection property, not physical progress.',
                'Only executed prefixes have real outcome labels.',
                'No inference-depth counterfactual or causal attribution from these traces.'])
    if (root/'complete.json').exists():
        result['complete'] = load(root/'complete.json')
        assert result['complete']['best_step'] == chosen['step']
        assert result['complete']['validation_steps'] == [r['step'] for r in reports]
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(selected_step=chosen['step'], rounds=[(r['step'], sum(bool(x['success']) for x in r['records'])) for r in reports])))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True); p.add_argument('--repo', default='.')
    p.add_argument('--output', required=True)
    main(p.parse_args())
