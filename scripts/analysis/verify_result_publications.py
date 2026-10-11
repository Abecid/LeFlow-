"""Verify the saved comparison ledger; no SSH, model calls or test-suite access."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def verify(repo, ledger_path):
    repo = repo.resolve()
    ledger = read(ledger_path)
    common_cases = None
    total_files = total_evaluations = 0
    identities = set()
    for run in ledger['completed_runs']:
        assert run['run_key'] not in identities, run['run_key']
        identities.add(run['run_key'])
        for relative, expected in run['artifact_sha256'].items():
            path = (repo / relative).resolve()
            assert path.is_relative_to(repo), relative
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
            total_files += 1
        outcomes = []
        for point in run['curve']:
            assert point['artifact'] in run['artifact_sha256']
            report = read(repo / point['artifact'])
            records = report['records']
            cases = sorted(tuple(r[k] for k in (
                'id', 'episode_sha256', 'reset_seed', 'model_seed', 'task'
            )) for r in records)
            assert len(cases) == len(set(cases)) == 104
            assert all(x[0].startswith('validation/') for x in cases)
            if common_cases is None:
                common_cases = cases
            assert cases == common_cases, run['run_key']
            successes = sum(bool(r['success']) for r in records)
            assert successes == point['successes']
            assert abs(report['metrics']['success_macro'] - successes / 104) < 1e-8
            assert sum(bool(r['controller_budget_exhausted']) for r in records) == point['timeouts']
            if run['trained_head']:
                assert report['step'] == point['step']
            outcomes.append(point)
            total_evaluations += 1
        chosen = max(outcomes, key=lambda p: (p['successes'], -p['step']))
        assert chosen['step'] == run['selected_step']
        assert chosen['successes'] == run['selected_successes']
        complete = read(repo / run['complete_artifact'])
        assert run['complete_artifact'] in run['artifact_sha256']
        if run['trained_head']:
            assert complete['best_step'] == chosen['step']
            assert complete['validation_steps'] == [p['step'] for p in outcomes]
            assert abs(complete['best'] - chosen['successes'] / 104) < 1e-8
            assert run['metrics_artifact'] in run['artifact_sha256']
    return dict(completed_runs=len(identities), artifacts_verified=total_files,
                validation_reports_verified=total_evaluations,
                cases_per_report=104, all_case_identities_match=True,
                active_runs=len(ledger['active_runs']), model_calls=0,
                simulator_calls=0, final_tests_read=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--ledger', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.repo, args.ledger or args.repo / 'docs/results-ledger.json'), indent=2))
