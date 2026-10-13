"""Compare factual matched-case reports; reject incompatible evaluation protocols."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path


def paired(reference, candidate):
    from crossbench.spec import Spec, METHODS
    a,b=reference['identity'],candidate['identity']
    native_a=str(a.get('protocol','')).startswith('native-swm')
    native_b=str(b.get('protocol','')).startswith('native-swm')
    if native_a != native_b:
        raise ValueError('Native and shared-execution reports are not comparable')
    if native_a:
        keys=('protocol','task','native_execution','action_budget','goal_offset','source',
              'cases','seed','normalized_action_bounds','normalization','split_file_sha256')
        if any(a.get(k)!=b.get(k) for k in keys):
            raise ValueError('Native inputs/execution/cases differ')
        expected=len(a['cases'])
        def index(r):return r['case']['id']
        def reset(r):return r['case']
    else:
        # New reports carry the full spec. Legacy reports carry its hash.
        specs=[x.get('specification') for x in (a,b) if x.get('specification')]
        if not specs:raise ValueError('At least one refinement report with full specification is required')
        s=Spec(**specs[0]);valid={replace(s,method=m).identity() for m in METHODS}
        for identity in (a,b):
            if 'specification' in identity:
                if Spec(**identity['specification']).identity() not in valid:
                    raise ValueError('Shared-execution specifications differ')
            elif identity.get('spec') not in valid:
                raise ValueError('Unrecognized legacy specification hash')
        if a.get('cache')!=b.get('cache') or a.get('cases',a.get('case_hash'))!=b.get('cases',b.get('case_hash')):
            raise ValueError('Cache or case identities differ')
        expected=s.eval_episodes
        def index(r):return r['id']
        def reset(r):return {k:r[k] for k in ('episode','source_offset','start','goal_offset','seed')}
    ra={index(r):r for r in reference['records']};rb={index(r):r for r in candidate['records']}
    if (len(ra)!=expected or len(rb)!=expected or set(ra)!=set(rb) or
            len(ra)!=len(reference['records']) or len(rb)!=len(candidate['records'])):
        raise ValueError('Incomplete, duplicated or unmatched case inventory')
    for key in ra:
        if reset(ra[key])!=reset(rb[key]):raise ValueError('Reset identity differs')
    wins=[k for k in ra if rb[k]['success'] and not ra[k]['success']]
    losses=[k for k in ra if ra[k]['success'] and not rb[k]['success']]
    return dict(episodes=expected,reference_successes=sum(r['success'] for r in ra.values()),
        candidate_successes=sum(r['success'] for r in rb.values()),
        delta_percentage_points=100*(len(wins)-len(losses))/expected,
        wins=wins,regressions=losses,
        latency_hardware_verified=False,
        warning='Compare latency only after verifying hardware, precision and software; legacy reports lack those fields')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',required=True);p.add_argument('--candidate',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    result=paired(json.loads(Path(a.reference).read_text()),json.loads(Path(a.candidate).read_text()))
    from flow_jepa.common import save_json
    save_json(a.output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
