"""Read-only final audit of the frozen progress-TTT run (CPU only)."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

import torch

from flow_jepa.common import digest, file_hash


def main(args):
    torch.set_num_threads(2)
    record=Path(args.record);repo=record/'repo';root=record/'campaign'
    run=root/'runs/progress_ttt_3072'
    c=json.loads((repo/'config/flow_metaworld_progress_ttt.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    reuse=json.loads((root/'reuse.json').read_text())
    dispatch=json.loads((record/'launcher.json').read_text())
    identity=json.loads((run/'run.json').read_text())
    usage=json.loads((run/'compute_usage.json').read_text())
    complete=json.loads((run/'complete.json').read_text())
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==dispatch['source']==identity['code']
    assert subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True)==''
    assert digest(c)==manifest['protocol']==identity['protocol']
    assert file_hash(root/'manifest.json')==identity['manifest']
    assert digest(manifest['entries'])==reuse['entries_sha256']
    original=json.loads((Path(reuse['source_root'])/'manifest.json').read_text())
    assert file_hash(Path(reuse['source_root'])/'manifest.json')==reuse['world_source_manifest']
    for key in reuse['unchanged_fields']:assert manifest[key]==original[key],key
    world=Path(reuse['source_root'])/'runs/world_3072/best.pt'
    assert file_hash(world)==identity['world_hash']==c['controller_grounded']['world_sha256']
    bank_meta=json.loads((root/'route-bank.json').read_text())
    assert file_hash(args.bank)==bank_meta['sha256']==identity['bank_hash']
    bank=torch.load(args.bank,map_location='cpu',mmap=True,weights_only=False)
    previous=torch.load(args.previous_bank,map_location='cpu',mmap=True,weights_only=False)
    for key in ('episode_ids','episode_hashes'):assert bank[key]==previous[key]
    for key in ('raw','chunks','routes','episode','time','mean','std'):
        a,b=bank[key],previous[key]
        assert a.shape==b.shape and a.dtype==b.dtype,key
        for offset in range(0,len(a),1024):assert torch.equal(a[offset:offset+1024],b[offset:offset+1024]),key
    assert len(bank['episode_ids'])==6222 and all(i.startswith('train/') for i in bank['episode_ids'])
    assert c['seeds']==[3072] and c['training']['global_batch']==64 and not c['execution']['test_enabled']
    assert complete['step'] <= 20000
    assert complete['stop_reason'] in ('update_limit','compute_cap')
    assert complete['validation_steps']==usage['validation_steps']
    assert 1 <= len(complete['validation_steps']) <= 4
    assert complete['validation_steps'][-1]==complete['step']
    assert usage['optimization_gpu_seconds']<=28800+usage['last_update_overrun_gpu_seconds']+1e-6
    assert usage['last_update_overrun_gpu_seconds']<8 and usage['gpu_count']==8
    assert abs(complete['optimization_gpu_hours']-usage['optimization_gpu_seconds']/3600)<1e-9
    checkpoints={}
    for name in ('last','best'):
        path=run/(name+'.pt');saved=torch.load(path,map_location='cpu',weights_only=False)
        for key in ('code','protocol','manifest','world_hash','bank_hash','method','seed','world_size'):
            assert saved[key]==identity[key],key
        assert not saved['fixture']
        assert all(torch.isfinite(v).all() for v in saved['model'].values())
        assert saved['best_step']==complete['best_step'] and saved['best']==complete['best']
        assert any(k.startswith('flow.') for k in saved['model'])
        assert not any(k.startswith('output.') for k in saved['model'])
        if name=='best':
            milestone=torch.load(run/'checkpoints'/f"step_{saved['step']:07d}.pt",map_location='cpu',weights_only=False)
            assert all(torch.equal(v,milestone['model'][k]) for k,v in saved['model'].items())
        checkpoints[name]=dict(step=saved['step'],sha256=file_hash(path),finite=True)
    logs=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    training=[x for x in logs if 'step' in x]
    assert [x['step'] for x in training]==[1]+list(range(50,complete['step']+1,50))
    assert [x['validation/checkpoint_step'] for x in logs if 'validation/checkpoint_step' in x]==complete['validation_steps']
    assert all(x['train/late_window_fraction']>=0 for x in training)
    assert all(x['train/execution_weight']==0 for x in training)
    assert all(x['train/reasoning_depth']==1+(x['step']-1)%3 for x in training)
    assert any(x['train/late_window_fraction']>0 for x in training)
    signatures=None;rounds=[]
    for step in complete['validation_steps']:
        report=json.loads((run/'validation'/f'step_{step:07d}.json').read_text())
        records=report['records'];assert len(records)==104
        current=sorted(tuple(r[k] for k in ('id','reset_seed','episode_sha256','model_seed')) for r in records)
        if signatures is None:signatures=current
        else:assert signatures==current
        assert all(x[0].startswith('validation/') for x in current)
        assert set(Counter(r['task'] for r in records).values())=={8}
        for record_row in records:
            ds=record_row['controller_diagnostics']['decisions']
            assert record_row['steps']<=200
            if not record_row['controller_budget_exhausted']:assert record_row['world_predictions']==480*len(ds)
            for i,d in enumerate(ds):
                assert d['candidate_world_transitions']==480
                assert d['scoring_context_fixed'] and d['flow_evaluations']==24
                assert d['reasoning_depth']==3
                assert d['fast_weights_fixed_during_search'] and d['fast_fit_batches_per_decision']==1
                assert d['fast_support_steps']==min(i,4)
                assert d['fast_weight_rank']==32 and d['fast_weight_ridge']==.125
                assert d['adaptation_scope']=='current_episode_executed_transitions'
                if i==0:assert d['fast_weight_delta_norm']==0
                rc=d['round_best_scores']
                assert len(rc)==3 and all(b<=a for a,b in zip(rc,rc[1:]))
                assert d['history_valid_steps']==min(i,4)
                assert d['warm_start_available']==(i>0)
                assert len(d['candidate_stall_penalties'])==8
                assert all(0<=v<=.020001 for v in d['candidate_stall_penalties'])
                scores=[a+b+s for a,b,s in zip(d['candidate_route_costs'],d['candidate_response_costs'],d['candidate_stall_penalties'])]
                # Serialized components are recombined as float64 here; actual
                # selection sums float32 tensors. Allow float32 rounding ties.
                assert scores[d['selected_anchor']]<=min(scores)+1e-7
                if i<len(ds)-1:assert 'actual_prefix_target_progress' in d
        count=sum(r['success'] for r in records)
        assert abs(count/104-report['metrics']['success_macro'])<1e-9
        rounds.append(dict(step=step,successes=count))
    selected=max(rounds,key=lambda x:(x['successes'],-x['step']))
    assert selected['step']==complete['best_step']
    source_files=json.loads((record/'runtime-files.json').read_text())
    assert source_files['commit']==identity['code']
    for path, expected in source_files['files'].items():assert file_hash(repo/path)==expected,path
    result=dict(runtime_file_checksums_verified=len(source_files['files']),source=identity['code'],run=identity,rounds=rounds,checkpoints=checkpoints,
        frozen_source_clean=True,data_and_world_identity_verified=True,identical_route_bank_tensors=True,
        training_metric_records=len(training),validation_metric_records=len(rounds),
        same_cases_across_rounds=True,causal_history_and_plan_reuse=True, episodic_fast_weights_and_reset_verified=True,
        score_reconstruction_verified=True,score_rounding_tolerance=1e-7,world_transition_budget_verified=True,
        training_coverage_logging_verified=True,compute=usage,complete=complete,
        final_tests_read=False,model_calls=0,simulator_calls=0)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(audit_passed=True,rounds=rounds,selected_step=selected['step'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('record','bank','previous-bank','output'):parser.add_argument('--'+name,required=True)
    main(parser.parse_args())
