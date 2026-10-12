"""Read-only final audit of the frozen repair-flow-TTT run (CPU only)."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import subprocess

import torch

from flow_jepa.common import digest, file_hash


def main(args):
    torch.set_num_threads(2)
    record=Path(args.record);repo=record/'repo';root=record/'campaign'
    method=getattr(args,'method','repair_flow_ttt')
    bayesian=method=='bayesian_repair_ttt'
    world_calls=280 if bayesian else 320
    depth_count=3 if bayesian else 2
    run=root/'runs'/f'{method}_3072'
    c=json.loads((repo/'config'/f'flow_metaworld_{method}.json').read_text())
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
    if bayesian:
        assert c['bayesian_repair_ttt']['training_rounds']==3
        assert c['flow_reasoning']['rounds']==2
        assert c['bayesian_repair_ttt']['acceptance_beta']==1.644854
    if method=='progress_objective_ttt':
        assert not c['progress_objective']['generated_outcome_labels']
        assert c['progress_objective']['score_weight']==.05
        assert c['progress_objective']['time_scale_blocks']==60
        assert c['progress_objective']['correction_bound']==.25
    assert complete['step'] <= 20000
    assert complete['stop_reason'] in ('update_limit','compute_cap')
    assert complete['validation_steps']==usage['validation_steps']
    assert 1 <= len(complete['validation_steps']) <= 4
    assert complete['validation_steps'][-1]==complete['step']
    assert usage['optimization_gpu_seconds']<=28800+usage['last_update_overrun_gpu_seconds']+1e-6
    assert usage['last_update_overrun_gpu_seconds']<8 and usage['gpu_count']==identity['world_size'] and usage['gpu_count'] in (4,8)
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
        assert any(k.startswith('repair_flow.') for k in saved['model'])
        assert not any(k.startswith('energy_base.') for k in saved['model'])
        assert not any(k.startswith('output.') for k in saved['model'])
        if bayesian:
            assert all(k in saved['model'] for k in ('prior_tril','log_noise_variance','fast_prior'))
            assert not any(k.startswith('goal_progress.') for k in saved['model'])
        if method=='progress_objective_ttt':
            assert any(k.startswith('goal_progress.') for k in saved['model'])
            assert 'progress_prior' in saved['model']
            assert 'progress_condition.weight' in saved['model']
        if name=='best':
            milestone=torch.load(run/'checkpoints'/f"step_{saved['step']:07d}.pt",map_location='cpu',weights_only=False)
            assert all(torch.equal(v,milestone['model'][k]) for k,v in saved['model'].items())
        checkpoints[name]=dict(step=saved['step'],sha256=file_hash(path),finite=True)
    logs=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    training=[x for x in logs if 'step' in x]
    assert [x['step'] for x in training]==[s for s in range(1,complete['step']+1) if s==1 or s%50 in ((0,1,2) if bayesian else (0,1))]
    assert [x['validation/checkpoint_step'] for x in logs if 'validation/checkpoint_step' in x]==complete['validation_steps']
    assert all(x['train/late_window_fraction']>=0 for x in training)
    assert all(x['train/execution_weight']==0 for x in training)
    assert all(x['train/reasoning_depth']==1+(x['step']-1)%depth_count for x in training)
    assert any(x['train/late_window_fraction']>0 for x in training)
    if method=='shifted_repair_ttt':
        deep=[x for x in training if x['train/reasoning_depth']==2]
        assert deep  # Zero shifted selections would be a valid negative result.
        for x in training:
            assert 0<=x['train/shifted_anchor_fraction']<=x['train/shifted_eligible_fraction']+1e-6
            if x['train/reasoning_depth']==1:assert x['train/shifted_anchor_fraction']==0
    signatures=None;rounds=[]
    for step in complete['validation_steps']:
        report=json.loads((run/'validation'/f'step_{step:07d}.json').read_text())
        for key in ('code','protocol','manifest','world_hash','bank_hash','method','seed','world_size'):
            assert report[key]==identity[key],key
        milestone_path=run/'checkpoints'/f'step_{step:07d}.pt'
        milestone=torch.load(milestone_path,map_location='cpu',weights_only=False)
        assert milestone['step']==step and not milestone['fixture']
        for key in ('code','protocol','manifest','world_hash','bank_hash','method','seed','world_size'):
            assert milestone[key]==identity[key],key
        assert all(torch.isfinite(v).all() for v in milestone['model'].values())
        checkpoints[f'step_{step:07d}']=dict(step=step,sha256=file_hash(milestone_path),finite=True)
        records=report['records'];assert len(records)==104
        current=sorted(tuple(r[k] for k in ('id','reset_seed','episode_sha256','model_seed')) for r in records)
        if signatures is None:signatures=current
        else:assert signatures==current
        assert all(x[0].startswith('validation/') for x in current)
        assert set(Counter(r['task'] for r in records).values())=={8}
        for record_row in records:
            ds=record_row['controller_diagnostics']['decisions']
            assert record_row['steps']<=200
            assert record_row['controller_budget_seconds']==10
            if not record_row['controller_budget_exhausted']:assert record_row['world_predictions']==world_calls*len(ds)
            for i,d in enumerate(ds):
                assert d['candidate_world_transitions']==world_calls
                assert d['scoring_context_fixed'] and d['flow_evaluations']==12
                assert d['reasoning_depth']==2
                assert d['fast_weights_fixed_during_search'] and d['fast_fit_batches_per_decision']==1
                assert d['fast_support_steps']==min(i,4)
                assert d['fast_weight_rank']==32
                if not bayesian:assert d['fast_weight_ridge']==.125
                assert d['adaptation_scope']=='current_episode_executed_transitions'
                if i==0:assert d['fast_weight_delta_norm']==0
                assert d['surrogate_gradient_evaluations']==0 and d['repair_incumbent_preserved']
                assert d['repair_solver_steps']==4 and d['repair_radius']==.25
                assert d['selected_distance_from_incumbent_linf']<=.250001
                if not bayesian:assert d['refinement_source']=='world_verified_incumbent'
                else:
                    from scripts.analysis.bayesian_repair_details import check_decision
                    check_decision(d)
                if method in ('retrieval_repair_ttt','shifted_repair_ttt','progress_objective_ttt','bayesian_repair_ttt'): assert d['repair_anchor_source'] in ('generated','retrieved','shifted')
                if method=='progress_objective_ttt':
                    assert d['progress_fit_scope']=='past_executed_prefixes_only_fixed_during_search'
                    if i==0:assert d['goal_progress_fast_delta_norm']==0
                    corrections=d['candidate_goal_progress_corrections']
                    penalties=d['candidate_stage_penalties']
                    advances=d['candidate_target_stage_advances']
                    assert len(corrections)==len(penalties)==len(advances)==8
                    assert all(math.isfinite(v) and abs(v)<=.250001 for v in corrections)
                    for advance,penalty in zip(advances,penalties):
                        expected=.05*(max(0,-advance)+max(0,advance-5/60))
                        assert abs(penalty-expected)<1e-7
                    selected_correction=corrections[d['selected_anchor']]
                    assert abs(d['goal_progress_error_correction']-selected_correction)<1e-7
                    assert abs(d['goal_progress_corrected']-(d['goal_progress_predicted']-max(0,selected_correction)))<1e-7
                    if i<len(ds)-1:
                        assert 'goal_progress_observed' in d
                        assert math.isfinite(d['goal_progress_observed'])
                rc=d['round_best_scores']
                assert len(rc)==2 and all(b<=a for a,b in zip(rc,rc[1:]))
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
        progress_score_and_causal_fit_verified=(method=='progress_objective_ttt'),
        posterior_acceptance_and_cached_incumbents_verified=bayesian,
        final_tests_read=False,model_calls=0,simulator_calls=0)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(audit_passed=True,rounds=rounds,selected_step=selected['step'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('record','bank','previous-bank','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--method',choices=['repair_flow_ttt','retrieval_repair_ttt','shifted_repair_ttt','progress_objective_ttt','bayesian_repair_ttt'],default='repair_flow_ttt')
    main(parser.parse_args())
