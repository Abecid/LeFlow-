"""Final CPU audit of immutable LeFlow-TTT execution and baseline identity."""
import argparse,json,subprocess,hashlib
from pathlib import Path
import torch


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def load(path):return json.loads(Path(path).read_text())


def main(a):
    torch.set_num_threads(2)
    record=Path(a.record);repo=record/'repo';root=record/'campaign';run=root/'runs/leflow_ttt_3072'
    registry=load(run/'registry.json');complete=load(run/'complete.json');usage=load(run/'compute_usage.json')
    assert load(run/'process-exit.json')['returncode']==0
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    assert revision==registry['code']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    runtime=load(run/'runtime-source.json')
    for p,expected in runtime['files'].items():assert sha(repo/p)==expected,p
    for p,expected in registry['files'].items():assert sha(run/p)==expected,p
    baseline=Path('/home/mtxu/adam/LeFlow-experiments/20261010-baseline-release')
    c=registry['configuration'];native=load(baseline/'repo/config/flow_metaworld_leflow_release.json')
    for k in ('data','encoder','training_tasks','heldout_tasks','seeds','evaluation'):
        assert c[k]==native[k],k
    assert c['baseline']['leflow']==native['baseline']['leflow']
    assert c['training']==native['training']
    assert not c['execution']['test_enabled']
    m=load(root/'manifest.json');old=load(baseline/'leflow-data/manifest.json')
    for k in ('entries','encoder','mean','std','goal_screening','expert_success'):assert m[k]==old[k],k
    assert complete['step']==20000 and complete['new_optimizer_updates']==18000
    assert complete['validation_steps']==[5000,10000,15000,20000]
    assert usage['optimization_gpu_seconds']<=28800 and usage['global_batch']==64 and usage['gpu_count']==8
    source=Path(c['leflow_ttt']['adapter_source'])
    assert sha(source)==complete['common_adapter']['source_checkpoint_sha256']
    original=torch.load(source,map_location='cpu',weights_only=False)
    best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
    for name,saved in [('best',best),('last',last)]:
        assert saved['code']==revision and saved['world_hash']==c['baseline']['world_sha256']
        assert saved['method']=='leflow_ttt' and saved['seed']==3072 and not saved['fixture']
        assert saved['common_adapter']==complete['common_adapter']
        for k,v in original['model'].items():
            if k.startswith('adapter.'):assert torch.equal(v,saved['model'][k]),k
        point=torch.load(run/f"checkpoints/step_{saved['step']:07d}.pt",map_location='cpu',weights_only=False)
        assert all(torch.equal(v,point['model'][k]) for k,v in saved['model'].items()),name
        assert all(torch.isfinite(v).all() for v in saved['model'].values())
    assert best['step']==complete['best_step'] and last['step']==20000
    reference=load(baseline/'leflow_3072/validation/step_0010000.json')
    refs={r['id']:r for r in reference['records']};scores=[]
    for step in complete['validation_steps']:
        report=load(run/f'validation/step_{step:07d}.json')
        assert len(report['records'])==104
        for r in report['records']:
            for k in ('episode_sha256','reset_seed','model_seed','task'):assert r[k]==refs[r['id']][k]
            ds=r['controller_diagnostics']['decisions']
            for i,d in enumerate(ds):
                assert d['history_chunks']==min(i,4) and d['inner_steps']==4 and d['world_transitions']==325
                if i==0:assert d['fast_weight_delta']==0
            assert r['steps']<=200 and r['controller_budget_seconds']==10
        scores.append((step,sum(bool(r['success']) for r in report['records'])))
    chosen=max(scores,key=lambda x:(x[1],-x[0]));assert chosen[0]==best['step']
    world=Path('/home/mtxu/adam/LeFlow-experiments/20261009-repaired-comparison/campaign/runs/world_3072/best.pt')
    assert sha(world)==c['baseline']['world_sha256']
    import wandb
    info=load(run/'run.json');online=wandb.Api().run('attentionx2023/flow-jepa-metaworld/'+info['id'])
    assert online.state=='finished'
    result=dict(source=revision,runtime_files_verified=len(runtime['files']),
        registry_files_verified=len(registry['files']),scores=scores,selected_step=chosen[0],
        best_last_match_registered_checkpoints=True,exact_baseline_adapter_unchanged=True,
        common_adapter=complete['common_adapter'],data_fields_identical=True,
        successful_expert_train_episodes=sum(r['split']=='train' and r['mode']=='expert' and r['expert_success'] for r in m['entries']),
        baseline_schedule_unchanged=True,world_hash_verified=True,case_pairing_verified=True,
        adaptation_episode_local=True,optimization_gpu_seconds=usage['optimization_gpu_seconds'],
        validation_gpu_seconds=usage['validation_gpu_seconds'],wandb_state=online.state,wandb_url=info['wandb_url'],
        model_calls=0,simulator_calls=0,test_files_read=0)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--record',required=True);p.add_argument('--output',required=True)
    main(p.parse_args())
