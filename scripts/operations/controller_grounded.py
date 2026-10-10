"""Isolated one-run coordinator; never dispatches baselines, variants or tests."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from flow_jepa.common import config, digest, file_hash, git_revision, save_json
from flow_jepa.campaign import adopt_data_manifest, acquire_gpus


def main(a):
    record, previous, cache = Path(a.record), Path(a.previous), Path(a.cache)
    record.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
    lock=(cache/'coordinator.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    root, repo = record/'campaign', record/'repo'
    config_name={'controller_grounded':'flow_metaworld_execution.json',
                 'latent_revision':'flow_metaworld_revision.json',
                 'execution_revision':'flow_metaworld_aligned.json',
                 'flow_reasoning':'flow_metaworld_flow_reasoning.json'}[a.variant]
    c=config(repo/'config'/config_name)
    world=previous/'campaign/runs/world_3072/best.pt'
    run_dir=root/'runs'/f'{a.variant}_3072'
    state=record/'status.json'
    began=time.perf_counter()

    def status(value,**fields):
        save_json(state,dict(status=value,code=git_revision(),pid=os.getpid(),
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**fields))

    if not root.exists():
        root.mkdir()
        for name in ('episodes','weights','vendor'):
            (root/name).symlink_to((previous/'campaign'/name).resolve(),target_is_directory=True)
        shutil.copy2(previous/'campaign/manifest.json',root/'manifest.json')
        adopt_data_manifest(root,config(previous/'repo/config/flow_metaworld_repair.json'),c)
        before=json.loads((previous/'campaign/manifest.json').read_text())
        after=json.loads((root/'manifest.json').read_text())
        for key in ('entries','encoder','mean','std','goal_screening','expert_success'):
            assert before[key]==after[key],key
        assert file_hash(world)==c['controller_grounded']['world_sha256']
        save_json(root/'reuse.json',dict(source_root=str(previous/'campaign'),
            world_source_manifest=file_hash(previous/'campaign/manifest.json'),
            entries_sha256=digest(before['entries']),encoder_sha256=before['encoder'],
            world_sha256=file_hash(world),unchanged_fields=['entries','encoder','mean','std','goal_screening','expert_success'],
            new_trainable_components=a.variant+' fresh policy only',validation_cases=104,
            seed=3072,test_enabled=False,source_code=git_revision()))
    cfg=str(repo/'config'/config_name)
    bank=str(cache/'route_bank.pt')
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',FLOW_DEVICE='cuda',CUDA_MODULE_LOADING='LAZY')
    status('cpu_verification')
    with (record/'tests.log').open('a') as log:
        subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_execution.py','-v'],
                       env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        if a.variant in ('latent_revision','execution_revision','flow_reasoning'):
            subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_latent_revision.py','-v'],
                           env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        if a.variant in ('execution_revision','flow_reasoning'):
            subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_execution_revision.py','-v'],
                           env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        if a.variant=='flow_reasoning':
            subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_flow_reasoning.py','-v'],
                           env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    if not Path(bank).exists():
        if a.reuse_bank:
            status('verifying_and_reusing_train_only_route_index')
            from flow_jepa.execution.bank import reuse_bank
            reuse_bank(a.reuse_bank, bank, c, root)
        else:
            status('building_train_only_route_index')
            with (record/'bank.log').open('a') as log:
                subprocess.run([sys.executable,'-u','-m','flow_jepa.execution.bank','--config',cfg,
                    '--root',str(root),'--output',bank],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    shutil.copy2(bank+'.json',root/'route-bank.json')
    status('acquiring_idle_gpus')
    selected, held=acquire_gpus(8,root,wait_hours=1,required=8)
    assert len(selected)==8
    save_json(root/'allocation.json',dict(gpus=selected,global_batch=64,optimization_gpu_seconds=28800))
    try:
        if not (root/'preflight.json').exists():
            status('bounded_training_data_preflight')
            one=env.copy();one['CUDA_VISIBLE_DEVICES']=selected[0][0]
            with (record/'preflight.log').open('a') as log:
                subprocess.run([sys.executable,'-u','-m','flow_jepa.execution.preflight',
                    '--config',cfg,'--root',str(root),'--world',str(world),'--bank',bank,
                    '--output',str(root/'preflight.json')],env=one,stdout=log,stderr=subprocess.STDOUT,check=True)
        preflight=json.loads((root/'preflight.json').read_text())
        # A failed latency gate is an implementation issue, not permission to
        # relax the controller allowance or quietly run an expensive sweep.
        if preflight['mean_ms']>120:
            raise RuntimeError(f"Full-controller preflight exceeds120ms: {preflight['mean_ms']}")
        env['CUDA_VISIBLE_DEVICES']=','.join(x[0] for x in selected)
        cmd=[sys.executable,'-m','torch.distributed.run','--nnodes=1','--nproc_per_node=8',
             '--rdzv-backend=c10d','--rdzv-endpoint=127.0.0.1:0','-m','flow_jepa.execution.train',
             '--config',cfg,'--root',str(root),'--run-dir',str(run_dir),'--world',str(world),'--bank',bank]
        status('training',command=cmd,preflight_mean_ms=preflight['mean_ms'])
        with (record/'training.log').open('a') as log:
            child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            save_json(record/'worker.json',dict(pid=child.pid,command=cmd))
            rc=child.wait()
            if rc:raise RuntimeError(f'Training process exited {rc}')
        status('completed_validation_ready_for_analysis',run_dir=str(run_dir),wall_seconds=time.perf_counter()-began)
    finally:
        for handle in held:handle.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('record','previous','cache'):p.add_argument('--'+name,required=True)
    p.add_argument('--variant',choices=['controller_grounded','latent_revision','execution_revision','flow_reasoning'],default='controller_grounded')
    p.add_argument('--reuse-bank')
    args=p.parse_args()
    try:main(args)
    except Exception as error:
        save_json(Path(args.record)/'failure.json',dict(error=repr(error),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        raise
