"""Sixteen sequential native jobs, frozen source, no MetaWorld retraining.

Default is a no-compute plan. --execute starts formal runs on the GPU server.
With --publish, each completed method's factual reports are committed/pushed
from a separate publication worktree; large datasets/checkpoints never enter Git.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from flow_jepa.common import save_json, file_hash
from .spec import METHODS, matrix


def command(argv,cwd,env=None,log=None):
    with open(log,'a') if log else open(os.devnull,'w') as f:
        subprocess.run(argv,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)


def publish(source,workspace):
    from .report import collect
    collect(workspace,source)
    pub=workspace/'publication'
    subprocess.run(['git','fetch','origin','main'],cwd=source,check=True)
    if not pub.exists():
        subprocess.run(['git','worktree','add','--detach',str(pub),'origin/main'],cwd=source,check=True)
    else:
        # A previous rejected push remains recoverable instead of being reset.
        if subprocess.check_output(['git','status','--porcelain'],cwd=pub,text=True).strip():
            raise RuntimeError('Publication worktree has uncommitted changes; inspect before retrying')
        subprocess.run(['git','rebase','origin/main'],cwd=pub,check=True)
    target=pub/'docs/reports'/workspace.name
    for directory in ('analysis','runs'):
        for path in (workspace/directory).rglob('*'):
            if not path.is_file() or 'wandb' in path.parts:continue
            if path.suffix not in ('.json','.jsonl','.npz'):continue
            # Full trajectories stay on the server; publish ten fixed cases per milestone.
            if 'trajectories' in path.parts and path.stem.isdigit() and int(path.stem)>=10:continue
            destination=target/path.relative_to(workspace);destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,destination)
    target.mkdir(parents=True,exist_ok=True)
    for name in ('plan.json','status.json'):
        if (workspace/name).exists():shutil.copy2(workspace/name,target/name)
    rel=str(target.relative_to(pub))
    subprocess.run(['git','add','--',rel],cwd=pub,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=pub).returncode:
        subprocess.run(['git','commit','-m','Publish cross-benchmark curves, cases and failure diagnostics'],cwd=pub,check=True)
    subprocess.run(['git','push','origin','HEAD:main'],cwd=pub,check=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=pub,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=pub,text=True).split()[0]
    if head!=remote:raise RuntimeError('Remote publication readback mismatch')
    save_json(workspace/'publication-receipt.json',dict(commit=head,verified_remote=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',required=True);p.add_argument('--data-root',required=True)
    p.add_argument('--gpus',type=int,choices=[1,4],default=4)
    p.add_argument('--execute',action='store_true');p.add_argument('--publish',action='store_true')
    p.add_argument('--tasks',nargs='+',choices=['pusht','cube'],default=['pusht','cube'])
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    a=p.parse_args();workspace=Path(a.workspace).resolve();data_root=Path(a.data_root).resolve()
    source=Path(__file__).resolve().parents[1];workspace.mkdir(parents=True,exist_ok=True)
    manifest=matrix();manifest['jobs']=[j for j in manifest['jobs'] if j['dataset'] in a.tasks and j['method'] in a.methods]
    save_json(workspace/'plan.json',manifest)
    if not a.execute:
        print(json.dumps(manifest,indent=2));return
    if not shutil.which('nvidia-smi'):
        save_json(workspace/'status.json',dict(status='blocked_before_launch',reason='no_nvidia_runtime',training_started=False))
        raise RuntimeError('No NVIDIA runtime; no training launched')
    lock=(workspace/'campaign.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=source,text=True).strip():
        raise RuntimeError('Freeze a clean committed source before launching')
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    frozen=workspace/'runtime'
    if not frozen.exists():subprocess.run(['git','worktree','add','--detach',str(frozen),sha],cwd=source,check=True)
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=frozen,text=True).strip()
    if actual!=sha:raise RuntimeError('Existing campaign uses another source; resume from its frozen runtime checkout')
    from flow_jepa.campaign import acquire_gpus
    selected,held=acquire_gpus(a.gpus,workspace,wait_hours=1,required=a.gpus)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=','.join(v[0] for v in selected),
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',STABLEWM_HOME=str(data_root),WANDB_MODE='online')
    logs=workspace/'logs';logs.mkdir(exist_ok=True)
    def status(value,**kw):save_json(workspace/'status.json',dict(status=value,source=sha,**kw))
    try:
        for task in a.tasks:
            source_json=data_root/'prepared'/task/'source.json'
            if not source_json.exists():
                status('preparing_source',dataset=task)
                command([sys.executable,'scripts/prepare_data.py','--task',task],frozen,env,logs/f'{task}-source.log')
            cache=workspace/'cache'/task
            args=[sys.executable,'-m','crossbench.prepare','--task',task,'--source-json',str(source_json),'--output',str(cache)]
            existing=data_root/'latents'/task/'manifest.json'
            if existing.exists():args+=['--latent-manifest',str(existing)]
            status('preparing_cache',dataset=task);command(args,frozen,env,logs/f'{task}-cache.log')
            for method in a.methods:
                run=workspace/'runs'/task/method;run.mkdir(parents=True,exist_ok=True)
                if (run/'complete.json').exists():
                    from .report import collect
                    collect(workspace,source)
                    if a.publish:publish(source,workspace)
                    continue
                status('native_preflight',dataset=task,method=method)
                one=env.copy();one['CUDA_VISIBLE_DEVICES']=selected[0][0]
                command([sys.executable,'-m','crossbench.preflight','--task',task,'--method',method,
                         '--cache',str(cache),'--output',str(run/'native-preflight.json')],frozen,one,logs/f'{task}-{method}-preflight.log')
                args=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={a.gpus}',
                      '-m','crossbench.train','--task',task,'--method',method,'--cache',str(cache),'--run-dir',str(run)]
                status('training_and_evaluating',dataset=task,method=method,command=args)
                command(args,frozen,env,logs/f'{task}-{method}.log')
                from .report import collect
                collect(workspace,source)
                if a.publish:status('publishing',dataset=task,method=method);publish(source,workspace)
        status('completed',new_runs=len(a.tasks)*len(a.methods),metaworld='reused_without_retraining')
    except BaseException as error:
        status('failed',error=repr(error))
        if a.publish:
            try:publish(source,workspace)
            except Exception as publication_error:
                save_json(workspace/'publication-error.json',dict(error=repr(publication_error)))
        raise
    finally:
        for f in held:f.close()
        lock.close()


if __name__=='__main__':main()
