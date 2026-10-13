"""Resume missing baseline training, then run all six frozen refinements.

Default is plan-only. --execute uses the existing NVIDIA server, never cloud.
--train-missing-baselines explicitly delegates to the existing four-GPU campaign;
completed baseline runs are preserved. New inference uses one exclusively locked
GPU. --publish pushes each completed comparison from a separate worktree.
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

from flow_jepa.common import save_json
from .runtime import variants


def publish(source, output):
    pub=output/'publication'
    subprocess.run(['git','fetch','origin','main'],cwd=source,check=True)
    if not pub.exists():
        subprocess.run(['git','worktree','add','--detach',str(pub),'origin/main'],cwd=source,check=True)
    else:
        if subprocess.check_output(['git','status','--porcelain'],cwd=pub,text=True).strip():
            raise RuntimeError('Uncommitted publication changes must be reconciled')
        subprocess.run(['git','rebase','origin/main'],cwd=pub,check=True)
    target=pub/'docs/reports'/output.name
    for path in output.rglob('*.json'):
        if any(x in path.relative_to(output).parts for x in ('publication','runtime')):continue
        dest=target/path.relative_to(output);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)
    rel=str(target.relative_to(pub))
    subprocess.run(['git','add','--',rel],cwd=pub,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=pub).returncode:
        subprocess.run(['git','commit','-m','Publish frozen refinement comparisons and failure evidence'],cwd=pub,check=True)
    subprocess.run(['git','push','origin','HEAD:main'],cwd=pub,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-workspace','output','data-root'):p.add_argument('--'+name,required=True)
    p.add_argument('--tasks',nargs='+',choices=('pusht','cube'),default=['pusht','cube'])
    p.add_argument('--train-missing-baselines',action='store_true')
    p.add_argument('--execute',action='store_true');p.add_argument('--publish',action='store_true')
    a=p.parse_args();base=Path(a.baseline_workspace).resolve();out=Path(a.output).resolve()
    source=Path(__file__).resolve().parents[1];out.mkdir(parents=True,exist_ok=True)
    a.data_root=str(Path(a.data_root).resolve())
    if base==out:raise ValueError('Refinement output must not overwrite baseline workspace')
    plan=dict(tasks=a.tasks,variants=list(variants()),baseline_workspace=str(base),
        training='reuse_completed; train_missing_only_if_explicit' if a.train_missing_baselines else 'reuse_only',
        new_method_optimizer_updates=0,sealed_tests_enabled=False,execution_requested=a.execute)
    save_json(out/'plan.json',plan)
    if not a.execute:print(json.dumps(plan,indent=2));return
    if not shutil.which('nvidia-smi'):
        save_json(out/'status.json',dict(status='blocked_before_launch',reason='no_existing_NVIDIA_server_runtime',
                  training_started=False,evaluation_started=False))
        raise RuntimeError('Existing GPU server unavailable; no training or evaluation launched')
    lock=(out/'campaign.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    held=[]
    def status(value,**kw):save_json(out/'status.json',dict(status=value,**kw))
    try:
        if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=source,text=True).strip():
            raise RuntimeError('Commit source before launching')
        sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
        frozen=out/'runtime'
        if not frozen.exists():
            subprocess.run(['git','worktree','add','--detach',str(frozen),sha],cwd=source,check=True)
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=frozen,text=True).strip()!=sha:
            raise RuntimeError('Resume from the frozen runtime revision, not a different checkout')
        if a.train_missing_baselines:
            cmd=[sys.executable,'-m','crossbench.campaign','--workspace',str(base),
                '--data-root',a.data_root,'--gpus','4','--execute','--tasks',*a.tasks]
            if a.publish:cmd.append('--publish')
            status('baseline_campaign',command=cmd)
            baseline_runtime = base/'runtime'
            training_source = baseline_runtime if baseline_runtime.exists() else frozen
            subprocess.run(cmd,cwd=training_source,check=True)
        for task in a.tasks:
            required=[base/'cache'/task/'manifest.json',base/'runs'/task/'leflow'/'best.pt',
                      base/'runs'/task/'leflow'/'complete.json']
            missing=[str(x) for x in required if not x.exists()]
            if missing:raise FileNotFoundError('Required trained baseline artifacts missing: '+', '.join(missing))
        from flow_jepa.campaign import acquire_gpus
        selected,held=acquire_gpus(1,out,wait_hours=1,required=1)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=selected[0][0],OMP_NUM_THREADS='2',
            STABLEWM_HOME=a.data_root,MUJOCO_GL='egl')
        for task in a.tasks:
            for name in variants():
                dest=out/task/name
                cmd=[sys.executable,'-m','leflow_refinement.crossbench','--cache',str(base/'cache'/task),
                    '--checkpoint',str(base/'runs'/task/'leflow'/'best.pt'),'--output',str(dest),
                    '--variant',name,'--device','cuda:0']
                status('evaluating',task=task,variant=name,command=cmd)
                log=out/'logs'/f'{task}-{name}.log';log.parent.mkdir(exist_ok=True)
                with log.open('a') as f:subprocess.run(cmd,cwd=frozen,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
                from .compare import paired
                current=json.loads((dest/'report.json').read_text())
                for bp in (base/'runs'/task).glob('*/best.json'):
                    best=json.loads(bp.read_text());rp=bp.parent/'validation'/f'step_{best["step"]:07d}'/'report.json'
                    if not rp.exists():continue
                    result=paired(json.loads(rp.read_text()),current)
                    save_json(out/'comparisons'/task/f'{name}-vs-{bp.parent.name}.json',result)
                status('variant_completed',task=task,variant=name,metrics=current['metrics'])
                if a.publish:publish(source,out)
        status('completed',new_method_training_updates=0,comparisons=len(a.tasks)*len(variants()))
        if a.publish:publish(source,out)
    except BaseException as error:
        status('failed',error=repr(error))
        if a.publish:
            try:publish(source,out)
            except Exception as e:save_json(out/'publication_error.json',dict(error=repr(e)))
        raise
    finally:
        for handle in held:handle.close()
        lock.close()


if __name__=='__main__':main()
