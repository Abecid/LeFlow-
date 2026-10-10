"""Launch one LeFlow-TTT candidate on eight idle GPUs and seal its artifacts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ..campaign import acquire_gpus
from ..common import config, file_hash, git_revision, save_json


def main(a):
    c = config(a.config); root, out = Path(a.root), Path(a.run_dir)
    out.mkdir(parents=True, exist_ok=True)
    lock=(out/'coordinator.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert c['primary_method']=='leflow_ttt' and c['execution']['training_enabled']
    if (out/'registry.json').exists():
        registry = json.loads((out/'registry.json').read_text())
        for path, expected in registry['files'].items():
            assert file_hash(out/path) == expected, path
        print('Verified sealed candidate; no rerun.', flush=True)
        return
    dirty = subprocess.check_output(['git','status','--porcelain'],text=True)
    if dirty: raise RuntimeError('Formal candidate requires an immutable clean checkout')
    devices, held = acquire_gpus(8, out, 1., required=8)
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = ','.join(x[0] for x in devices)
    env['FLOW_DEVICE'] = 'cuda'
    env['FLOW_EGL_DEVICES'] = '0,0,0,0,0,0,0,0'
    env['MUJOCO_EGL_DEVICE_ID'] = '0'
    for k,v in dict(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_MODULE_LOADING='LAZY',
        WANDB_MODE='online',WANDB_ENTITY='attentionx2023',MUJOCO_GL='egl',
        LIBGL_ALWAYS_SOFTWARE='1',GALLIUM_DRIVER='llvmpipe',LP_NUM_THREADS='2').items(): env[k]=v
    runtime = Path(sys.executable).resolve().parent.parent
    env['LD_LIBRARY_PATH'] = str(runtime/'lib') + ':' + env.get('LD_LIBRARY_PATH','')
    env['__EGL_VENDOR_LIBRARY_FILENAMES'] = str(runtime/'share/glvnd/egl_vendor.d/50_mesa.json')
    module = 'train'
    command = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
        '-m','flow_jepa.leflow_ttt.'+module,'--config',str(Path(a.config).resolve()),
        '--root',str(root.resolve()),'--run-dir',str(out.resolve()),'--world',str(Path(a.world).resolve())]
    runtime_files=sorted(set(p for folder in ('flow_jepa','scripts/operations/baseline_runtime_guard') for p in Path(folder).rglob('*.py')))
    save_json(out/'runtime-source.json',dict(code=git_revision(),files={str(p):file_hash(p) for p in runtime_files}))
    save_json(out/'launch.json',dict(code=git_revision(),command=command,devices=devices,
        config_sha256=file_hash(a.config),world_sha256=file_hash(a.world),pid=os.getpid()))
    began=time.time()
    with (out/'worker.log').open('a') as log:
        child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        save_json(out/'worker.json',dict(pid=child.pid,command=command))
        result=subprocess.CompletedProcess(command,child.wait())
    save_json(out/'process-exit.json',dict(returncode=result.returncode,wall_seconds=time.time()-began))
    if result.returncode: raise RuntimeError('Candidate worker failed; inspect worker.log')
    complete=json.loads((out/'complete.json').read_text())
    if module=='train':
        assert len(complete['validation_steps']) == 4
        reports=[out/'validation'/f'step_{s:07d}.json' for s in complete['validation_steps']]
        paths=[out/'best.pt',out/'last.pt',out/'compute_usage.json',out/'complete.json']+reports
        paths+=sorted((out/'checkpoints').glob('*.pt'))
    else: reports=[out/'validation.json']; paths=reports+[out/'complete.json']
    for path in reports:
        r=json.loads(path.read_text()); assert len(r['records']) == 104
        assert len({x['id'] for x in r['records']}) == 104
        assert r['code'] == git_revision()
        assert all(x['steps']<=200 for x in r['records'])
    save_json(out/'registry.json',dict(code=git_revision(),method=c['primary_method'],seed=3072,
        configuration=c,config_sha256=file_hash(a.config),world_sha256=file_hash(a.world),
        files={str(p.relative_to(out)):file_hash(p) for p in paths},
        test_files_read=0,policy='Preserve this completed candidate; no automatic new training, variants or tests.'))
    for handle in held: handle.close()


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for k in ('config','root','run-dir','world'): p.add_argument('--'+k,required=True)
    main(p.parse_args())
