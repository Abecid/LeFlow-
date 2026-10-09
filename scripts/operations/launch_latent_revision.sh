#!/usr/bin/env bash
set -euo pipefail
record=/home/mtxu/adam/LeFlow-experiments/20261009-latent-revision
runtime=/tmp/mtxu-flow-jepa-20261007/env
cd "$record/repo"
export CUDA_MODULE_LOADING=LAZY OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export WANDB_MODE=online WANDB_ENTITY=attentionx2023
export LD_LIBRARY_PATH="$runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=egl FLOW_EGL_DEVICES=0,0,0,0,0,0,0,0
export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe LP_NUM_THREADS=2
export __EGL_VENDOR_LIBRARY_FILENAMES="$runtime/share/glvnd/egl_vendor.d/50_mesa.json"
# CPU-only renderer check on a registered TRAIN reset, before spending GPUs.
"$runtime/bin/python" - <<'PY'
import json
from pathlib import Path
import h5py
import numpy as np
from flow_jepa.environment import make_env
p=Path('/home/mtxu/adam/LeFlow-experiments/20261009-repaired-comparison/campaign')
c=json.loads(Path('config/flow_metaworld_revision.json').read_text())
m=json.loads((p/'manifest.json').read_text())
r=next(x for x in m['entries'] if x['split']=='train')
with h5py.File(p/r['path']) as f:initial=f['initial_rgb'][:]
env=make_env(r['task'],r['seed'],c['data'])
try:
    env.reset()
    assert np.array_equal(env.render(),initial), 'Renderer changed registered RGB'
finally:
    env.close()
print('Training reset RGB exactly matches registered cache',flush=True)
PY
exec "$runtime/bin/python" -u -m scripts.operations.controller_grounded \
  --record "$record" --variant latent_revision \
  --previous /home/mtxu/adam/LeFlow-experiments/20261009-repaired-comparison \
  --cache /tmp/mtxu-latent-revision-20261009
