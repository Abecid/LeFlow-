#!/usr/bin/env bash
set -euo pipefail
record=/home/mtxu/adam/LeFlow-experiments/20261009-controller-grounded
runtime=/tmp/mtxu-flow-jepa-20261007/env
cd "$record/repo"
export CUDA_MODULE_LOADING=LAZY OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export WANDB_MODE=online WANDB_ENTITY=attentionx2023
export LD_LIBRARY_PATH="$runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=egl FLOW_EGL_DEVICES=0,0,0,0,0,0,0,0
export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe LP_NUM_THREADS=2
export __EGL_VENDOR_LIBRARY_FILENAMES="$runtime/share/glvnd/egl_vendor.d/50_mesa.json"
# Reuse the exact Mesa renderer used to collect the registered RGB images.
# This CPU check resets one training case; it never steps/evaluates a policy.
"$runtime/bin/python" - <<'PY'
import json
from pathlib import Path
import h5py
import numpy as np
from flow_jepa.environment import make_env
p=Path('/home/mtxu/adam/LeFlow-experiments/20261009-controller-grounded')
c=json.loads((p/'repo/config/flow_metaworld_execution.json').read_text())
m=json.loads((p/'campaign/manifest.json').read_text())
r=next(x for x in m['entries'] if x['split']=='train')
with h5py.File(p/'campaign'/r['path']) as f:
    initial=f['initial_rgb'][:] if 'initial_rgb' in f else f['rgb'][0]
env=make_env(r['task'],r['seed'],c['data'])
try:
    env.reset()
    assert np.array_equal(env.render(),initial), 'Renderer changed registered RGB'
finally:
    env.close()
print('Training reset RGB exactly matches registered cache',flush=True)
PY
exec "$runtime/bin/python" -u -m scripts.operations.controller_grounded \
  --record "$record" \
  --previous /home/mtxu/adam/LeFlow-experiments/20261009-repaired-comparison \
  --cache /tmp/mtxu-controller-grounded-20261009
