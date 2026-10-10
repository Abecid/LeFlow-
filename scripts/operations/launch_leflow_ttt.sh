#!/usr/bin/env bash
set -euo pipefail
record=/home/mtxu/adam/LeFlow-experiments/20261010-leflow-ttt
runtime=/tmp/mtxu-flow-jepa-20261007/env
baseline=/home/mtxu/adam/LeFlow-experiments/20261010-baseline-release
world=/home/mtxu/adam/LeFlow-experiments/20261009-repaired-comparison/campaign/runs/world_3072/best.pt
cd "$record/repo"
export CUDA_MODULE_LOADING=LAZY OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export WANDB_MODE=online WANDB_ENTITY=attentionx2023
export LD_LIBRARY_PATH="$runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=egl FLOW_EGL_DEVICES=0,0,0,0,0,0,0,0
export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe LP_NUM_THREADS=2
export __EGL_VENDOR_LIBRARY_FILENAMES="$runtime/share/glvnd/egl_vendor.d/50_mesa.json"
"$runtime/bin/python" -m flow_jepa.baselines.prepare \
 --config config/flow_metaworld_leflow_ttt.json --source "$baseline/leflow-data" \
 --root "$record/campaign" --world "$world" \
 --source-config "$baseline/repo/config/flow_metaworld_leflow_release.json"
"$runtime/bin/python" - <<'CHECK'
import json
from pathlib import Path
import h5py,numpy as np
from flow_jepa.environment import make_env
p=Path('/home/mtxu/adam/LeFlow-experiments/20261010-leflow-ttt/campaign')
c=json.loads(Path('config/flow_metaworld_leflow_ttt.json').read_text())
m=json.loads((p/'manifest.json').read_text());r=next(x for x in m['entries'] if x['split']=='train')
with h5py.File(p/r['path']) as f:initial=f['initial_rgb'][:]
env=make_env(r['task'],r['seed'],c['data'])
try:
 env.reset();assert np.array_equal(env.render(),initial),'Renderer changed registered RGB'
finally:env.close()
print('Training reset RGB exactly matches registered cache',flush=True)
CHECK
exec "$runtime/bin/python" -u -m flow_jepa.leflow_ttt.launch \
 --config config/flow_metaworld_leflow_ttt.json --root "$record/campaign" \
 --run-dir "$record/campaign/runs/leflow_ttt_3072" --world "$world"
