#!/usr/bin/env bash
set -euo pipefail
record=/home/mtxu/adam/LeFlow-experiments/20261007-joint-flow
scratch=/tmp/mtxu-flow-jepa-20261007
cd "$record/repo"
export CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_MODULE_LOADING=LAZY OMP_NUM_THREADS=2
export WANDB_MODE=online WANDB_ENTITY=attentionx2023
export LD_LIBRARY_PATH="$scratch/env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=egl FLOW_EGL_DEVICES=0,0,0,0 LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe LP_NUM_THREADS=2
export __EGL_VENDOR_LIBRARY_FILENAMES="$scratch/env/share/glvnd/egl_vendor.d/50_mesa.json"
exec "$scratch/env/bin/python" -u -m flow_jepa.campaign --root "$record/campaign" --gpus 4
