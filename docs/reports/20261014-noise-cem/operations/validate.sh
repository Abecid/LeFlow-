#!/bin/bash
set -euo pipefail
B=/home/mtxu/adam/LeFlow-experiments/20261014-noise-cem
P=/tmp/mtxu-crossbench-20261013/env/bin/python
cd "$B/runtime"
tar -xzf "$B/source.tar.gz"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NVIDIA_TF32_OVERRIDE=0
export LD_LIBRARY_PATH=/tmp/mtxu-flow-jepa-20261007/env/lib MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe LP_NUM_THREADS=2
export __EGL_VENDOR_LIBRARY_FILENAMES=/tmp/mtxu-flow-jepa-20261007/env/share/glvnd/egl_vendor.d/50_mesa.json SDL_VIDEODRIVER=dummy
CUDA_VISIBLE_DEVICES='' "$P" -m pytest -q tests/test_noise_cem*.py tests/test_leflow_refinement*.py tests/self_refine --junitxml="$B/logs/cpu-tests.xml" > "$B/logs/cpu-tests.log" 2>&1
CUDA_VISIBLE_DEVICES=0 "$P" -u "$B/preflight_noise.py" --task pusht > "$B/logs/preflight-pusht.log" 2>&1 &
PID_PUSHT=$!
CUDA_VISIBLE_DEVICES=1 "$P" -u "$B/preflight_noise.py" --task cube > "$B/logs/preflight-cube.log" 2>&1 &
PID_CUBE=$!
wait "$PID_PUSHT"
wait "$PID_CUBE"
"$P" -c 'import json,pathlib; b=pathlib.Path("/home/mtxu/adam/LeFlow-experiments/20261014-noise-cem"); print((b/"logs/cpu-tests.log").read_text()); print(json.dumps({t:json.loads((b/"preflight"/(t+".json")).read_text())["weights_unchanged"] for t in ["pusht","cube"]}))'
