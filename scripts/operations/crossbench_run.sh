#!/usr/bin/env bash
# Run on the existing GPU server, from a clean checkout. No new cloud resources.
set -euo pipefail
workspace="${1:?usage: crossbench_run.sh WORKSPACE DATA_ROOT}"
data_root="${2:?existing STABLEWM_HOME / dataset volume required}"
python="${LEFLOW_PYTHON:-python}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 WANDB_MODE=online
export SDL_VIDEODRIVER=dummy MUJOCO_GL="${MUJOCO_GL:-egl}"
exec "$python" -u -m crossbench.campaign --workspace "$workspace" --data-root "$data_root" \
  --gpus 4 --execute --publish
