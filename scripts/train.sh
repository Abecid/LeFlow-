#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${STABLEWM_HOME:?Set STABLEWM_HOME to the prepared data/cache directory}"
GPUS="${GPUS:-4}"
if ! [[ "$GPUS" =~ ^[1-4]$ ]]; then
  echo "GPUS must be 1, 2, 3, or 4" >&2
  exit 2
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
exec torchrun --nnodes=1 --nproc_per_node="$GPUS" \
  --master-addr="${MASTER_ADDR:-127.0.0.1}" --master-port="${MASTER_PORT:-29500}" \
  train_subgoals.py "$@"
