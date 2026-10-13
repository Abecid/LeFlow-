#!/usr/bin/env bash
# Launch exactly ONE newly implemented baseline from a clean, frozen checkout.
# Usage: bash scripts/operations/run_additional_baseline.sh VARIANT RECORD PREVIOUS CACHE GPUS
# Requires your existing cached MetaWorld-v3 data, frozen V-JEPA/world and GPU runtime.
set -euo pipefail
variant="${1:?variant required}"
record="${2:?isolated record directory required}"
previous="${3:?previous campaign record required}"
cache="${4:?fresh cache directory required}"
gpus="${5:-4}"
case "$variant" in
  vanilla_action_flow|action_flow_search|sage_port|diffusion_policy_port|feedback_wm_port) ;;
  *) echo "Unsupported additional baseline: $variant" >&2; exit 2 ;;
esac
if [[ "$gpus" != "4" && "$gpus" != "8" ]]; then
  echo "GPUS must be 4 or 8" >&2; exit 2
fi
source_repo="$(git rev-parse --show-toplevel)"
git -C "$source_repo" diff --quiet
git -C "$source_repo" diff --cached --quiet
git -C "$source_repo" fetch origin main
test "$(git -C "$source_repo" rev-parse HEAD)" = "$(git -C "$source_repo" rev-parse origin/main)" || {
  echo "Checkout differs from remote main; refusing untracked source" >&2; exit 2; }
mkdir -p "$record" "$cache"
if [[ ! -d "$record/repo/.git" ]]; then
  git clone --local --branch main "$source_repo" "$record/repo"
fi
test "$(git -C "$record/repo" rev-parse HEAD)" = "$(git -C "$source_repo" rev-parse HEAD)" || {
  echo "Existing frozen experiment checkout differs; use a fresh record" >&2; exit 2; }
runtime="${LEFLOW_RUNTIME:-/tmp/mtxu-flow-jepa-20261007/env}"
test -x "$runtime/bin/python" || { echo "Set LEFLOW_RUNTIME to GPU conda environment" >&2; exit 2; }
cd "$record/repo"
export CUDA_MODULE_LOADING=LAZY OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export WANDB_MODE=online WANDB_ENTITY="${WANDB_ENTITY:-attentionx2023}"
export LD_LIBRARY_PATH="$runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=egl FLOW_EGL_DEVICES=0,0,0,0,0,0,0,0
export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe LP_NUM_THREADS=2
export __EGL_VENDOR_LIBRARY_FILENAMES="$runtime/share/glvnd/egl_vendor.d/50_mesa.json"
exec "$runtime/bin/python" -u -m scripts.operations.controller_grounded \
  --record "$record" --variant "$variant" --gpus "$gpus" \
  --previous "$previous" --cache "$cache" \
  --reuse-bank "${LEFLOW_ROUTE_BANK:-/tmp/mtxu-flow-reasoning-20261010/route_bank.pt}"
