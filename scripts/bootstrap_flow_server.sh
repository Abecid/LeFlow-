#!/usr/bin/env bash
set -euo pipefail
repo_dir="${FLOW_REPO_DIR:-$HOME/research/LeFlow-joint-flow}"
data_dir="${FLOW_DATA_DIR:-$HOME/flow-jepa-data/v1}"
branch=research/joint-flow-metaworld
if [ ! -d "$repo_dir/.git" ]; then
  mkdir -p "$(dirname "$repo_dir")"
  git clone --branch "$branch" https://github.com/Abecid/LeFlow-.git "$repo_dir"
else
  test "$(git -C "$repo_dir" remote get-url origin)" = https://github.com/Abecid/LeFlow-.git
  test -z "$(git -C "$repo_dir" status --porcelain)"
  test "$(git -C "$repo_dir" branch --show-current)" = "$branch"
  git -C "$repo_dir" fetch origin "$branch"
  # Existing registered campaigns must retain their exact source revision.
  if [ ! -f "$data_dir/campaign.json" ]; then
    git -C "$repo_dir" merge --ff-only "origin/$branch"
  fi
fi
cd "$repo_dir"
conda_bin="$(command -v conda || true)"
if [ -z "$conda_bin" ]; then
  for candidate in "$HOME/miniconda3/bin/conda" "$HOME/miniforge3/bin/conda" "$HOME/anaconda3/bin/conda"; do
    if [ -x "$candidate" ]; then conda_bin="$candidate"; break; fi
  done
fi
if [ -z "$conda_bin" ]; then
  echo 'Conda is required on the server. No environment or GPU jobs were changed.' >&2
  exit 1
fi
if ! "$conda_bin" run -n flow-jepa python -c 'import sys' >/dev/null 2>&1; then
  "$conda_bin" create -y -n flow-jepa -c conda-forge python=3.11 pip git ffmpeg libgl libegl
fi
# Pin the CUDA 12.6 build rather than silently using the newest CUDA wheel.
"$conda_bin" run --no-capture-output -n flow-jepa python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
"$conda_bin" run --no-capture-output -n flow-jepa python -m pip install -r requirements-flow.txt
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 "$conda_bin" run --no-capture-output -n flow-jepa python -m pytest tests_flow -q
mkdir -p "$data_dir"
nohup "$conda_bin" run --no-capture-output -n flow-jepa python -u -m flow_jepa.campaign \
  --root "$data_dir" --gpus "${FLOW_MAX_GPUS:-4}" \
  >> "$data_dir/launcher.log" 2>&1 < /dev/null &
echo "Supervisor PID: $!"
echo "Launch log: $data_dir/launcher.log"
echo "The supervisor verifies online W&B, waits for idle GPUs, runs GPU preflight, then prepares data and trains."
