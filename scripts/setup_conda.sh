#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if ! command -v conda >/dev/null 2>&1; then
  echo "Install Miniforge/Conda, then rerun this script (see docs/TRAINING.md)." >&2
  exit 1
fi
conda env create -f environment.yml
conda run -n btm-jepa python scripts/preflight.py --skip-data --skip-wandb
