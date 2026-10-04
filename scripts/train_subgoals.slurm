#!/usr/bin/env bash
#SBATCH --job-name=btm-jepa
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
# Submit from the repository: sbatch --export=ALL,STABLEWM_HOME=/data/leflow scripts/train_subgoals.slurm
# Activate btm-jepa before sbatch, or load your site's Conda module here.
export GPUS=4
bash scripts/train.sh "$@"
