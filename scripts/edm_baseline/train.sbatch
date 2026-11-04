#!/bin/bash
#SBATCH --job-name=edm-baseline
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/edm-%j.log
set -euo pipefail
: "${EDM_CODE_ROOT:=${SLURM_SUBMIT_DIR:?}}"
cd "$EDM_CODE_ROOT"
export EDM_WORKSPACE="${EDM_WORKSPACE:-$EDM_CODE_ROOT}"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 WANDB_MODE=offline
export TORCH_HOME="${TORCH_HOME:-$EDM_WORKSPACE/.torch-cache}"
srun "${EDM_PYTHON:-$EDM_CODE_ROOT/.venv/bin/python}" scripts/edm_baseline/train.py --game "${1:?Specify Breakout or Boxing}" --seed 0 "${@:2}"
