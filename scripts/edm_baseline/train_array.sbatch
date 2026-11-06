#!/bin/bash
#SBATCH --job-name=edm-three-seeds
#SBATCH --array=0-5%6
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/edm-%A_%a.log
set -euo pipefail
: "${EDM_CODE_ROOT:=${SLURM_SUBMIT_DIR:?}}"
task_id="${SLURM_ARRAY_TASK_ID:?Run with sbatch --array=0-5}"
case "$task_id" in
    0|1|2) game=Breakout ;;
    3|4|5) game=Boxing ;;
    *) echo "Invalid array task: $task_id" >&2; exit 2 ;;
esac
seed=$((task_id % 3))
cd "$EDM_CODE_ROOT"
exec bash scripts/edm_baseline/train.sbatch "$game" --seed "$seed" "$@"
