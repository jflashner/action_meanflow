#!/bin/bash
# Build the conda environment. ARC requires this to run on a compute node of the same type as the training jobs
# (not on a login node, not inside a batch job). From a login node of the cluster set in config.sh:
#
#   source scripts/arc/config.sh
#   interact --account=$ACCOUNT --partition=$GPU_PARTITION --nodes=1 --ntasks-per-node=8 --gres=gpu:1 -t 1:00:00
#   bash scripts/arc/setup_env.sh        # then `exit` to release the node
#
# It also downloads DIAMOND's pretrained agents (Hugging Face Hub) and the LPIPS weights to the home cache.
set -euo pipefail
ARC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "$ARC_DIR/config.sh"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Run this inside an interactive job on a $GPU_PARTITION node (see the top of this file)." >&2
    exit 1
fi

module reset >/dev/null 2>&1 || true
module load Miniforge3
if [[ ! -d "$ENV_PREFIX" ]]; then
    conda create -y -p "$ENV_PREFIX" python=3.11
fi
set +u; source activate "$ENV_PREFIX"; set -u

cd "$REPO_DIR"
pip install --upgrade pip
pip install -r requirements.txt


python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), torch.cuda.get_device_name())
from huggingface_hub import hf_hub_download
for game in ["Boxing", "Breakout", "Pong"]:
    hf_hub_download(repo_id="eloialonso/diamond", filename=f"atari_100k/models/{game}.pt")
hf_hub_download(repo_id="eloialonso/diamond", filename="atari_100k/config/agent/default.yaml")
import lpips
lpips.LPIPS(net="alex")
import gymnasium, ale_py
env = gymnasium.make("BoxingNoFrameskip-v4")
env.reset(seed=0)
print("Atari OK, pretrained agents and LPIPS weights cached")
PY

mkdir -p "$DATA_ROOT" "$RUNS_ROOT" "$LOG_DIR" "$IDM_ROOT" "$RESULTS_DIR"
echo "Environment ready: $ENV_PREFIX"
