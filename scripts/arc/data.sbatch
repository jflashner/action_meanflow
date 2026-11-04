#!/bin/bash
# Collect the static dataset of one game with DIAMOND's pretrained agent, then train its inverse dynamics model.
# Submit with: GAME=Boxing bash scripts/arc/submit.sh data
#SBATCH --job-name=data
#SBATCH --nodes=1
#SBATCH --ntasks=1

source "$ARC_DIR/common.sh"
: "${GAME:?Set GAME}"

out="$DATA_ROOT/$GAME"
if [[ -f "$out/test/info.pt" ]]; then
    echo "Dataset already collected in $out"
else
    rm -rf "$out"  # remove a partial collection
    python src/collect_dataset.py --game "$GAME" --out-dir "$out" --train-steps "$TRAIN_STEPS" \
        --test-episodes "$TEST_EPISODES" --num-envs $(( SLURM_CPUS_PER_TASK > 3 ? SLURM_CPUS_PER_TASK - 2 : 1 ))
fi

if [[ -f "$IDM_ROOT/$GAME/idm.pt" ]]; then
    echo "Inverse dynamics model already trained"
else
    python src/train_idm.py --data-dir "$out" --out "$IDM_ROOT/$GAME/idm.pt" ${IDM_ARGS:-}
fi
