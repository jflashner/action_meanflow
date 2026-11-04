#!/bin/bash
# Train one experiment on one game for several seeds (concurrently, on one GPU), then evaluate each seed
# (sequentially, so that the speed measurements have the GPU to themselves). Safe to resubmit: finished runs are
# skipped and interrupted ones resume from their last checkpoint.
# Submit with: EXP=wm_meanflow GAME=Boxing SEEDS="1 2 3" bash scripts/arc/submit.sh train
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue

source "$ARC_DIR/common.sh"
: "${EXP:?Set EXP}" "${GAME:?Set GAME}" "${SEEDS:?Set SEEDS}"
[[ -f "$DATA_ROOT/$GAME/test/info.pt" ]] || { echo "No dataset in $DATA_ROOT/$GAME, run the data job first"; exit 1; }

pids=()
for seed in $SEEDS; do
    log="$LOG_DIR/${EXP}_${GAME}_seed${seed}.train.log"
    train_one "$EXP" "$GAME" "$seed" >> "$log" 2>&1 &
    pids+=($!)
    echo "seed $seed -> $log"
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ $status == 0 ]] || echo "WARNING: at least one training run failed, see the logs above"

if [[ "${RUN_EVAL:-1}" == 1 ]]; then
    for seed in $SEEDS; do
        eval_one "$EXP" "$GAME" "$seed" >> "$LOG_DIR/${EXP}_${GAME}_seed${seed}.eval.log" 2>&1 || status=1
    done
fi
exit $status
