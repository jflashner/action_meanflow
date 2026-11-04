#!/bin/bash
# (Re-)evaluate finished runs, one seed after the other. Extra eval_wm.py arguments go in EVAL_ARGS.
# Submit with: EXP=wm_meanflow GAME=Boxing SEEDS="1 2 3" EVAL_ARGS="--nfe 1 2 4" bash scripts/arc/submit.sh eval
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks=1

source "$ARC_DIR/common.sh"
: "${EXP:?Set EXP}" "${GAME:?Set GAME}" "${SEEDS:?Set SEEDS}"
status=0
for seed in $SEEDS; do
    # shellcheck disable=SC2086
    eval_one "$EXP" "$GAME" "$seed" ${EVAL_ARGS:-} 2>&1 | tee -a "$LOG_DIR/${EXP}_${GAME}_seed${seed}.eval.log" || status=1
done
exit $status
