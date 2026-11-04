#!/bin/bash
# Submit one job with the resources from config.sh. Variables for the job are passed as environment variables.
#
#   GAME=Boxing bash scripts/arc/submit.sh data
#   EXP=wm_meanflow GAME=Boxing SEEDS="1 2 3" bash scripts/arc/submit.sh train [--dependency=afterok:123]
#   EXP=wm_meanflow GAME=Boxing SEEDS="1 2 3" EVAL_ARGS="--nfe 1 2 4" bash scripts/arc/submit.sh eval
#   bash scripts/arc/submit.sh aggregate
#
# Extra arguments are passed to sbatch. DRY_RUN=1 prints the sbatch command instead of running it.
set -euo pipefail
ARC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ARC_DIR
# shellcheck source=config.sh
source "$ARC_DIR/config.sh"

kind=${1:?Usage: submit.sh <data|train|eval|aggregate> [sbatch args...]}
shift
if [[ "$ACCOUNT" == "CHANGE_ME" ]]; then
    echo "Set ACCOUNT in $ARC_DIR/config.sh (run \`quota\` to see your Slurm accounts)" >&2
    exit 1
fi

gpu_job=(--partition="$GPU_PARTITION" --gres=gpu:1 ${QOS:+--qos=$QOS})
case "$kind" in
    data)
        : "${GAME:?Set GAME}"
        name="data_${GAME}"
        resources=("${gpu_job[@]}" --cpus-per-task=10 --mem=48G --time="$DATA_TIME")
        ;;
    train|eval)
        : "${EXP:?Set EXP}" "${GAME:?Set GAME}"
        export SEEDS EXTRA_OVERRIDES RUNS_ROOT RESULTS_DIR
        n=$(wc -w <<< "$SEEDS")
        name="${kind}_${EXP#wm_}_${GAME}"
        time=$TRAIN_TIME
        [[ "$kind" == eval ]] && time="06:00:00"
        resources=("${gpu_job[@]}" --cpus-per-task=$((CPUS_PER_RUN * n)) --mem=$((MEM_PER_RUN_GB * n))G --time="$time")
        ;;
    aggregate)
        export RUNS_ROOT RESULTS_DIR
        name="aggregate"
        resources=(--partition="$LIGHT_PARTITION" ${LIGHT_GRES:+--gres=$LIGHT_GRES} --cpus-per-task=2 --mem=16G --time=00:30:00)
        ;;
    *)
        echo "Unknown job type: $kind" >&2
        exit 1
        ;;
esac

mkdir -p "$LOG_DIR"
cmd=(sbatch --parsable --account="$ACCOUNT" --job-name="$name" --output="$LOG_DIR/%x-%j.out" --export=ALL
     ${MAIL_USER:+--mail-user=$MAIL_USER --mail-type=END,FAIL} "${resources[@]}" "$@" "$ARC_DIR/$kind.sbatch")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "${cmd[*]}" >&2
    echo "DRYRUN_$name"
else
    "${cmd[@]}"
fi
