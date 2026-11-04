#!/bin/bash
# Submit the whole pipeline from a login node:
#   1. per game: collect the dataset and train the inverse dynamics model (skipped if already done);
#   2. per (experiment, game): train all seeds on one GPU, then evaluate them (waits for step 1);
#   3. aggregate all evaluations into tables and figures (waits for step 2).
#
#   bash scripts/arc/submit_all.sh            # everything in config.sh
#   bash scripts/arc/submit_all.sh --smoke    # tiny end-to-end test in $SCRATCH_ROOT/smoke (about 30 min)
#   bash scripts/arc/submit_all.sh --dry-run  # print the sbatch commands only
#
# GAMES, EXPERIMENTS and SEEDS from config.sh can be overridden: GAMES="Boxing" bash scripts/arc/submit_all.sh
set -euo pipefail
ARC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ARC_DIR
for arg in "$@"; do
    case "$arg" in
        --smoke) export SMOKE=1 ;;
        --dry-run) export DRY_RUN=1 ;;
        *) echo "Unknown option $arg" >&2; exit 1 ;;
    esac
done
# shellcheck source=config.sh
source "$ARC_DIR/config.sh"
export SEEDS EXTRA_OVERRIDES RUNS_ROOT RESULTS_DIR

if [[ ! -d "$ENV_PREFIX" ]]; then
    echo "Conda env $ENV_PREFIX not found: build it first (see scripts/arc/setup_env.sh)" >&2
    exit 1
fi

train_jobs=()
for game in $GAMES; do
    dep=()
    if [[ -f "$DATA_ROOT/$game/test/info.pt" && -f "$IDM_ROOT/$game/idm.pt" ]]; then
        echo "$game: dataset and IDM found in $DATA_ROOT/$game"
    else
        jid=$(GAME=$game bash "$ARC_DIR/submit.sh" data)
        echo "$game: data job $jid"
        dep=(--dependency=afterok:$jid)
    fi
    for exp in $EXPERIMENTS; do
        jid=$(EXP=$exp GAME=$game bash "$ARC_DIR/submit.sh" train "${dep[@]}")
        echo "  $exp (seeds $SEEDS): train+eval job $jid"
        train_jobs+=("$jid")
    done
done

deps=$(IFS=:; echo "${train_jobs[*]}")
jid=$(bash "$ARC_DIR/submit.sh" aggregate --dependency=afterany:"$deps")
echo "aggregate job $jid -> $RESULTS_DIR"
echo "Monitor with: squeue -u \$USER    Logs: $LOG_DIR"
