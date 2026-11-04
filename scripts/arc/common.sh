# Sourced at the start of every job script: config, modules, conda env, GPU logging, shared functions.
# Job scripts are copied by Slurm to a spool folder, so they find this folder through ARC_DIR (set by submit.sh).
set -euo pipefail
: "${ARC_DIR:?Submit jobs with bash scripts/arc/submit.sh, which sets ARC_DIR}"
# shellcheck source=config.sh
source "$ARC_DIR/config.sh"

module reset >/dev/null 2>&1 || true   # ARC: use `module reset`, never `module purge`
module load Miniforge3
set +u; source activate "$ENV_PREFIX"; set -u   # ARC: `source activate`, not `conda activate`

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="$CPUS_PER_RUN"
mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

echo "Job ${SLURM_JOB_ID:-local} ($SLURM_JOB_NAME) on $(hostname), $(date)"
echo "Partition ${SLURM_JOB_PARTITION:-?}, CPUs ${SLURM_CPUS_PER_TASK:-?}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"
echo "Repo $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo '?'), env $ENV_PREFIX"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), torch.cuda.get_device_name() if torch.cuda.is_available() else '')"

# GPU utilization log, as ARC recommends (check that requested GPUs are actually busy)
if command -v nvidia-smi >/dev/null 2>&1 && [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv -l 60 \
        > "$LOG_DIR/$SLURM_JOB_NAME-$SLURM_JOB_ID.gpu.csv" &
    GPU_LOGGER_PID=$!
    trap 'kill $GPU_LOGGER_PID 2>/dev/null || true' EXIT
fi

run_dir_of() {  # run_dir_of <experiment> <game> <seed>
    echo "$RUNS_ROOT/$1/$2/seed$3"
}

train_one() {  # train_one <experiment> <game> <seed>: start, resume, or skip if finished
    local exp=$1 game=$2 seed=$3
    local run_dir; run_dir=$(run_dir_of "$exp" "$game" "$seed")
    if [[ -f "$run_dir/.run_is_over" ]]; then
        echo "[$exp $game seed$seed] already trained"
    elif [[ -f "$run_dir/checkpoints/state.pt" ]]; then
        echo "[$exp $game seed$seed] resuming from $run_dir"
        (cd "$run_dir" && python src/main.py common.resume=True hydra.output_subdir=null hydra.run.dir=.)
    else
        echo "[$exp $game seed$seed] starting in $run_dir"
        rm -rf "$run_dir"
        mkdir -p "$(dirname "$run_dir")"
        # shellcheck disable=SC2086
        python "$REPO_DIR/src/main.py" +experiment="$exp" env.train.id="${game}NoFrameskip-v4" \
            static_dataset.path="$DATA_ROOT/$game" hydra.run.dir="$run_dir" common.seed="$seed" \
            wandb.mode="$WANDB_MODE" wandb.project="$WANDB_PROJECT" ${WANDB_ENTITY:+wandb.entity=$WANDB_ENTITY} \
            wandb.name="${exp}_${game}_s${seed}" wandb.group="${exp}_${game}" $EXTRA_OVERRIDES
    fi
}

eval_one() {  # eval_one <experiment> <game> <seed> [extra eval_wm.py args]
    local exp=$1 game=$2 seed=$3; shift 3
    local run_dir; run_dir=$(run_dir_of "$exp" "$game" "$seed")
    if [[ ! -f "$run_dir/.run_is_over" ]]; then
        echo "[$exp $game seed$seed] not finished, skipping evaluation"
        return 1
    fi
    if [[ -f "$run_dir/eval/$EVAL_NAME.json" && "${FORCE_EVAL:-0}" != 1 ]]; then
        echo "[$exp $game seed$seed] already evaluated ($run_dir/eval/$EVAL_NAME.json)"
        return 0
    fi
    local idm_args=()
    [[ -f "$IDM_ROOT/$game/idm.pt" ]] && idm_args=(--idm "$IDM_ROOT/$game/idm.pt")
    echo "[$exp $game seed$seed] evaluating"
    python "$REPO_DIR/src/eval_wm.py" --run-dir "$run_dir" --data-dir "$DATA_ROOT/$game" \
        --horizon "$EVAL_HORIZON" --num-rollouts "$EVAL_NUM_ROLLOUTS" --name "$EVAL_NAME" "${idm_args[@]}" "$@"
}
