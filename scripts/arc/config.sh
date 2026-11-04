# Settings for the ARC scripts. Edit once, then everything reads from here.
# All jobs must run on ONE cluster (scratch is not shared between clusters): submit from that cluster's login node.

# --- Slurm ---------------------------------------------------------------------------------------------------
ACCOUNT="CHANGE_ME"              # your Slurm account (run `quota` on a login node, or see coldfront.arc.vt.edu)
CLUSTER="tinkercliffs"           # tinkercliffs (login: tinkercliffs1/2.arc.vt.edu) | falcon (login: falcon1/2.arc.vt.edu)

# QOS applies to the GPU jobs. "short" QoS: priority 1200 instead of 1000, at most 1 day per job, billed at 2x, and at
# most 10 running jobs per user on A100 (more jobs wait in the queue). Use "" for the base QoS (7 days, 1x billing).
# Run `showqos` on a login node to check the names.
if [[ "$CLUSTER" == "falcon" ]]; then
    GPU_PARTITION="l40s_normal_q"    # L40S-48G: ARC's recommendation for small models (75 SU/GPU-h). Or a30_normal_q.
    QOS="fal_l40s_normal_short"
    LIGHT_PARTITION="t4_normal_q"    # Falcon has no CPU-only partition: small jobs (aggregation) go to a T4 (25 SU/GPU-h)
    LIGHT_GRES="gpu:1"
else
    GPU_PARTITION="a100_normal_q"    # A100-80G (100 SU/GPU-h). Or h200_normal_q (150 SU/GPU-h, QoS tc_h200_normal_short).
    QOS="tc_a100_normal_short"
    LIGHT_PARTITION="normal_q"       # CPU nodes
    LIGHT_GRES=""
fi
MAIL_USER=""                     # e.g. you@vt.edu to get emails on job end/failure

# --- Paths -----------------------------------------------------------------------------------------------------
REPO_DIR="$HOME/action_meanflow"                       # where you cloned the repo
ENV_PREFIX="$HOME/envs/action_meanflow_${GPU_PARTITION%%_*}"  # conda env, built on the GPU node type (setup_env.sh)
SCRATCH_ROOT="/scratch/$USER/action_meanflow"          # files are purged after 90 days: copy results somewhere safe
DATA_ROOT="$SCRATCH_ROOT/data"                         # static datasets: $DATA_ROOT/<Game>/{train,test}
IDM_ROOT="$SCRATCH_ROOT/idm"                           # inverse dynamics models: $IDM_ROOT/<Game>/idm.pt
RUNS_ROOT="${RUNS_ROOT:-$SCRATCH_ROOT/runs}"           # training runs: $RUNS_ROOT/<experiment>/<Game>/seed<k>
LOG_DIR="$SCRATCH_ROOT/logs"                           # Slurm logs and GPU utilization logs
RESULTS_DIR="${RESULTS_DIR:-$SCRATCH_ROOT/results}"     # tables and figures from aggregate.sbatch

# --- Experiments (GAMES, EXPERIMENTS, SEEDS, EXTRA_OVERRIDES can also be set in the environment) -------------------
# Default: the core comparison on Boxing with one seed (1 data job + 4 training jobs).
# `submit_all.sh --full`: all games, the MeanFlow ablations and 3 seeds (3 data jobs + 21 training jobs). It uses the
# same folders, so the runs and evaluations already finished by the core set are skipped.
CORE_EXPERIMENTS="wm_edm wm_fm wm_shortcut wm_meanflow"
if [[ "${FULL:-0}" == 1 ]]; then
    GAMES="${GAMES:-Boxing Breakout Pong}"
    EXPERIMENTS="${EXPERIMENTS:-$CORE_EXPERIMENTS wm_meanflow_xpred wm_meanflow_selfforcing wm_meanflow_ctxnoise}"
    SEEDS="${SEEDS:-1 2 3}"      # seeds of one (experiment, game) share one GPU, run concurrently
    DEFAULT_TRAIN_TIME="12:00:00"
else
    GAMES="${GAMES:-Boxing}"     # Boxing first: stochastic opponent, the hard case for 1-step models
    EXPERIMENTS="${EXPERIMENTS:-$CORE_EXPERIMENTS}"
    SEEDS="${SEEDS:-1}"
    DEFAULT_TRAIN_TIME="08:00:00"
fi
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"  # extra Hydra overrides for every training run, e.g. "training.num_final_epochs=200"

# Data collection (DIAMOND's pretrained agents, epsilon-greedy)
TRAIN_STEPS=100000
TEST_EPISODES=30
IDM_ARGS=""                      # extra train_idm.py arguments, e.g. "--steps 10000"

# Evaluation
EVAL_HORIZON=300                 # rollout length in frames
EVAL_NUM_ROLLOUTS=64
EVAL_NAME="test_ema"

# Resources
CPUS_PER_RUN=4
MEM_PER_RUN_GB=24                # the dataset is cached in RAM as float32 (~5 GB per 100k frames) for each run
# Time limits. A job that hits its limit can simply be resubmitted: runs resume from their last checkpoint (every 1000
# updates), finished runs and existing evaluations are skipped. Estimates (not yet measured on A100) for 100k updates
# with 3 seeds on one GPU: ~3 h for EDM / flow matching, ~9 h for MeanFlow + self-forcing, plus ~45 min of evaluation.
# One seed alone on the GPU is faster.
TRAIN_TIME="${TRAIN_TIME:-$DEFAULT_TRAIN_TIME}"   # per training job (all seeds run in parallel), includes evaluation
DATA_TIME="01:00:00"             # data collection + inverse dynamics model, ~30 min
EVAL_TIME="03:00:00"             # standalone re-evaluation jobs (submit.sh eval)

# Logging: disabled | offline (sync later with `wandb sync`) | online
WANDB_MODE="disabled"
WANDB_PROJECT="action_meanflow"
WANDB_ENTITY=""

# --- Smoke test (bash scripts/arc/submit_all.sh --smoke): tiny version of the whole pipeline, in its own folder ----
if [[ "${SMOKE:-0}" == 1 ]]; then
    SCRATCH_ROOT="$SCRATCH_ROOT/smoke"
    DATA_ROOT="$SCRATCH_ROOT/data"; IDM_ROOT="$SCRATCH_ROOT/idm"; RUNS_ROOT="$SCRATCH_ROOT/runs"
    LOG_DIR="$SCRATCH_ROOT/logs"; RESULTS_DIR="$SCRATCH_ROOT/results"
    GAMES="Boxing"
    SEEDS="1"
    TRAIN_STEPS=5000
    TEST_EPISODES=4
    IDM_ARGS="--steps 500"
    EXTRA_OVERRIDES="training.num_final_epochs=2 denoiser.training.steps_first_epoch=300 denoiser.training.steps_per_epoch=300 evaluation.every=1 denoiser.training.self_forcing.start_after_steps=100"
    EVAL_HORIZON=50
    EVAL_NUM_ROLLOUTS=8
    TRAIN_TIME="01:00:00"
    DATA_TIME="00:30:00"
fi
