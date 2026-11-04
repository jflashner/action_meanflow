# Settings for the ARC scripts. Edit once, then everything reads from here.
# All jobs must run on ONE cluster (scratch is not shared between clusters): submit from that cluster's login node.

# --- Slurm ---------------------------------------------------------------------------------------------------
ACCOUNT="CHANGE_ME"              # your Slurm account (run `quota` on a login node, or see coldfront.arc.vt.edu)
CLUSTER="falcon"                 # falcon (login: falcon1/2.arc.vt.edu) | tinkercliffs (login: tinkercliffs1/2.arc.vt.edu)

if [[ "$CLUSTER" == "falcon" ]]; then
    GPU_PARTITION="l40s_normal_q"    # L40S-48G: ARC's recommendation for small models (75 SU/GPU-h). Or a30_normal_q.
    LIGHT_PARTITION="t4_normal_q"    # Falcon has no CPU-only partition: small jobs (aggregation) go to a T4 (25 SU/GPU-h)
    LIGHT_GRES="gpu:1"
else
    GPU_PARTITION="a100_normal_q"    # A100-80G (100 SU/GPU-h). Or h200_normal_q (150 SU/GPU-h).
    LIGHT_PARTITION="normal_q"       # CPU nodes
    LIGHT_GRES=""
fi
QOS=""                           # empty: partition default (7 days max). e.g. fal_l40s_normal_short (1 day, 2x billing, higher priority)
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

# --- Experiments (GAMES, EXPERIMENTS, SEEDS, EXTRA_OVERRIDES can also be set in the environment) -----------------------------------------------------------------------------------------------
GAMES="${GAMES:-Boxing Breakout Pong}"     # Boxing first: stochastic opponent, the hard case for 1-step models
EXPERIMENTS="${EXPERIMENTS:-wm_edm wm_fm wm_shortcut wm_meanflow wm_meanflow_xpred wm_meanflow_selfforcing wm_meanflow_ctxnoise}"
SEEDS="${SEEDS:-1 2 3}"          # seeds of one (experiment, game) share one GPU, run concurrently
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
TRAIN_TIME="1-00:00:00"          # per training job (all seeds run in parallel), includes evaluation
DATA_TIME="02:00:00"

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
    TRAIN_TIME="02:00:00"
    DATA_TIME="01:00:00"
fi
