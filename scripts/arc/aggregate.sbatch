#!/bin/bash
# Tables (CSV, Markdown) and figures from all evaluations.
# Submit with: bash scripts/arc/submit.sh aggregate
#SBATCH --job-name=aggregate
#SBATCH --nodes=1
#SBATCH --ntasks=1

source "$ARC_DIR/common.sh"
python scripts/aggregate_results.py --runs-root "$RUNS_ROOT" --out "$RESULTS_DIR" --eval-name "$EVAL_NAME"
