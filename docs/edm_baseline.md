# Offline EDM baseline

Breakout and Boxing, seed 0: **100 epochs × 1,000 optimizer updates**, batch 32. This trains DIAMOND's EDM frame generator on fixed gameplay, not the full Atari RL benchmark. Actor and reward/termination models are frozen.

The U-Net, EDM objective, native data loader, four-frame/action history, two training targets, AdamW settings, and sampler are unchanged from DIAMOND commit `5bcd1599755b4f2fae8e5e079e02f0728e174965`. Backbone/config hashes are checked before training. Packaging relocates the existing experiment scripts; it does not replace running ARC jobs.

## Setup

Use Python 3.10 and one CUDA GPU (tested original experiments: A30 24 GB, PyTorch 2.4.1/CUDA 12.1). Install Breakout and Boxing ROMs through your ALE setup; ROMs are not included.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements-edm.txt
export EDM_WORKSPACE="$PWD"
export TORCH_HOME="$EDM_WORKSPACE/.torch-cache"
```

`EDM_WORKSPACE` is the absolute data/output directory; code is always loaded from this checkout. It may point at an existing ARC workspace to reuse its immutable datasets. No datasets or weights are uploaded here.

## Data

If `datasets/{Breakout,Boxing}-v1` and the corresponding `*-native-v2` datasets already exist, reuse them. Otherwise:

```bash
python scripts/edm_baseline/atari_suite.py prepare
python scripts/edm_baseline/atari_suite.py collect --game Breakout
python scripts/edm_baseline/atari_suite.py collect --game Boxing
python scripts/edm_baseline/prepare_native_edm.py
```

Collection uses only pretrained DIAMOND actor weights at the pinned Hugging Face revision in `config/edm_baseline/protocol.json`. Each game has at least 100,000 training transitions, plus 16 validation and 16 test episodes, split by reset seed before windowing. Native conversion verifies pixel/action alignment and file hashes. Regenerating data is not guaranteed bitwise identical across platforms; use the same saved datasets for model comparisons.

## Run

First run the two-epoch, ten-updates-per-epoch smoke check inside a GPU allocation:

```bash
python scripts/edm_baseline/train.py --game Breakout --smoke
python scripts/edm_baseline/train.py --game Boxing --smoke
```

This checks native batch/action/context alignment, finite training, validation, frozen auxiliary models, and identical predictions after checkpoint reload. Smoke outputs use separate directories. Then:

```bash
python scripts/edm_baseline/train.py --game Breakout
python scripts/edm_baseline/train.py --game Boxing
```

For ARC, submit from the checkout root after activating the environment:

```bash
mkdir -p logs
export EDM_CODE_ROOT="$PWD"
export EDM_PYTHON="$(command -v python)"
sbatch --account=llm4se --partition=a30_normal_q --qos=fal_a30_normal_short --exclude=fal028 scripts/edm_baseline/train.sbatch Breakout
sbatch --account=llm4se --partition=a30_normal_q --qos=fal_a30_normal_short --exclude=fal028 scripts/edm_baseline/train.sbatch Boxing
```

Account, partition, GPU request and walltime can be overridden with `sbatch` options. Each command creates a new job; do not resubmit an active run. Output directories must not already exist, and this entry point does not implement resume.

## Outputs and interpretation

Results go to `$EDM_WORKSPACE/results/edm-uniform-100x1000-v1/<Game>-seed0/`. Validation runs at epoch 1 and every 10 epochs; the final scheduled checkpoint receives full evaluation. Training/validation JSONL, resolved configuration, source hashes, checkpoint files, per-window metrics, rollout GIFs, and `evaluation.json` are saved. `complete.json` appears only after successful evaluation.

Evaluation uses EDM at 1 and 3 denoising steps: next-frame MSE/PSNR/LPIPS, 300-step recorded-action rollout curves, correct-versus-random-action PSNR, and batch-one latency. Randomized actions can coincide with true actions; the changed fraction is reported. This measures action sensitivity, not counterfactual correctness. Latency includes CPU-to-GPU inputs and GPU-to-CPU output, excluding environment/display/network time. Long rollouts include only sufficiently long held-out episodes, with cohort IDs recorded. No gameplay returns or FVD are measured.

The schedule changes epoch grouping, validation cadence and iterator boundaries from the earlier 10,000 + 225 × 400 schedule; it is not a bitwise-equivalent rerun. One seed does not establish variation across training seeds. Existing test results have been inspected; checkpoints are selected by fixed update budget, not test performance.

CPU packaging check: `python scripts/edm_baseline/check_package.py`. Full GPU smoke checks remain necessary before launching this packaged entry point in a new environment.
