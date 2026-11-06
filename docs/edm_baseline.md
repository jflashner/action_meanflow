# Offline EDM baseline

Train DIAMOND's EDM generator on fixed Breakout and Boxing gameplay with seeds **0, 1 and 2**. Each run uses **100 epochs × 1,000 updates**, one A30, batch 32, four conditioning frames/actions, and two prediction targets. The actor and reward/termination models receive no updates. This is an offline generator comparison, not a reproduction of DIAMOND's Atari RL scores.

The U-Net, native data loader, EDM loss, optimizer settings and evaluation functions are preserved from the completed ARC experiments. Backbone/config hashes are checked against DIAMOND commit `5bcd1599755b4f2fae8e5e079e02f0728e174965`.

## Completed results

All six uniform-schedule runs completed training and final evaluation. [Results and figures](../results/edm-reference/uniform-100x1000-v1/summary/README.md) report means over three seeds. The rollout bands show sample standard deviation across seed-level trajectory means.

The reference directory contains each run's `evaluation.json`, `complete.json`, training summary, architecture, original source hashes, and [dataset/hardware provenance](../results/edm-reference/uniform-100x1000-v1/provenance.json). It contains no checkpoints, ROMs or gameplay data. These are recorded ARC results, not new runs of the packaged launcher.

| Experiment | Games / seeds | Updates per run | Schedule | Status |
| --- | --- | ---: | --- | --- |
| `atari-v1` | Both / 0, 1, 2 | 20,000 | Earlier custom loop | Superseded |
| `offline-edm-v2` | Both / 0, 1, 2 | 100,000 | 10,000 first; 400 thereafter | Complete |
| `uniform-100x1000-v1/EDM` | Both / 0, 1, 2 | 100,000 | 100 × 1,000 | All six complete |

New runs use `results/edm-uniform-100x1000-v1/<Game>-seed<N>/`. Existing output directories are rejected; there is no resume option. Historical ARC artifacts remain under their original experiment names.

## Setup and data

Use Python 3.10 and CUDA (original runs: PyTorch 2.4.1 / CUDA 12.1, A30 24 GB). Install the two Atari ROMs through your ALE setup.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements-edm.txt
export EDM_WORKSPACE="$PWD"
export TORCH_HOME="$EDM_WORKSPACE/.torch-cache"
```

`EDM_WORKSPACE` is the data/output directory; code always comes from this checkout. On ARC, set it to `/projects/llm4se/jaisharma/world-models/action-meanflow` to reuse the existing immutable datasets. Do not recollect datasets that already exist.

For a new workspace, run collection inside a GPU allocation:

```bash
python scripts/edm_baseline/atari_suite.py prepare
python scripts/edm_baseline/atari_suite.py collect --game Breakout
python scripts/edm_baseline/atari_suite.py collect --game Boxing
python scripts/edm_baseline/prepare_native_edm.py
```

Collection uses the pinned pretrained DIAMOND actor and exploration settings in `config/edm_baseline/protocol.json`. Train/validation/test split by reset episode. Existing training datasets contain 101,053 Breakout and 101,294 Boxing transitions; collection finishes whole episodes after reaching 100,000 transitions. Regeneration is not guaranteed bitwise identical across platforms. Use the same saved datasets across methods and training seeds.

## Check and train

Run the CPU package check, then the GPU smoke tests in a one-GPU allocation:

```bash
python scripts/edm_baseline/check_package.py
python scripts/edm_baseline/train.py --game Breakout --seed 0 --smoke
python scripts/edm_baseline/train.py --game Boxing --seed 0 --smoke
```

Smoke tests use separate output directories and check native batch/action alignment, generated context, finite gradients, frozen auxiliary models and checkpoint reload. Full runs keep the same training and evaluation code for all seeds:

```bash
python scripts/edm_baseline/train.py --game Boxing --seed 1
```

For all six runs on ARC, submit from this checkout after activating the environment:

```bash
mkdir -p logs
export EDM_CODE_ROOT="$PWD"
export EDM_PYTHON="$(command -v python)"
sbatch --account=llm4se --partition=a30_normal_q --qos=fal_a30_normal_short --exclude=fal028 scripts/edm_baseline/train_array.sbatch
```

Array tasks 0–2 are Breakout seeds 0–2; tasks 3–5 are Boxing seeds 0–2. Each requests one A30, four CPUs, 32 GB RAM and six hours. Override concurrency with `--array=0-5%3`, or submit only selected tasks. Do not submit both the array and duplicate individual jobs. Original full runs took approximately four hours including evaluation.

For one job:

```bash
sbatch --account=llm4se --partition=a30_normal_q --qos=fal_a30_normal_short --exclude=fal028 scripts/edm_baseline/train.sbatch Boxing --seed 2
```

Validation runs at epoch 1, every 10 epochs and the final epoch. Final evaluation uses the scheduled 100,000-update checkpoint and EDM at 1 and 3 steps. `complete.json` is written only after evaluation succeeds. Inspect the Slurm log, `training-progress.jsonl`, `validation-progress.jsonl` and completion record for each run.

## Regenerate results and plots

For newly completed runs:

```bash
python scripts/edm_baseline/report.py
```

To regenerate the committed reference results without a GPU or dataset:

```bash
python scripts/edm_baseline/report.py --runs-dir results/edm-reference/uniform-100x1000-v1
```

For the original ARC output layout:

```bash
python scripts/edm_baseline/report.py --runs-dir "$EDM_WORKSPACE/results/uniform-100x1000-v1/EDM"
```

Outputs go into `<runs-dir>/summary/`, or `--output-dir`: mean-only Markdown/CSV tables, per-seed JSON statistics, shaded PSNR/LPIPS rollout plots and a peak-memory plot (PDF/PNG). Reporting checks all six completions, seed identities, matching evaluation windows, finite metrics and the 100,000-update budget.

Next-frame metrics use 128 test windows per game. Rollouts use 12 eligible Breakout and 16 Boxing trajectories, with PSNR at all 300 steps and LPIPS at steps 1, 4, 8, 16, 32, 64, 100, 200 and 300. Action sensitivity compares recorded versus randomized final actions with shared noise; random actions can coincide with recorded ones. Latency is batch-one FP32 with CPU/GPU transfers, excluding environment/display time. Memory is peak PyTorch-allocated live tensors, not total GPU usage. No FVD or gameplay returns are measured. Seed variation reflects training randomness on a shared dataset.
