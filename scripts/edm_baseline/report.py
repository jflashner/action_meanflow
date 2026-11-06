import argparse
import csv
import statistics
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from paths import ROOT

parser = argparse.ArgumentParser()
parser.add_argument('--runs-dir', type=Path, default=ROOT / 'results/edm-uniform-100x1000-v1')
parser.add_argument('--output-dir', type=Path)
args = parser.parse_args()
RUNS = args.runs_dir.expanduser().resolve()
OUT = args.output_dir.expanduser().resolve() if args.output_dir else RUNS / 'summary'
OUT.mkdir(parents=True, exist_ok=True)
curves, cohorts, hashes = {}, {}, {}


def read(path):
    raw = path.read_bytes()
    hashes[str(path.relative_to(RUNS))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


full_results = {}
for game in ('Breakout', 'Boxing'):
    full_results[game] = []
    evaluations = []
    for seed in range(3):
        run = RUNS / f'{game}-seed{seed}'
        done = read(run / 'complete.json')
        assert done['status'] == 'complete' and not done['smoke']
        assert (done['method'], done['game'], done['seed']) == ('EDM', game, seed)
        assert done['training']['updates'] == 100000
        assert done['training']['actor_updates'] == done['training']['reward_updates'] == 0
        result = read(run / 'evaluation.json')
        full_results[game].append(result)
        evaluations.append(result['test'])
    ids = evaluations[0]['rollout_window_ids']
    assert all(e['rollout_window_ids'] == ids for e in evaluations)
    assert all(e['window_ids'] == evaluations[0]['window_ids'] for e in evaluations)
    assert len({pair[0] for pair in ids}) == len(ids)
    cohorts[game] = {'trajectories_per_seed': len(ids), 'window_ids': ids}
    curves[game] = {}
    for steps in ('1', '3'):
        curves[game][steps] = {}
        for metric in ('psnr_db', 'lpips_alex'):
            indices = [i for i, row in enumerate(evaluations[0]['rollout'][steps]) if metric in row]
            assert [i + 1 for i in indices] == (list(range(1, 301)) if metric == 'psnr_db' else [1, 4, 8, 16, 32, 64, 100, 200, 300])
            for e in evaluations:
                assert [i for i, row in enumerate(e['rollout'][steps]) if metric in row] == indices
            values = np.asarray([[e['rollout'][steps][i][metric] for i in indices] for e in evaluations])
            assert values.shape == (3, len(indices)) and np.isfinite(values).all()
            curves[game][steps][metric] = {
                'steps': [i + 1 for i in indices],
                'mean': values.mean(0).tolist(),
                'std': values.std(0, ddof=1).tolist(),
                'seed_means': values.tolist(),
            }

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
fig, axes = plt.subplots(2, 2, figsize=(7, 4.4), sharex=True, sharey='row')
for col, game in enumerate(('Breakout', 'Boxing')):
    axes[0, col].set_title(f"{game} ({cohorts[game]['trajectories_per_seed']} trajectories per seed)", fontsize=9)
    for row, metric in enumerate(('psnr_db', 'lpips_alex')):
        ax = axes[row, col]
        for steps, color, style in [('1', '#0072B2', '-'), ('3', '#D55E00', '--')]:
            curve = curves[game][steps][metric]
            x, mean, std = (np.asarray(curve[k]) for k in ('steps', 'mean', 'std'))
            ax.fill_between(x, np.maximum(0, mean - std), mean + std, color=color, alpha=0.16, linewidth=0)
            ax.plot(x, mean, color=color, linestyle=style, linewidth=1.2,
                    marker='o' if metric == 'lpips_alex' else None, markersize=3,
                    label=f'EDM, {steps} step' + ('s' if steps == '3' else ''))
        ax.set_xlim(1, 300)
        ax.set_xticks([1, 100, 200, 300])
        ax.grid(axis='y', color='#e5e7eb', linewidth=0.5)
        ax.set_axisbelow(True)
        if col == 0:
            ax.set_ylabel('PSNR (dB; higher is better)' if row == 0 else 'LPIPS (lower is better)')
        if row == 1:
            ax.set_xlabel('Autoregressive rollout step')
for row, metric in enumerate(('psnr_db', 'lpips_alex')):
    ymax = max(max(np.asarray(curves[g][s][metric]['mean']) + np.asarray(curves[g][s][metric]['std']))
               for g in curves for s in ('1', '3'))
    axes[row, 0].set_ylim(0, ymax * 1.06)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=9)
fig.text(0.5, 0.905, 'Mean and shaded ±1 SD across 3 training seeds', ha='center', fontsize=8, color='#4b5563')
fig.subplots_adjust(left=0.10, right=0.97, top=0.82, bottom=0.11, hspace=0.22, wspace=0.13)
for extension in ('pdf', 'png'):
    fig.savefig(OUT / f'edm-rollout-psnr-lpips-three-seeds.{extension}', dpi=300, facecolor='white')
plt.close(fig)
(OUT / 'plot-data-and-provenance.json').write_text(json.dumps({
    'experiment': 'uniform-100x1000-v1', 'method': 'EDM', 'training_seeds': [0, 1, 2],
    'updates': 100000, 'split': 'test', 'horizon': 300, 'cohorts': cohorts,
    'curves': curves, 'source_sha256': hashes,
    'aggregation': 'Equal-weight mean of per-seed trajectory means; PSNR averaged in dB. No smoothing.',
    'lpips_sampling': 'Steps 1, 4, 8, 16, 32, 64, 100, 200, 300 only; connecting lines and shading interpolate visually.',
    'shading': 'Sample standard deviation (ddof=1) across the three seed-level trajectory means; lower bound clipped at zero. Not trajectory standard deviation or confidence intervals.',
    'protocol': 'Matched four-frame initial histories and recorded actions, same trajectories across seeds. Generated frames become context.'
}, indent=2) + '\n')
print(json.dumps({'output': str(OUT), 'checks': 'passed',
                  'step300': {g: {s: {m: curves[g][s][m]['mean'][-1] for m in ('psnr_db', 'lpips_alex')} for s in ('1', '3')} for g in curves}}, indent=2))

rows, statistics_rows = [], []
for game, results in full_results.items():
    for steps in ('1', '3'):
        samples = {
            'next_psnr_db': [d['test']['sampling'][steps]['psnr_db'] for d in results],
            'next_lpips': [d['test']['sampling'][steps]['lpips_alex'] for d in results],
            'step300_psnr_db': [d['test']['rollout'][steps][-1]['psnr_db'] for d in results],
            'step300_lpips': [d['test']['rollout'][steps][-1]['lpips_alex'] for d in results],
            'median_ms': [d['latency'][steps]['median_ms'] for d in results],
            'p95_ms': [d['latency'][steps]['p95_ms'] for d in results],
            'fps': [1000 / statistics.mean(d['latency'][steps]['measurements_ms']) for d in results],
            'peak_allocated_mib': [d['latency'][steps]['peak_allocated_mib'] for d in results],
            'action_delta_psnr_db': [d['test']['sampling'][steps]['action_delta_psnr_db'] for d in results],
        }
        assert all(np.isfinite(values).all() for values in samples.values())
        rows.append(dict(game=game, sampling_steps=int(steps), **{k: statistics.mean(v) for k, v in samples.items()}))
        statistics_rows.append(dict(game=game, sampling_steps=int(steps), metrics={
            k: dict(mean=statistics.mean(v), sample_std=statistics.stdev(v), per_seed=v) for k, v in samples.items()}))
with (OUT / 'metrics.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
(OUT / 'metrics.json').write_text(json.dumps(dict(seeds=[0, 1, 2], results=statistics_rows), indent=2) + '\n')
lines = [
    '# EDM: three-seed results', '',
    'Mean over seeds 0, 1 and 2, each trained for 100,000 updates on the same fixed dataset per game.', '',
    '| Game | Steps | Next PSNR (dB) | Next LPIPS | PSNR at 300 | LPIPS at 300 | Median ms | p95 ms | FPS |',
    '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
]
for r in rows:
    lines.append(f"| {r['game']} | {r['sampling_steps']} | {r['next_psnr_db']:.2f} | {r['next_lpips']:.6f} | {r['step300_psnr_db']:.2f} | {r['step300_lpips']:.4f} | {r['median_ms']:.2f} | {r['p95_ms']:.2f} | {r['fps']:.2f} |")
lines += ['',
    'Next-frame evaluation uses the same held-out windows across seeds. Rollouts use the same eligible trajectories per game and recorded actions; generated frames become context.',
    'Latency columns average the per-run median and p95; FPS averages 1000 / mean latency per run. Batch-one FP32 includes CPU/GPU transfers, excludes environment/display time.',
    'The rollout bands show sample standard deviation across three seed-level trajectory means. LPIPS is measured at nine marked steps only. These are generator metrics, not gameplay scores. FVD is not measured.', '',
    '![Rollout quality](edm-rollout-psnr-lpips-three-seeds.png)', '',
    '![Inference memory](edm-inference-memory-three-seeds.png)', '',
    'Memory is peak PyTorch-allocated live tensor memory during inference (including warmup), averaged across seeds. It excludes CUDA context and other allocations; it is not total device usage or incremental memory per frame.',
]
(OUT / 'README.md').write_text('\n'.join(lines) + '\n')

fig, ax = plt.subplots(figsize=(3.4, 2.5))
x = np.arange(2)
for steps, offset, color in [(1, -0.18, '#0072B2'), (3, 0.18, '#D55E00')]:
    values = [r['peak_allocated_mib'] for r in rows if r['sampling_steps'] == steps]
    bars = ax.bar(x + offset, values, width=0.34, color=color, label=f'EDM, {steps} step' + ('s' if steps == 3 else ''))
    ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=7)
ax.set_xticks(x, ['Breakout', 'Boxing'])
ax.set_ylim(0, max(r['peak_allocated_mib'] for r in rows) * 1.28)
ax.set_ylabel('Peak allocated GPU memory (MiB)')
ax.set_title('Mean across 3 training seeds', fontsize=9)
ax.legend(frameon=False, fontsize=7, loc='upper center', ncol=2)
fig.tight_layout()
for extension in ('pdf', 'png'):
    fig.savefig(OUT / f'edm-inference-memory-three-seeds.{extension}', dpi=300, facecolor='white')
plt.close(fig)
