#! /usr/bin/env python
"""
Aggregate the JSON files written by src/eval_wm.py into tables (CSV + Markdown) and figures.

Run folders are expected at <runs_root>/<experiment>/<Game>/seed<k>, as created by scripts/arc/train.sbatch.

    python scripts/aggregate_results.py --runs-root /scratch/$USER/action_meanflow/runs --out results/atari
"""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


# Fixed color per method (color follows the method, never its rank), plus a marker shape as secondary encoding
METHOD_STYLE = {
    "wm_edm": ("EDM (DIAMOND)", "#2a78d6", "o"),
    "wm_fm": ("Flow matching", "#eb6834", "s"),
    "wm_shortcut": ("Shortcut", "#1baf7a", "^"),
    "wm_meanflow": ("MeanFlow", "#eda100", "D"),
    "wm_meanflow_xpred": ("MeanFlow x-pred", "#e87ba4", "v"),
    "wm_meanflow_selfforcing": ("MeanFlow + self-forcing", "#008300", "P"),
    "wm_meanflow_ctxnoise": ("MeanFlow + context noise", "#6250d6", "X"),
}
OTHER_STYLE = ("#6b6a63", "o")

COLUMNS = [
    ("next_frame", "psnr"),
    ("next_frame", "ssim"),
    ("next_frame", "lpips"),
    ("rollout", "psnr_mean16"),
    ("rollout", "psnr_mean50"),
    ("rollout", "psnr_t100"),
    ("rollout", "lpips_mean50"),
    ("rollout", "delta_psnr_mean16"),
    ("rollout", "delta_psnr_mean50"),
    ("idm", "real_actions"),
    ("idm", "random_actions"),
    ("idm", "gt"),
    ("speed", "ms_per_frame"),
    ("speed", "fps"),
]


def label(method: str) -> str:
    return METHOD_STYLE.get(method, (method,))[0]


def style(method: str):
    return METHOD_STYLE[method][1:] if method in METHOD_STYLE else OTHER_STYLE


def load(runs_root: Path, eval_name: str) -> List[dict]:
    records = []
    for path in sorted(runs_root.glob(f"*/*/seed*/eval/{eval_name}.json")):
        method, game, seed = path.parts[-5], path.parts[-4], path.parts[-3]
        res = json.loads(path.read_text(encoding="utf-8"))
        for nfe, r in res["nfe"].items():
            values = {}
            for section, key in COLUMNS:
                if section == "rollout":
                    v = r["rollout"]["summary"].get(key)
                elif section == "idm":
                    v = r["rollout"].get("idm_accuracy", {}).get(key)
                else:
                    v = r.get(section, {}).get(key)
                values[f"{section}/{key}"] = v
            records.append(
                {
                    "method": method,
                    "game": game,
                    "seed": seed,
                    "nfe": int(nfe),
                    "values": values,
                    "curves": r["rollout"]["curves"],
                    "gpu": res.get("gpu"),
                }
            )
    return records


def aggregate(records: List[dict]) -> Dict:
    groups = defaultdict(list)
    for r in records:
        groups[(r["game"], r["method"], r["nfe"])].append(r)
    return groups


def fmt(values: List[float]) -> str:
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{np.mean(values):.3f} ± {np.std(values):.3f}"


def write_tables(groups: Dict, out: Path) -> None:
    keys = [f"{s}/{k}" for s, k in COLUMNS]
    order = list(METHOD_STYLE)
    sort_key = lambda g: (g[0], order.index(g[1]) if g[1] in order else len(order), g[1], g[2])
    rows = []
    for g in sorted(groups, key=sort_key):
        game, method, nfe = g
        rs = groups[g]
        row = {"game": game, "method": method, "label": label(method), "nfe": nfe, "num_seeds": len(rs)}
        for k in keys:
            vals = [r["values"][k] for r in rs if r["values"][k] is not None]
            row[f"{k}_mean"] = float(np.mean(vals)) if vals else None
            row[f"{k}_std"] = float(np.std(vals)) if vals else None
            row[k] = fmt(vals)
        rows.append(row)

    with (out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["game", "method", "label", "nfe", "num_seeds"] + [f"{k}_{s}" for k in keys for s in ("mean", "std")]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    md_keys = [
        ("next_frame/psnr", "PSNR (1 frame)"),
        ("next_frame/lpips", "LPIPS (1 frame)"),
        ("rollout/psnr_mean50", "PSNR (rollout, 50)"),
        ("rollout/psnr_t100", "PSNR @100"),
        ("rollout/delta_psnr_mean16", "ΔPSNR (16)"),
        ("idm/random_actions", "IDM acc (random a)"),
        ("speed/fps", "FPS"),
    ]
    lines = []
    for game in sorted({r["game"] for r in rows}):
        lines += [f"## {game}", "", "| Method | NFE | Seeds | " + " | ".join(h for _, h in md_keys) + " |"]
        lines.append("|" + "---|" * (3 + len(md_keys)))
        for r in rows:
            if r["game"] == game:
                lines.append(f"| {r['label']} | {r['nfe']} | {r['num_seeds']} | " + " | ".join(r[k] for k, _ in md_keys) + " |")
        lines.append("")
    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")


def plot(groups: Dict, out: Path, plot_nfe: Dict[str, int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e4e3dc",
            "grid.linewidth": 0.6,
            "axes.edgecolor": "#8a897f",
            "axes.labelcolor": "#3d3c36",
            "xtick.color": "#5c5b53",
            "ytick.color": "#5c5b53",
            "lines.linewidth": 2,
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
        }
    )
    games = sorted({g[0] for g in groups})
    fig_width = max(7.0, 4.2 * len(games))
    legend_cols = max(1, int(fig_width // 3.4))  # labels are up to ~30 characters

    def add_legend(fig, handles, labels_):
        fig.legend(handles, labels_, loc="lower center", ncol=min(legend_cols, len(labels_)), frameon=False)

    def legend_space(fig, labels_):
        rows = -(-len(labels_) // min(legend_cols, len(labels_)))
        return min(0.45, (0.12 + 0.25 * rows) / fig.get_size_inches()[1])

    methods = [m for m in METHOD_STYLE if any(g[1] == m for g in groups)]
    methods += sorted({g[1] for g in groups} - set(methods))

    def chosen(game, method):
        """Group shown in the curves: the method's main sampling budget (e.g. MeanFlow 1 step, EDM 3 steps)."""
        nfes = sorted(nfe for (g, m, nfe) in groups if g == game and m == method)
        if not nfes:
            return None
        want = plot_nfe.get(method, nfes[0])
        return (game, method, want if want in nfes else nfes[0])

    # PSNR and delta-PSNR vs rollout step
    for curve, ylabel, fname in (
        ("psnr", "PSNR vs ground truth (dB)", "rollout_psnr.png"),
        ("delta_psnr", "ΔPSNR, real − random actions (dB)", "rollout_delta_psnr.png"),
        ("lpips", "LPIPS vs ground truth", "rollout_lpips.png"),
    ):
        fig, axes = plt.subplots(1, len(games), figsize=(fig_width, 3.4), squeeze=False)
        drawn = False
        for ax, game in zip(axes[0], games):
            for method in methods:
                g = chosen(game, method)
                if g is None or curve not in groups[g][0]["curves"]:
                    continue
                ys = np.array([r["curves"][curve] for r in groups[g]])
                mean = ys.mean(0)
                t = np.arange(1, len(mean) + 1)
                color, _ = style(method)
                ax.plot(t, mean, color=color, label=f"{label(method)} ({g[2]} step{'s' if g[2] > 1 else ''})")
                if len(ys) > 1:
                    ax.fill_between(t, mean - ys.std(0), mean + ys.std(0), color=color, alpha=0.15, linewidth=0)
                drawn = True
            ax.set_title(game, loc="left", fontsize=10, color="#1f1e1b")
            ax.set_xlabel("Rollout step (frames)")
            ax.set_xscale("log")
        axes[0][0].set_ylabel(ylabel)
        if drawn:
            handles, labels_ = axes[0][0].get_legend_handles_labels()
            add_legend(fig, handles, labels_)
            fig.tight_layout(rect=(0, legend_space(fig, labels_), 1, 1))
            fig.savefig(out / fname, dpi=180)
        plt.close(fig)

    # Quality vs number of sampling steps, and quality vs speed
    for xkey, xlabel, fname, logx in (
        ("nfe", "Sampling steps (NFE)", "quality_vs_nfe.png", True),
        ("speed/fps", "Frames per second (batch 1)", "quality_vs_fps.png", True),
    ):
        fig, axes = plt.subplots(1, len(games), figsize=(fig_width, 3.4), squeeze=False)
        for ax, game in zip(axes[0], games):
            for method in methods:
                pts = []
                for (g, m, nfe), rs in sorted(groups.items(), key=lambda kv: kv[0][2]):
                    if g == game and m == method:
                        x = nfe if xkey == "nfe" else np.mean([r["values"][xkey] or np.nan for r in rs])
                        y = np.mean([r["values"]["next_frame/psnr"] for r in rs])
                        pts.append((x, y))
                if pts:
                    color, marker = style(method)
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, color=color, marker=marker, markersize=8, markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=label(method))
            ax.set_title(game, loc="left", fontsize=10, color="#1f1e1b")
            ax.set_xlabel(xlabel)
            if logx:
                ax.set_xscale("log", base=2 if xkey == "nfe" else 10)
        axes[0][0].set_ylabel("Next-frame PSNR (dB)")
        handles, labels_ = axes[0][0].get_legend_handles_labels()
        if handles:
            add_legend(fig, handles, labels_)
            fig.tight_layout(rect=(0, legend_space(fig, labels_), 1, 1))
            fig.savefig(out / fname, dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eval-name", type=str, default="test_ema_h300", help="Name of the eval JSON (eval_wm.py --name).")
    parser.add_argument(
        "--plot-nfe",
        type=str,
        nargs="*",
        default=["wm_edm=3", "wm_fm=4"],
        help="Sampling steps shown in the rollout curves per method (default: smallest evaluated).",
    )
    args = parser.parse_args()

    records = load(args.runs_root, args.eval_name)
    if not records:
        raise SystemExit(f"No {args.eval_name}.json found under {args.runs_root}/*/*/seed*/eval/")
    args.out.mkdir(parents=True, exist_ok=True)
    groups = aggregate(records)
    write_tables(groups, args.out)
    plot(groups, args.out, dict((k, int(v)) for k, v in (s.split("=") for s in args.plot_nfe)))
    print(f"{len(records)} results ({len({(r['method'], r['game'], r['seed']) for r in records})} runs) -> {args.out}")
    print((args.out / "results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
