"""
Plot stacked bars for the GPU-only pipeline benchmark in a single figure.

Run:
    python tests/plot_benchmark_pipeline_cupy.py

Reads tests/benchmark_pipeline_cupy.json written by
tests/test_benchmark_pipeline_cupy.py and outputs
tests/benchmark_pipeline_cupy.png with subplots:
    - others (non-weighted, non-tvl1)
    - ls_weighted
    - tvl1
Each subplot has its own y-axis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
RESULTS_JSON = HERE / "benchmark_pipeline_cupy.json"
OUTPUT_PNG = HERE / "benchmark_pipeline_cupy.png"


def load_results() -> List[Dict[str, object]]:
    if not RESULTS_JSON.exists():
        raise FileNotFoundError(
            "benchmark_pipeline_cupy.json not found. Run pytest on tests/test_benchmark_pipeline_cupy.py first."
        )
    return json.loads(RESULTS_JSON.read_text())


def _plot_subplot(ax, results_map, algos, title):
    fontsize = 20
    if not algos:
        ax.text(0.5, 0.5, "not measured", ha="center", va="center")
        ax.set_xticks([])
        ax.set_title(title, fontsize=fontsize)
        return

    comps = [
        ("upload_ms", "upload"),
        ("phase_retrieval_ms", "phase retrieval"),
        ("unwrap_ms", "phase unwrapping"),
        ("download_ms", "download"),
    ]
    # Colorblind-friendly Okabe-Ito palette (muted, distinct)
    colors = [
        "#56B4E9",  # upload - sky blue
        "#E69F00",  # phase retrieval - orange
        "#009E73",  # gpu unwrapping - bluish green
        "#CC79A7",  # download - reddish purple
    ]

    x = np.arange(len(algos))
    bottoms = np.zeros_like(x, dtype=float)
    for (key, label), color in zip(comps, colors):
        heights = [results_map[a][key] for a in algos]
        ax.bar(
            x,
            heights,
            bottom=bottoms,
            label=label,
            color=color,
            edgecolor="white",
            yerr=None,
            error_kw={
                "ecolor": "white",
                "elinewidth": 1.5,
                "capsize": 5,
                "capthick": 1.5,
            },
        )
        bottoms += np.asarray(heights)

    ax.set_xticks(x)
    xticks = [a.replace("algo_", "") for a in algos]
    xticks = [x.replace("_", " ") for x in xticks]
    if "poisson" in algos[0]:
        xticks[0] = "LS Poisson"
        xticks[1] = "LSP Per Grad"
    ax.set_xticklabels(labels=xticks, fontsize=fontsize-4)
    ax.set_ylabel("Time (ms)", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def plot(results: List[Dict[str, object]]) -> None:
    plt.style.use("dark_background")
    results_map = {r["algorithm"]: r for r in results}
    algos_all = sorted(results_map.keys())

    base_algos = [a for a in algos_all if a not in {"algo_ls_weighted", "algo_tvl1"}]
    weighted_algos = [a for a in algos_all if a == "algo_ls_weighted"]
    tvl1_algos = [a for a in algos_all if a == "algo_tvl1"]

    fig = plt.figure(figsize=(17, 5), facecolor="#181C24")
    widths = [2.2, 1.0, 1.0, 0.6]  # extra space for legend
    gs = fig.add_gridspec(1, 4, width_ratios=widths, wspace=0.25)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax_legend = fig.add_subplot(gs[0, 3])
    ax_legend.axis("off")

    _plot_subplot(ax0, results_map, base_algos, "CuPy Pipeline Steps")
    _plot_subplot(ax1, results_map, weighted_algos, "CuPy pipeline (ls_weighted)")
    _plot_subplot(ax2, results_map, tvl1_algos, "CuPy pipeline (tvl1)")

    fig.subplots_adjust(wspace=0.3, left=0.05, right=0.95, top=0.9, bottom=0.15)
    handles, labels = ax0.get_legend_handles_labels()
    ax_legend.legend(handles[::-1], labels[::-1], title="Step", loc="center")

    for ax in (ax0, ax1, ax2):
        ax.set_facecolor("#181C24")

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=200)
    print(f"Saved {OUTPUT_PNG}")


def main():
    plot(load_results())


if __name__ == "__main__":
    main()
