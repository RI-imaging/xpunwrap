"""
Plot benchmark results produced by tests/test_benchmark_algorithms.py.

Run:
    python tests/plot_benchmarks_algorithms.py

Expects tests/benchmark_results.json to exist (written by the pytest run).
Produces two dark-themed PNGs:
    1) benchmark_results.png — processing time only.
    2) benchmark_results_with_transfer.png — processing + transfer times.
Each has three subplots:
    - ls_poisson variants
    - ls_weighted
    - tvl1
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
RESULTS_JSON = HERE / "benchmark_results.json"
OUTPUT_PNG = HERE / "benchmark_results.png"
OUTPUT_PNG_WITH_XFER = HERE / "benchmark_results_with_transfer.png"


def load_results() -> List[Dict[str, object]]:
    if not RESULTS_JSON.exists():
        raise FileNotFoundError(
            "benchmark_results.json not found. "
            "Run pytest on tests/test_benchmark_algorithms.py first."
        )
    return json.loads(RESULTS_JSON.read_text())


def _group_defs():
    return [
        ("ls_poisson family", ["algo_ls_poisson", "algo_ls_poisson_pg"]),
        ("ls_weighted", ["algo_ls_weighted"]),
        ("tvl1", ["algo_tvl1"]),
    ]


def _plot(results, use_transfers: bool, output_path: Path):
    plt.style.use("dark_background")
    algos_available = sorted({r["algorithm"] for r in results})
    backends = sorted(
        {r["backend"] for r in results},
        key=lambda b: (b != "numpy", b),
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    width = 0.35 if len(backends) > 1 else 0.55

    for ax, (title, algos) in zip(axes, _group_defs()):
        target_algos = [a for a in algos if a in algos_available]
        x = np.arange(len(target_algos)) if target_algos else np.arange(1)
        if not target_algos:
            ax.text(0.5, 0.5, "not measured", ha="center", va="center")
            ax.set_xticks([])
            ax.set_title(title)
            continue

        for idx, backend in enumerate(backends):
            backend_results = {
                r["algorithm"]: r for r in results
                if r["backend"] == backend
            }
            if use_transfers:
                means = [
                    backend_results[a].get("h2d_ms", 0.0)
                    + backend_results[a].get("d2h_ms", 0.0)
                    for a in target_algos
                ]
                stds = [
                    math.sqrt(
                        backend_results[a].get("std_h2d_ms", 0.0) ** 2
                        + backend_results[a].get("std_d2h_ms", 0.0) ** 2
                    )
                    for a in target_algos
                ]
            else:
                means = [backend_results[a]["proc_ms"] for a in target_algos]
                stds = [
                    backend_results[a].get("std_proc_ms", 0.0)
                    for a in target_algos
                ]

            offset = (idx - (len(backends) - 1) / 2) * width
            ax.bar(
                x + offset,
                means,
                width=width * 0.9,
                yerr=stds,
                error_kw={
                    "ecolor": "white",
                    "elinewidth": 1.5,
                    "capsize": 5,
                    "capthick": 1.5,
                },
                label=backend,
            )

        ax.set_ylabel("Time (ms)")
        ax.set_title(
            title + (" (transfer only)" if use_transfers else "")
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [a.replace("algo_", "") for a in target_algos],
            rotation=15,
            ha="right",
        )
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    axes[0].legend(title="Backend", loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved {output_path}")


def plot(results: List[Dict[str, object]]) -> None:
    _plot(results, use_transfers=False, output_path=OUTPUT_PNG)
    _plot(results, use_transfers=True, output_path=OUTPUT_PNG_WITH_XFER)

    # Also print transfer times so they are visible even if not plotted.
    print("\nTransfer times (ms, mean +/- std):")
    algos_available = sorted({r["algorithm"] for r in results})
    backends = sorted(
        {r["backend"] for r in results},
        key=lambda b: (b != "numpy", b),
    )
    for backend in backends:
        backend_results = {
            r["algorithm"]: r for r in results
            if r["backend"] == backend
        }
        for algo in algos_available:
            r = backend_results[algo]
            h2d = r.get("h2d_ms", 0.0)
            h2d_std = r.get("std_h2d_ms", 0.0)
            d2h = r.get("d2h_ms", 0.0)
            d2h_std = r.get("std_d2h_ms", 0.0)
            print(
                f"  {backend:5s} | {algo:22s} | "
                f"H2D {h2d:7.2f}+/-{h2d_std:5.2f} | "
                f"D2H {d2h:7.2f}+/-{d2h_std:5.2f}"
            )


def main():
    plot(load_results())


if __name__ == "__main__":
    main()
