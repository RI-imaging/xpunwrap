"""
Compare CPU (skimage unwrap_phase) vs GPU (ls_poisson) on hologram_cell data.

Run:
    python examples/compare_cpu_gpu.py

Outputs a dark-themed bar chart: examples/compare_cpu_gpu.png
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import qpretrieve
import unwrap_phase_gpu as upg

DATA_PATH = Path(__file__).parent.parent / "tests" / "data" / "hologram_cell.npz"
OUTPUT_PNG = Path(__file__).parent / "compare_cpu_gpu.png"
STACK_DEPTH = 256
TILE_SIZE = 256


def _load_raw_numpy() -> dict[str, np.ndarray]:
    edata = np.load(DATA_PATH)
    data = np.asarray(edata["data"], dtype=np.float32)
    bg = np.asarray(edata["bg_data"], dtype=np.float32)
    tile_h = int(np.ceil(TILE_SIZE / data.shape[-2]))
    tile_w = int(np.ceil(TILE_SIZE / data.shape[-1]))
    data = np.tile(data, (tile_h, tile_w))[:TILE_SIZE, :TILE_SIZE]
    bg = np.tile(bg, (tile_h, tile_w))[:TILE_SIZE, :TILE_SIZE]
    return {"data": data, "bg": bg}


def _stack_phase(arr: np.ndarray) -> np.ndarray:
    return np.repeat(arr[None, ...], repeats=STACK_DEPTH, axis=0)


def time_cpu_pipeline(raw: dict[str, np.ndarray], repeats: int = 3):
    qpretrieve.set_ndarray_backend("numpy")
    try:
        algo = upg.algos_available()["algo_skimage_unwrap"]
    except Exception:
        return None
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        holo = qpretrieve.OffAxisHologram(data=raw["data"])
        bg = qpretrieve.OffAxisHologram(data=raw["bg"])
        holo.run_pipeline(filter_name="disk", filter_size=1 / 2, scale_to_filter=True)
        bg.process_like(holo)
        phase_wrp = np.asarray(holo.phase - bg.phase, dtype=np.float32)
        algo(phase_wrp)
        times.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(times)), float(np.std(times))


def time_gpu_pipeline(raw: dict[str, np.ndarray], repeats: int = 3):
    try:
        import cupy as cp  # noqa: F401
    except Exception:
        return None

    upg.set_ndarray_backend("cupy")
    qpretrieve.set_ndarray_backend("cupy")
    xp = upg.get_ndarray_backend()
    algo = upg.algos_available()["algo_ls_poisson"]

    def _sync():
        if hasattr(xp, "cuda"):
            xp.cuda.Stream.null.synchronize()

    times = []
    # Warmup once
    _sync()
    holo_gpu = xp.asarray(raw["data"])
    bg_gpu = xp.asarray(raw["bg"])
    holo_obj = qpretrieve.OffAxisHologram(data=holo_gpu)
    bg_obj = qpretrieve.OffAxisHologram(data=bg_gpu)
    holo_obj.run_pipeline(filter_name="disk", filter_size=1 / 2, scale_to_filter=True)
    bg_obj.process_like(holo_obj)
    phase_wrp = xp.asarray(holo_obj.phase - bg_obj.phase, dtype=xp.float32)
    algo(phase_wrp)
    _sync()

    for _ in range(repeats):
        _sync()
        start = time.perf_counter()
        holo_gpu = xp.asarray(raw["data"])
        bg_gpu = xp.asarray(raw["bg"])
        holo_obj = qpretrieve.OffAxisHologram(data=holo_gpu)
        bg_obj = qpretrieve.OffAxisHologram(data=bg_gpu)
        holo_obj.run_pipeline(filter_name="disk", filter_size=1 / 2, scale_to_filter=True)
        bg_obj.process_like(holo_obj)
        phase_wrp = xp.asarray(holo_obj.phase - bg_obj.phase, dtype=xp.float32)
        algo(phase_wrp)
        _sync()
        times.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(times)), float(np.std(times))


def main():
    raw = _load_raw_numpy()

    cpu = time_cpu_pipeline(raw)
    gpu = time_gpu_pipeline(raw)

    labels = []
    means = []
    stds = []
    if cpu:
        labels.append("CPU skimage")
        means.append(cpu[0])
        stds.append(cpu[1])
    if gpu:
        labels.append("GPU ls_poisson")
        means.append(gpu[0])
        stds.append(gpu[1])

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='#181C24')
    x = np.arange(len(labels))
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=["#4cc9f0", "#f72585"],
        edgecolor="white",
        error_kw={
            "ecolor": "white",
            "elinewidth": 1.5,
            "capsize": 5,
            "capthick": 1.5,
        },
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Time (ms)")
    ax.set_title("CPU (skimage) vs GPU (ls_poisson)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_facecolor('#181C24')

    for xi, mean in zip(x, means):
        if mean is not None:
            ax.text(xi, mean, f"{mean:.1f} ms", ha="center", va="bottom", fontsize=8)
        else:
            ax.text(xi, 0, "GPU missing", ha="center", va="bottom", fontsize=8, rotation=90)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=200)
    print(f"Saved {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
