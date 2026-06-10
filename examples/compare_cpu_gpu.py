"""CPU vs GPU phase unwrapping speed comparison

Compares four pipelines on hologram_cell data:

- CPU  skimage    + NumPy   (qpretrieve NumPy FFT   + skimage unwrap)
- CPU  skimage    + pyFFTW  (qpretrieve pyFFTW FFT  + skimage unwrap)
- CPU  ls_poisson + pyFFTW  (qpretrieve pyFFTW FFT  + xpunwrap Poisson)
- GPU  ls_poisson + CuPy    (full CuPy pipeline)

Run::

    python examples/compare_cpu_gpu.py

"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import qpretrieve
import xpunwrap
import xpunwrap.fourier as xp_fourier

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


def _time_skimage_pipeline(
        raw: dict[str, np.ndarray],
        fft_interface,
        repeats: int = 3,
):
    """Shared timing loop for skimage-based pipelines."""
    qpretrieve.set_ndarray_backend("numpy")
    try:
        algo = xpunwrap.algos_available()["algo_skimage_unwrap"]
    except Exception:
        return None

    def _run():
        holo = qpretrieve.OffAxisHologram(raw["data"], fft_interface)
        bg = qpretrieve.OffAxisHologram(raw["bg"], fft_interface)
        holo.run_pipeline(
            filter_name="disk", filter_size=1 / 2, scale_to_filter=True)
        bg.process_like(holo)
        algo(np.asarray(holo.phase - bg.phase, dtype=np.float32))

    _run()  # warmup (important for pyFFTW plan compilation)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _run()
        times.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(times)), float(np.std(times))


def time_cpu_skimage_numpy(raw: dict[str, np.ndarray], repeats: int = 3):
    """CPU pipeline: qpretrieve (NumPy FFT) + skimage unwrap."""
    return _time_skimage_pipeline(
        raw, qpretrieve.fourier.FFTFilterNumpy, repeats)


def time_cpu_skimage_pyfftw(raw: dict[str, np.ndarray], repeats: int = 3):
    """CPU pipeline: qpretrieve (pyFFTW) + skimage unwrap."""
    return _time_skimage_pipeline(
        raw, qpretrieve.fourier.FFTFilterPyFFTW, repeats)


def time_cpu_ls_poisson_pyfftw(raw: dict[str, np.ndarray], repeats: int = 3):
    """CPU pipeline: qpretrieve (pyFFTW) + xpunwrap ls_poisson (pyFFTW)."""
    from xpunwrap.fourier import FFTEnginePyFFTW
    if FFTEnginePyFFTW is None:
        return None

    qpretrieve.set_ndarray_backend("numpy")
    xpunwrap.set_ndarray_backend("numpy")
    fft_interface = qpretrieve.fourier.FFTFilterPyFFTW
    xp_fourier.PREFERRED_ENGINE = "FFTEnginePyFFTW"
    algo = xpunwrap.algos_available()["algo_ls_poisson"]

    try:
        # Warmup
        holo = qpretrieve.OffAxisHologram(raw["data"], fft_interface)
        bg = qpretrieve.OffAxisHologram(raw["bg"], fft_interface)
        holo.run_pipeline(
            filter_name="disk", filter_size=1 / 2, scale_to_filter=True)
        bg.process_like(holo)
        algo(np.asarray(holo.phase - bg.phase, dtype=np.float32))

        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            holo = qpretrieve.OffAxisHologram(raw["data"], fft_interface)
            bg = qpretrieve.OffAxisHologram(raw["bg"], fft_interface)
            holo.run_pipeline(
                filter_name="disk", filter_size=1 / 2, scale_to_filter=True)
            bg.process_like(holo)
            algo(np.asarray(holo.phase - bg.phase, dtype=np.float32))
            times.append((time.perf_counter() - start) * 1000.0)
    finally:
        xp_fourier.PREFERRED_ENGINE = None

    return float(np.mean(times)), float(np.std(times))


def time_gpu_pipeline(raw: dict[str, np.ndarray], repeats: int = 3):
    """GPU pipeline: qpretrieve (CuPy FFT) + xpunwrap ls_poisson (CuPy FFT)."""
    try:
        import cupy as cp  # noqa: F401
    except Exception:
        return None

    qpretrieve.set_ndarray_backend("cupy")
    fft_interface = qpretrieve.fourier.FFTFilterCupy
    xpunwrap.set_ndarray_backend("cupy")
    xp = xpunwrap.get_ndarray_backend()
    algo = xpunwrap.algos_available()["algo_ls_poisson"]

    def _sync():
        if hasattr(xp, "cuda"):
            xp.cuda.Stream.null.synchronize()

    # Warmup
    _sync()
    holo_gpu = xp.asarray(raw["data"])
    bg_gpu = xp.asarray(raw["bg"])
    holo_obj = qpretrieve.OffAxisHologram(holo_gpu, fft_interface)
    bg_obj = qpretrieve.OffAxisHologram(bg_gpu, fft_interface)
    holo_obj.run_pipeline(
        filter_name="disk", filter_size=1 / 2, scale_to_filter=True)
    bg_obj.process_like(holo_obj)
    algo(xp.asarray(holo_obj.phase - bg_obj.phase, dtype=xp.float32))
    _sync()

    times = []
    for _ in range(repeats):
        _sync()
        start = time.perf_counter()
        holo_gpu = xp.asarray(raw["data"])
        bg_gpu = xp.asarray(raw["bg"])
        holo_obj = qpretrieve.OffAxisHologram(holo_gpu, fft_interface)
        bg_obj = qpretrieve.OffAxisHologram(bg_gpu, fft_interface)
        holo_obj.run_pipeline(
            filter_name="disk", filter_size=1 / 2, scale_to_filter=True)
        bg_obj.process_like(holo_obj)
        algo(xp.asarray(holo_obj.phase - bg_obj.phase, dtype=xp.float32))
        _sync()
        times.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(times)), float(np.std(times))


def main():
    raw = _load_raw_numpy()

    skimage_numpy = time_cpu_skimage_numpy(raw)
    skimage_pyfftw = time_cpu_skimage_pyfftw(raw)
    ls_pyfftw = time_cpu_ls_poisson_pyfftw(raw)
    ls_cupy = time_gpu_pipeline(raw)

    # Wong (2011) / Nature Methods colorblind-safe palette
    clr_blue, clr_green = "#56B4E9", "#009E73"
    clr_orange, clr_purple = "#D55E00", "#CC79A7"
    entries = [
        ("skimage\n+ NumPy\n(CPU)", skimage_numpy, clr_blue),
        ("skimage\n+ pyFFTW\n(CPU)", skimage_pyfftw, clr_green),
        ("ls_poisson\n+ pyFFTW\n(CPU)", ls_pyfftw, clr_orange),
        ("ls_poisson\n+ CuPy\n(GPU)", ls_cupy, clr_purple),
    ]
    entries = [(lbl, res, col) for lbl, res, col in entries if res is not None]

    labels = [e[0] for e in entries]
    means = [e[1][0] for e in entries]
    stds = [e[1][1] for e in entries]
    colors = [e[2] for e in entries]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#181C24")
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, color=colors, edgecolor="white",
           error_kw={"ecolor": "white", "elinewidth": 1.5,
                     "capsize": 5, "capthick": 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, ha="center")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Phase unwrapping speed comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_facecolor("#181C24")

    for xi, mean, std in zip(x, means, stds):
        ax.text(xi, mean + std, f"{mean:.1f} ms",
                ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=200)
    print(f"Saved {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
