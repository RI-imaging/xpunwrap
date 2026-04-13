"""
Benchmark pipeline on GPU only:
upload -> phase retrieval -> unwrapping -> download.

Run:
    pytest -q tests/test_benchmark_pipeline_cupy.py

Results are written to tests/benchmark_pipeline_cupy.json and plotted by
tests/plot_benchmark_pipeline_cupy.py.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import qpretrieve

import xpunwrap

DATA_PATH = Path(__file__).parent.parent / "data" / "hologram_cell.npz"
RESULTS_PATH = Path(__file__).parent / "benchmark_pipeline_cupy.json"
STACK_REPEATS = 256
TILE_SHAPE = (256, 256)


def _synchronize(xp_module) -> None:
    if hasattr(xp_module, "cuda"):
        xp_module.cuda.Stream.null.synchronize()


def _load_and_tile_raw() -> Dict[str, np.ndarray]:
    edata_np = np.load(DATA_PATH)
    data = np.asarray(edata_np["data"], dtype=np.float64)
    bg_data = np.asarray(edata_np["bg_data"], dtype=np.float64)

    h, w = data.shape[-2:]
    tile_h = int(np.ceil(TILE_SHAPE[0] / h))
    tile_w = int(np.ceil(TILE_SHAPE[1] / w))
    data = np.tile(data, (tile_h, tile_w))
    data = data[:TILE_SHAPE[0], :TILE_SHAPE[1]]
    bg_data = np.tile(bg_data, (tile_h, tile_w))
    bg_data = bg_data[:TILE_SHAPE[0], :TILE_SHAPE[1]]
    return {"data": data, "bg": bg_data}


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("cupy") is None,
    reason="cupy not installed",
)
@pytest.mark.parametrize(
    "algo_name",
    sorted(
        name for name in xpunwrap.algos_available().keys()
        # if name != "algo_tvl1"
    ),
)
def test_benchmark_pipeline_cupy(algo_name):
    xpunwrap.set_ndarray_backend("cupy")
    qpretrieve.set_ndarray_backend("cupy")
    xp = xpunwrap.get_ndarray_backend()
    algo = xpunwrap.algos_available()[algo_name]

    raw = _load_and_tile_raw()

    upload_times: List[float] = []
    retr_times: List[float] = []
    unwrap_times: List[float] = []
    download_times: List[float] = []

    repeats = 3

    # Warmup once to build FFT plans etc.
    _synchronize(xp)
    holo_gpu = xp.asarray(raw["data"])
    bg_gpu = xp.asarray(raw["bg"])
    holo_obj = qpretrieve.OffAxisHologram(data=holo_gpu)
    bg_obj = qpretrieve.OffAxisHologram(data=bg_gpu)
    holo_obj.run_pipeline(
        filter_name="disk",
        filter_size=1 / 2,
        scale_to_filter=True,
    )
    bg_obj.process_like(holo_obj)
    phase_wrp = xp.asarray(holo_obj.phase - bg_obj.phase, dtype=xp.float64)
    phase_wrp = xp.repeat(phase_wrp, repeats=STACK_REPEATS, axis=0)
    algo(phase_wrp)
    _synchronize(xp)

    for _ in range(repeats):
        _synchronize(xp)
        start = time.perf_counter()
        holo_gpu = xp.asarray(raw["data"])
        bg_gpu = xp.asarray(raw["bg"])
        _synchronize(xp)
        upload_times.append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        holo_obj = qpretrieve.OffAxisHologram(data=holo_gpu)
        bg_obj = qpretrieve.OffAxisHologram(data=bg_gpu)
        holo_obj.run_pipeline(
            filter_name="disk",
            filter_size=1 / 2,
            scale_to_filter=True,
        )
        bg_obj.process_like(holo_obj)
        phase_wrp = xp.asarray(holo_obj.phase - bg_obj.phase, dtype=xp.float64)
        phase_wrp = xp.repeat(phase_wrp, repeats=STACK_REPEATS, axis=0)
        _synchronize(xp)
        retr_times.append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        out = algo(phase_wrp)
        _synchronize(xp)
        unwrap_times.append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        _ = out.get()
        _synchronize(xp)
        download_times.append((time.perf_counter() - start) * 1000.0)

    def _mean(arr):
        arr_np = np.asarray(arr, dtype=float)
        return float(arr_np.mean())

    result: Dict[str, object] = {
        "algorithm": algo_name,
        "backend": "cupy",
        "repeats": repeats,
        "upload_ms": _mean(upload_times),
        "phase_retrieval_ms": _mean(retr_times),
        "unwrap_ms": _mean(unwrap_times),
        "download_ms": _mean(download_times),
    }

    # Append to JSON
    if RESULTS_PATH.exists():
        existing = json.loads(RESULTS_PATH.read_text())
    else:
        existing = []
    existing = [r for r in existing if r.get("algorithm") != algo_name]
    existing.append(result)
    RESULTS_PATH.write_text(json.dumps(existing, indent=2))
