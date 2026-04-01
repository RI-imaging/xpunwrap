"""
Benchmark unwrap_phase_gpu algorithms on the hologram_cell dataset.

Run with:
    pytest -q tests/test_benchmark_algorithms.py

Uses the pytest-benchmark plugin to measure timings. Results are also
persisted to tests/benchmark_results.json for downstream plotting.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import qpretrieve

import unwrap_phase_gpu as upg

BENCH_RESULTS: List[Dict[str, object]] = []
RESULTS_PATH = Path(__file__).parent / "benchmark_results.json"
DATA_PATH = Path(__file__).parent / "data" / "hologram_cell.npz"
STACK_REPEATS = 32  # aim to fill ~8 GB with float64 on 1024x1024 tiles
TILE_SHAPE = (512, 512)  # will tile to this size


def _synchronize(xp_module) -> None:
    """Synchronize GPU work if the backend is CuPy."""
    if hasattr(xp_module, "cuda"):
        xp_module.cuda.Stream.null.synchronize()


@pytest.fixture
def benchmark_cell_phase_data(backend):
    """
    Load the hologram cell dataset and build a larger phase stack.

    This mirrors the pipeline in tests/conftest.py but scales the Z-stack
    to stress the algorithms for benchmarking.
    """
    # Build the stack in NumPy to provide a host copy for transfer benchmarks.
    qpretrieve.set_ndarray_backend("numpy")
    edata_np = np.load(DATA_PATH)
    holo = qpretrieve.OffAxisHologram(data=edata_np["data"])
    bg = qpretrieve.OffAxisHologram(data=edata_np["bg_data"])
    holo.run_pipeline(filter_name="disk", filter_size=1 / 2, scale_to_filter=True)
    bg.process_like(holo)
    phase_wrp_np = np.asarray(holo.phase - bg.phase, dtype=np.float64)
    # Tile to target spatial size
    h, w = phase_wrp_np.shape[-2:]
    tile_h = int(np.ceil(TILE_SHAPE[0] / h))
    tile_w = int(np.ceil(TILE_SHAPE[1] / w))
    phase_wrp_np = np.tile(phase_wrp_np, (tile_h, tile_w))
    phase_wrp_np = phase_wrp_np[:TILE_SHAPE[0], :TILE_SHAPE[1]]
    phase_wrp_np = np.repeat(phase_wrp_np, repeats=STACK_REPEATS, axis=0)

    xp = upg.get_ndarray_backend()
    if backend == "cupy":
        # Keep both host and device copies for timing transfers.
        phase_wrp_xp = xp.asarray(phase_wrp_np)
    else:
        phase_wrp_xp = phase_wrp_np

    return {"np_stack": phase_wrp_np, "xp_stack": phase_wrp_xp}


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    """Start fresh: remove any stale results from prior runs."""
    BENCH_RESULTS.clear()
    RESULTS_PATH.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "algo_name",
    sorted(
        name for name in upg.algos_available().keys()
    ),
)
def test_benchmark_algorithms(benchmark, backend, benchmark_cell_phase_data, algo_name):
    """
    Measure runtime via pytest-benchmark for each backend + algorithm.

    The `backend` and `benchmark_cell_phase_data` fixtures handle data prep.
    """
    xp = upg.get_ndarray_backend()
    algo = upg.algos_available()[algo_name]
    host_stack = benchmark_cell_phase_data["np_stack"]
    device_stack = benchmark_cell_phase_data["xp_stack"]

    h2d_times: List[float] = []
    proc_times: List[float] = []
    d2h_times: List[float] = []
    validated_host_array = False

    def run():
        if backend == "cupy":
            # Ensure we're timing a true host->device copy.
            nonlocal validated_host_array
            if not validated_host_array:
                assert not hasattr(host_stack, "__cuda_array_interface__")
                validated_host_array = True
            _synchronize(xp)
            # Host to device
            start = time.perf_counter()
            stack_gpu = xp.asarray(host_stack)
            _synchronize(xp)
            h2d_ms = (time.perf_counter() - start) * 1000.0
        else:
            stack_gpu = device_stack
            h2d_ms = 0.0

        # Processing
        start = time.perf_counter()
        out = algo(stack_gpu)
        _synchronize(xp)
        proc_ms = (time.perf_counter() - start) * 1000.0

        # Device to host
        if backend == "cupy":
            start = time.perf_counter()
            _ = out.get()
            _synchronize(xp)
            d2h_ms = (time.perf_counter() - start) * 1000.0
        else:
            d2h_ms = 0.0

        h2d_times.append(h2d_ms)
        proc_times.append(proc_ms)
        d2h_times.append(d2h_ms)

    # Use pedantic to control rounds/iterations to avoid auto-calibration surprises.
    benchmark.pedantic(run, rounds=5, iterations=1)

    def _mean_std(arr):
        arr_np = np.asarray(arr, dtype=float)
        return float(arr_np.mean()), float(arr_np.std(ddof=0))

    h2d_ms, std_h2d = _mean_std(h2d_times)
    proc_ms, std_proc = _mean_std(proc_times)
    d2h_ms, std_d2h = _mean_std(d2h_times)

    BENCH_RESULTS.append(
        {
            "algorithm": algo_name,
            "backend": backend,
            "rounds": 5,
            "iterations": 1,
            "h2d_ms": h2d_ms,
            "proc_ms": proc_ms,
            "d2h_ms": d2h_ms,
            "std_h2d_ms": std_h2d,
            "std_proc_ms": std_proc,
            "std_d2h_ms": std_d2h,
        }
    )
    # Write incrementally so the file exists even if the session aborts later.
    RESULTS_PATH.write_text(json.dumps(BENCH_RESULTS, indent=2))


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Persist aggregated benchmark results for plotting after the test run."""
    if not BENCH_RESULTS:
        return
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(BENCH_RESULTS, indent=2))
