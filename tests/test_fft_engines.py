import numpy as np
import pytest

import xpunwrap
from xpunwrap.fourier import (
    FFTEngineCupy,
    FFTEngineNumpy,
    FFTEnginePyFFTW,
    get_best_engine,
    get_fft_engine,
)

pyfftw_missing = FFTEnginePyFFTW is None
cupy_missing = FFTEngineCupy is None or not FFTEngineCupy.is_available


def test_select_numpy_when_forced(numpy_backend, preferred_engine):
    preferred_engine("FFTEngineNumpy")
    assert get_best_engine() is FFTEngineNumpy


@pytest.mark.skipif(pyfftw_missing, reason="pyfftw not installed")
def test_select_pyfftw_by_default(numpy_backend, preferred_engine):
    preferred_engine(None)
    assert get_best_engine() is FFTEnginePyFFTW


def test_get_fft_engine_caches_instance(numpy_backend):
    assert get_fft_engine() is get_fft_engine()


@pytest.mark.skipif(cupy_missing, reason="cupy not installed")
def test_select_cupy_under_cupy_backend():
    xpunwrap.set_ndarray_backend("cupy")
    try:
        assert get_best_engine() is FFTEngineCupy
    finally:
        xpunwrap.set_ndarray_backend("numpy")


@pytest.mark.skipif(pyfftw_missing, reason="pyfftw not installed")
def test_pyfftw_matches_numpy(numpy_backend, preferred_engine):
    # float64 isolates the FFT convention (normalisation) from single-precision
    # rounding, so the two engines must agree to near machine precision.
    rng = np.random.default_rng(7)
    wrapped = rng.uniform(-np.pi, np.pi, size=(3, 32, 24))

    preferred_engine("FFTEngineNumpy")
    out_np = xpunwrap.algo_ls_poisson(wrapped)

    preferred_engine("FFTEnginePyFFTW")
    out_fftw = xpunwrap.algo_ls_poisson(wrapped)

    assert np.allclose(out_np, out_fftw, rtol=1e-9, atol=1e-9)


@pytest.mark.skipif(cupy_missing, reason="cupy not installed")
def test_cupy_matches_numpy(numpy_backend, preferred_engine):
    rng = np.random.default_rng(7)
    wrapped = rng.uniform(-np.pi, np.pi, size=(3, 32, 24))

    preferred_engine("FFTEngineNumpy")
    out_np = xpunwrap.algo_ls_poisson(wrapped)

    preferred_engine("FFTEngineCupy")
    xpunwrap.set_ndarray_backend("cupy")
    _cp = xpunwrap.get_ndarray_backend()
    out_cp = xpunwrap.algo_ls_poisson(_cp.asarray(wrapped))

    assert np.allclose(out_np, out_cp.get(), rtol=1e-9, atol=1e-9)
