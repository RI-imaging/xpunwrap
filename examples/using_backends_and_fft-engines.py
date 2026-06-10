"""NDArray backend and FFT Engine Selection

Demonstrates how to choose between the NumPy (CPU) and CuPy (GPU) array
backends and how the FFT engine is selected automatically to match.

The FFT engine is only relevant for the least-squares Poisson solvers
(``algo_ls_poisson`` and ``algo_ls_poisson_pg``); the weighted solver
(``algo_ls_weighted``) is Jacobi-based and uses no FFT.

Run::

    python examples/using_backends_and_fft-engines.py

"""
from __future__ import annotations

import numpy as np

import xpunwrap
import xpunwrap.fourier as fourier

# 1. Dumm wrapped phase data (N=1, H=64, W=64)
rng = np.random.default_rng(0)
_tx = np.linspace(-3.0, 3.0, 64)
_x, _y = np.meshgrid(_tx, _tx)
_phase = (20.0 * np.exp(-0.25 * (_x ** 2 + _y ** 2)) +
          rng.normal(0, 0.3, (64, 64)))
wrapped_np = np.angle(np.exp(1j * _phase))[None, ...].astype(np.float32)

# 2. NumPy array backend → FFT engine auto-selects pyFFTW (if installed)
xpunwrap.set_ndarray_backend("numpy")
xp = xpunwrap.get_ndarray_backend()

fft_cls = fourier.get_best_engine()
print(f"Array backend : {xp.__name__} (CPU)")
print(f"FFT engine    : {fft_cls.__name__}")

wrapped = xp.asarray(wrapped_np)
result_auto = xpunwrap.algo_ls_poisson(wrapped, restore_plane=False)
print(f"Result shape  : {result_auto.shape}, dtype: {result_auto.dtype}")

# 3. Force the NumPy FFT engine via PREFERRED_ENGINE
fourier.PREFERRED_ENGINE = "FFTEngineNumpy"
fft_cls_forced = fourier.get_best_engine()
print(f"\nArray backend : {xp.__name__} (CPU)")
print(f"Preferred FFT : {fourier.PREFERRED_ENGINE}")
print(f"FFT engine    : {fft_cls_forced.__name__}")

result_numpy_fft = xpunwrap.algo_ls_poisson(wrapped, restore_plane=False)
print(f"Result shape  : {result_numpy_fft.shape}, "
      f"dtype: {result_numpy_fft.dtype}")

# Both FFT engines produce the same answer (within float32 rounding).
max_diff = float(np.max(np.abs(
    np.asarray(result_auto) - np.asarray(result_numpy_fft)
)))
print(f"Max |auto - numpy| : {max_diff:.2e}  (should be near zero)")

# Reset so the rest of this script uses auto-selection again.
fourier.PREFERRED_ENGINE = None

# 4. CuPy array backend (skipped when CuPy is not installed)
try:
    import cupy  # noqa: F401

    _cupy_available = True
except ImportError:
    _cupy_available = False

if _cupy_available:
    xpunwrap.set_ndarray_backend("cupy")
    xp_gpu = xpunwrap.get_ndarray_backend()
    fft_cls_gpu = fourier.get_best_engine()
    print(f"\nArray backend : {xp_gpu.__name__} (GPU)")
    print(f"FFT engine    : {fft_cls_gpu.__name__}")

    wrapped_gpu = xp_gpu.asarray(wrapped_np)
    result_gpu = xpunwrap.algo_ls_poisson(wrapped_gpu, restore_plane=False)
    print(f"Result shape  : {result_gpu.shape}, dtype: {result_gpu.dtype}")

    # Compare GPU result against CPU result.
    max_diff_gpu = float(np.max(np.abs(
        np.asarray(result_auto) - result_gpu.get()
    )))
    print(f"Max |cpu - gpu| : {max_diff_gpu:.2e}")
else:
    print("\nCuPy not installed — GPU section skipped.")

# 5. Use xp directly for array operations alongside the unwrapper
xpunwrap.set_ndarray_backend("numpy")
xp = xpunwrap.get_ndarray_backend()

wrapped = xp.asarray(wrapped_np)
result = xpunwrap.algo_ls_poisson(wrapped, restore_plane=False)

# All array ops go through xp, so this code is backend-agnostic.
residual = xp.angle(xp.exp(1j * result)) - xp.angle(xp.exp(1j * wrapped))
print(f"\nMean abs residual (wrapped): {float(
    xp.mean(xp.abs(residual))):.4f} rad")

"""
Example output:

Array backend : numpy (CPU)
FFT engine    : FFTEnginePyFFTW
Result shape  : (1, 64, 64), dtype: float32

Array backend : numpy (CPU)
Preferred FFT : FFTEngineNumpy
FFT engine    : FFTEngineNumpy
Result shape  : (1, 64, 64), dtype: float32
Max |auto - numpy| : 4.77e-06  (should be near zero)

Array backend : cupy (GPU)
FFT engine    : FFTEngineCupy
Result shape  : (1, 64, 64), dtype: float32
Max |cpu - gpu| : 4.29e-06

Mean abs residual (wrapped): 0.1254 rad
"""
