"""FFT engines for the least-squares Poisson solver.

The active array backend (``xp``) decides numpy-vs-cupy; this layer decides
which FFT implementation to use within the CPU (numpy) case, preferring pyFFTW
when it is installed. pyFFTW cannot operate on CuPy arrays, so the cupy array
backend always uses the cupy FFT engine.
"""
import warnings

from .._ndarray_backend import xp
from .base import FFTEngine
from .fft_numpy import FFTEngineNumpy

try:
    from .fft_pyfftw import FFTEnginePyFFTW
except ImportError:  # pragma: no cover - pyfftw optional
    FFTEnginePyFFTW = None

try:
    from .fft_cupy import FFTEngineCupy
except ImportError:  # pragma: no cover - cupy optional
    FFTEngineCupy = None

#: Optional override, e.g. set to ``"FFTEngineNumpy"`` to force the numpy FFT.
PREFERRED_ENGINE = None

_INSTANCES = {}


def get_best_engine() -> type[FFTEngine]:
    """Return the FFT engine class for the active array backend.

    CuPy arrays require the CuPy FFT engine. On the NumPy backend, pyFFTW is
    preferred when installed, otherwise the NumPy FFT engine.

    Returns
    -------
    type
        An :class:`~xpunwrap.fourier.base.FFTEngine` subclass
        (not an instance).

    Raises
    ------
    RuntimeError
        If the CuPy array backend is active but the CuPy FFT engine is
        unavailable.
    """
    if xp.is_cupy():
        if FFTEngineCupy is not None and FFTEngineCupy.is_available:
            return FFTEngineCupy
        raise RuntimeError(
            "cupy array backend is active but the cupy FFT engine is "
            "unavailable."
        )

    candidates = [FFTEnginePyFFTW, FFTEngineNumpy]
    if PREFERRED_ENGINE is not None:
        for cand in candidates:
            if (cand is not None and cand.is_available
                    and cand.__name__ == PREFERRED_ENGINE):
                return cand
        warnings.warn(
            f"Preferred FFT engine '{PREFERRED_ENGINE}' is unavailable; "
            f"falling back to the best available engine."
        )
    for cand in candidates:
        if cand is not None and cand.is_available:
            return cand
    return FFTEngineNumpy


def get_fft_engine() -> FFTEngine:
    """Return a cached instance of the best FFT engine.

    Caching the instance keeps pyFFTW plans warm across repeated solver calls.

    Returns
    -------
    FFTEngine
        Singleton instance of the selected
        :class:`~xpunwrap.fourier.base.FFTEngine`.
    """
    cls = get_best_engine()
    inst = _INSTANCES.get(cls.__name__)
    if inst is None:
        inst = cls()
        _INSTANCES[cls.__name__] = inst
    return inst
