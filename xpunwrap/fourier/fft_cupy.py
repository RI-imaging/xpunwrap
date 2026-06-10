from typing import Any

from .base import FFTEngine

try:
    import cupy as _cp
except Exception:  # pragma: no cover - cupy optional
    _cp = None


class FFTEngineCupy(FFTEngine):
    """FFT engine using :mod:`cupy.fft` (GPU)."""

    is_available = _cp is not None
    backend_expected = "cupy"

    def fft2(self, data: Any) -> Any:
        return _cp.fft.fft2(data, axes=(-2, -1))

    def ifft2(self, data: Any) -> Any:
        return _cp.fft.ifft2(data, axes=(-2, -1))
