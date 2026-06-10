import numpy as np

from .base import FFTEngine


class FFTEngineNumpy(FFTEngine):
    """FFT engine using :mod:`numpy.fft` (CPU, always available)."""

    is_available = True
    backend_expected = "numpy"

    def fft2(self, data: np.ndarray) -> np.ndarray:
        return np.fft.fft2(data, axes=(-2, -1))

    def ifft2(self, data: np.ndarray) -> np.ndarray:
        return np.fft.ifft2(data, axes=(-2, -1))
