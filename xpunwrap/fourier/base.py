from abc import ABC, abstractmethod

import numpy as np


class FFTEngine(ABC):
    """2D FFT over the last two axes of an (N, H, W) stack.

    Subclasses provide forward and inverse complex 2D transforms using the
    NumPy normalisation convention (``ifft2`` scaled by ``1 / (H * W)``).
    """

    #: Whether this engine can be used in the current environment.
    is_available = False
    #: Array backend this FFT engine operates on (``"numpy"`` or ``"cupy"``).
    backend_expected = None

    @abstractmethod
    def fft2(self, data: np.ndarray) -> np.ndarray:
        """Forward 2D FFT over the last two axes.

        Parameters
        ----------
        data : ndarray, shape (..., H, W)
            Complex input array.

        Returns
        -------
        ndarray
            Complex spectrum, same shape as ``data``.
        """

    @abstractmethod
    def ifft2(self, data: np.ndarray) -> np.ndarray:
        """Inverse 2D FFT over the last two axes.

        Normalised by ``1 / (H * W)`` (NumPy convention).

        Parameters
        ----------
        data : ndarray, shape (..., H, W)
            Complex spectrum.

        Returns
        -------
        ndarray
            Complex spatial array, same shape as ``data``.
        """
