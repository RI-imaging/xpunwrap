import multiprocessing as mp

import numpy as np
import pyfftw

from .base import FFTEngine


class FFTEnginePyFFTW(FFTEngine):
    """FFT engine using `pyFFTW <https://pyfftw.readthedocs.io/>`_ (CPU).

    Transforms run on byte-aligned :class:`pyfftw.FFTW` plan objects, cached
    per ``(shape, dtype, direction)`` so repeated transforms of the same shape
    reuse the plan. The inverse transform keeps pyFFTW's default
    ``normalise_idft=True``, matching :func:`numpy.fft.ifft2`.

    pyFFTW operates on NumPy arrays only; it cannot handle CuPy arrays.
    """

    is_available = True
    backend_expected = "numpy"

    def __init__(self):
        # (shape, dtype_str, direction) -> (in_arr, out_arr, fftw_obj)
        self._plans = {}

    def _plan(self, shape, dtype, direction):
        """Retrieve or create a cached pyFFTW plan.

        Parameters
        ----------
        shape : tuple of int
            Array shape, e.g. ``(N, H, W)``.
        dtype : dtype-like
            Complex dtype for the transform (e.g. ``numpy.complex64``).
        direction : {"FFTW_FORWARD", "FFTW_BACKWARD"}
            Transform direction.

        Returns
        -------
        tuple
            ``(in_arr, out_arr, fftw_obj)`` — byte-aligned input/output
            arrays and the compiled :class:`pyfftw.FFTW` plan.
        """
        key = (shape, np.dtype(dtype).str, direction)
        plan = self._plans.get(key)
        if plan is None:
            in_arr = pyfftw.empty_aligned(shape, dtype=dtype)
            out_arr = pyfftw.empty_aligned(shape, dtype=dtype)
            fftw_obj = pyfftw.FFTW(in_arr, out_arr, axes=(-2, -1),
                                   direction=direction,
                                   threads=mp.cpu_count())
            plan = (in_arr, out_arr, fftw_obj)
            self._plans[key] = plan
        return plan

    def fft2(self, data):
        """Forward 2D FFT using a cached pyFFTW plan.

        Parameters
        ----------
        data : numpy.ndarray, shape (..., H, W)
            Complex input array. Copied into a byte-aligned buffer before the
            transform.

        Returns
        -------
        numpy.ndarray
            Complex spectrum, same shape as ``data``. A copy of the internal
            output buffer is returned so the plan can be safely reused.
        """
        in_arr, out_arr, fftw_obj = self._plan(
            data.shape, data.dtype, "FFTW_FORWARD")
        in_arr[:] = data
        fftw_obj()
        # out_arr is reused by the cached plan, so return a copy.
        return out_arr.copy()

    def ifft2(self, data):
        """Inverse 2D FFT using a cached pyFFTW plan.

        Normalised by ``1 / (H * W)`` (NumPy convention).

        Parameters
        ----------
        data : numpy.ndarray, shape (..., H, W)
            Complex spectrum. Copied into a byte-aligned buffer before the
            transform.

        Returns
        -------
        numpy.ndarray
            Complex spatial array, same shape as ``data``. A copy of the
            internal output buffer is returned so the plan can be safely
            reused.
        """
        in_arr, out_arr, fftw_obj = self._plan(
            data.shape, data.dtype, "FFTW_BACKWARD")
        in_arr[:] = data
        fftw_obj()
        return out_arr.copy()
