"""
Shared building blocks for the least-squares Poisson unwrap algorithms.

These helpers are used by both ``ls_poisson`` and ``ls_poisson_pg``.
"""

from .._dtype_utils import complex_dtype_for_real, real_pi
from .._ndarray_backend import xp
from ..fourier import get_fft_engine


def wrap_phase(x: xp.ndarray) -> xp.ndarray:
    """
    Wrap phase to [-pi, pi).

    Parameters
    ----------
    x : xp.ndarray
        Input phase values.

    Returns
    -------
    xp.ndarray
        Wrapped values in [-pi, pi).
    """
    dtype = x.dtype
    pi = real_pi(xp, dtype)
    two_pi = dtype.type(2) * pi
    return (x + pi) % two_pi - pi


def divergence_stack(gx: xp.ndarray, gy: xp.ndarray) -> xp.ndarray:
    """
    Compute the periodic divergence of a wrapped-gradient stack.

    Parameters
    ----------
    gx, gy : xp.ndarray
        Gradient stacks, shape (N, H, W).

    Returns
    -------
    xp.ndarray
        Divergence, shape (N, H, W).
    """
    # Periodic backward-difference divergence used by the FFT Poisson solve.
    # div g = (g_x - g_x shifted right) + (g_y - g_y shifted down)
    return (
        gx - xp.roll(gx, 1, axis=2)
        + gy - xp.roll(gy, 1, axis=1)
    )


def poisson_solve_fft_stack(rhs: xp.ndarray) -> xp.ndarray:
    """
    Solve the periodic Poisson equation in the Fourier domain.

    Returns the positive-denominator solution; callers multiply by -1 to
    match the discrete Laplacian sign convention.

    Parameters
    ----------
    rhs : xp.ndarray
        Right-hand side, shape (N, H, W).

    Returns
    -------
    xp.ndarray
        Solution, shape (N, H, W).
    """
    _, H, W = rhs.shape
    dtype = rhs.dtype
    pi = real_pi(xp, dtype)
    two = dtype.type(2)

    ky = xp.fft.fftfreq(H).astype(dtype, copy=False).reshape(1, H, 1)
    kx = xp.fft.fftfreq(W).astype(dtype, copy=False).reshape(1, 1, W)

    denom = (two - two * xp.cos(two * pi * kx)) + \
            (two - two * xp.cos(two * pi * ky))

    denom[:, 0, 0] = dtype.type(1)

    complex_dtype = complex_dtype_for_real(xp, dtype)
    complex_dt = xp.dtype(complex_dtype)
    fft = get_fft_engine()
    rhs_hat = fft.fft2(rhs.astype(complex_dtype, copy=False))
    phi_hat = rhs_hat / denom
    phi_hat[:, 0, 0] = complex_dt.type(0)

    out = xp.real(fft.ifft2(phi_hat))
    return out.astype(dtype, copy=False)
