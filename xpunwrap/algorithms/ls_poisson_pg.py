from .._dtype_utils import complex_dtype_for_real, real_pi
from .._ndarray_backend import xp
from ._plane_utils import restore_mean_plane


def algo_ls_poisson_pg(phase_wrapped: xp.ndarray, restore_plane: bool = False) -> xp.ndarray:
    """
    Least-squares unwrapping with periodic gradient enforcement.

    Parameters
    ----------
    phase_wrapped : xp.ndarray
        Wrapped phase, shape (H, W) or (N, H, W), values in [-pi, pi).
    restore_plane : bool, optional
        If True, reintroduce the mean wrapped gradient plane to preserve linear
        ramps lost in the Poisson null space. Default False.

    Returns
    -------
    xp.ndarray
        Unwrapped phase with the same shape as the input.

    Notes
    -----
    This implementation enforces periodic boundary conditions on the wrapped
    gradients prior to solving the Poisson equation.

    References
    ----------
    .. [1] D. C. Ghiglia and M. D. Pritt, "Two-Dimensional Phase Unwrapping:
       Theory, Algorithms, and Software," Wiley, 1998.
    """
    input_2d = False
    if phase_wrapped.ndim != 3:
        if phase_wrapped.ndim == 2:
            input_2d = True
            phase_wrapped = xp.expand_dims(phase_wrapped, axis=0)
        else:
            raise ValueError('phase_wrapped.ndim must be 2 or 3')

    # periodic gradients
    gx, gy = wrapped_gradients_stack(phase_wrapped)
    gx, gy = enforce_periodic_gradients_stack(gx, gy)
    rhs = divergence_stack(gx, gy)
    phi = poisson_solve_fft_stack(rhs)
    # invert in this case
    phi *= -1
    if restore_plane:
        phi = restore_mean_plane(phi, phase_wrapped)
    if input_2d:
        phi = phi[0]
    return phi


def wrap_phase(x):
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


def wrapped_gradients_stack(phi):
    """
    Wrapped forward gradients for a stack of phases.

    Parameters
    ----------
    phi : xp.ndarray
        Wrapped phase, shape (N, H, W).

    Returns
    -------
    gx, gy : xp.ndarray
        Wrapped gradients with shape (N, H, W).
    """
    gx = wrap_phase(phi[:, :, 1:] - phi[:, :, :-1])
    gy = wrap_phase(phi[:, 1:, :] - phi[:, :-1, :])

    gx = xp.pad(gx, ((0, 0), (0, 0), (0, 1)))
    gy = xp.pad(gy, ((0, 0), (0, 1), (0, 0)))

    return gx, gy


def enforce_periodic_gradients_stack(gx, gy):
    """
    Enforce periodic boundary conditions on wrapped gradients.

    Parameters
    ----------
    gx, gy : xp.ndarray
        Gradient stacks, shape (N, H, W).

    Returns
    -------
    gx, gy : xp.ndarray
        Periodic gradients, shape (N, H, W).
    """
    # x-direction periodicity
    gx[:, :, -1] = gx[:, :, 0]

    # y-direction periodicity
    gy[:, -1, :] = gy[:, 0, :]

    return gx, gy


def divergence_stack(gx, gy):
    """
    Compute divergence of wrapped gradients for a stack.

    Parameters
    ----------
    gx, gy : xp.ndarray
        Gradient stacks, shape (N, H, W).

    Returns
    -------
    xp.ndarray
        Divergence, shape (N, H, W).
    """
    div = xp.zeros_like(gx)

    div[:, :, :-1] += gx[:, :, :-1]
    div[:, :, 1:] -= gx[:, :, :-1]

    div[:, :-1, :] += gy[:, :-1, :]
    div[:, 1:, :] -= gy[:, :-1, :]

    return div


def poisson_solve_fft_stack(rhs):
    """
    Solve the periodic Poisson equation in the Fourier domain.

    Parameters
    ----------
    rhs : xp.ndarray
        Right-hand side, shape (N, H, W).

    Returns
    -------
    xp.ndarray
        Solution, shape (N, H, W).
    """
    N, H, W = rhs.shape
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
    rhs_hat = xp.fft.fft2(rhs.astype(complex_dtype, copy=False), axes=(-2, -1))
    phi_hat = rhs_hat / denom
    phi_hat[:, 0, 0] = complex_dt.type(0)

    out = xp.real(xp.fft.ifft2(phi_hat, axes=(-2, -1)))
    return out.astype(dtype, copy=False)
