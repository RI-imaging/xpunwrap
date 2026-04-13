from .._dtype_utils import complex_dtype_for_real, real_pi
from .._ndarray_backend import xp
from ._plane_utils import restore_mean_plane


def algo_ls_poisson(
    phase_wrapped: xp.ndarray,
    restore_plane: bool = False,
) -> xp.ndarray:
    """
    Batched 2D phase unwrapping using a least-squares Poisson solver.

    Parameters
    ----------
    phase_wrapped : xp.ndarray
        Wrapped phase, shape (N, H, W) or (H, W), values in [-pi, pi).
    restore_plane : bool, optional
        If True, add back the average wrapped gradient plane (avoids losing
        linear ramps inherent to the input). Default False.

    Returns
    -------
    phase_unwrapped : xp.ndarray
        Unwrapped phase, same shape as input.

    References
    ----------
    .. [1] D. C. Ghiglia and M. D. Pritt, "Two-Dimensional Phase Unwrapping:
       Theory, Algorithms, and Software," Wiley, 1998.
    """
    input_2d = False
    if phase_wrapped.ndim == 2:
        input_2d = True
        phase_wrapped = xp.expand_dims(phase_wrapped, axis=0)
    elif phase_wrapped.ndim != 3:
        raise ValueError("phase_wrapped must have shape (H, W) or (N, H, W).")

    N, H, W = phase_wrapped.shape
    dtype = phase_wrapped.dtype
    pi = real_pi(xp, dtype)
    two_pi = dtype.type(2) * pi

    # Wrapped gradients (forward differences)
    gx = xp.diff(phase_wrapped, axis=2, append=phase_wrapped[:, :, -1:])
    gy = xp.diff(phase_wrapped, axis=1, append=phase_wrapped[:, -1:, :])

    # Wrap gradients to [-pi, pi)
    gx = (gx + pi) % two_pi - pi
    gy = (gy + pi) % two_pi - pi

    # Divergence of wrapped gradients
    div_x = xp.diff(gx, axis=2, prepend=gx[:, :, :1])
    div_y = xp.diff(gy, axis=1, prepend=gy[:, :1, :])
    div = div_x + div_y

    # Fourier domain Poisson solve (batched)
    ky = xp.fft.fftfreq(H).astype(dtype, copy=False).reshape(1, H, 1)
    kx = xp.fft.fftfreq(W).astype(dtype, copy=False).reshape(1, 1, W)

    # Laplacian eigenvalues
    complex_dtype = complex_dtype_for_real(xp, dtype)
    complex_dt = xp.dtype(complex_dtype)
    complex_one = xp.asarray(1j, dtype=complex_dt)
    two_pi_c = xp.asarray(two_pi, dtype=complex_dt)
    kx_c = kx.astype(complex_dtype, copy=False)
    ky_c = ky.astype(complex_dtype, copy=False)
    denom = (
        (two_pi_c * complex_one * kx_c) ** 2
        + (two_pi_c * complex_one * ky_c) ** 2
    )
    denom[:, 0, 0] = complex_dt.type(1)  # avoid division by zero

    # FFT over spatial axes only
    div_hat = xp.fft.fft2(div.astype(complex_dtype, copy=False), axes=(1, 2))
    phi_hat = div_hat / denom
    phi_hat[:, 0, 0] = complex_dt.type(0)  # enforce zero-mean solution

    # Inverse FFT
    phase_unwrapped = xp.fft.ifft2(phi_hat, axes=(1, 2)).real
    phase_unwrapped = phase_unwrapped.astype(dtype, copy=False)

    if restore_plane:
        phase_unwrapped = restore_mean_plane(phase_unwrapped, phase_wrapped)

    if input_2d:
        phase_unwrapped = phase_unwrapped[0]
    return phase_unwrapped
