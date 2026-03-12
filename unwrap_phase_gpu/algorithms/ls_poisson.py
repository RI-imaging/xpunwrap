from .._ndarray_backend import xp


def algo_ls_poisson(phase_wrapped: xp.ndarray) -> xp.ndarray:
    """
    Batched 2D phase unwrapping using a least-squares Poisson solver.

    Parameters
    ----------
    phase_wrapped : xp.ndarray
        Wrapped phase, shape (N, H, W) or (H, W), values in (-pi, pi).

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

    # Wrapped gradients (forward differences)
    gx = xp.diff(phase_wrapped, axis=2, append=phase_wrapped[:, :, -1:])
    gy = xp.diff(phase_wrapped, axis=1, append=phase_wrapped[:, -1:, :])

    # Wrap gradients to [-pi, pi)
    two_pi = dtype.type(2 * xp.pi)
    gx = (gx + xp.pi) % two_pi - xp.pi
    gy = (gy + xp.pi) % two_pi - xp.pi

    # Divergence of wrapped gradients
    div_x = xp.diff(gx, axis=2, prepend=gx[:, :, :1])
    div_y = xp.diff(gy, axis=1, prepend=gy[:, :1, :])
    div = div_x + div_y

    # Fourier domain Poisson solve (batched)
    ky = xp.fft.fftfreq(H).reshape(1, H, 1)
    kx = xp.fft.fftfreq(W).reshape(1, 1, W)

    # Laplacian eigenvalues
    denom = (2 * xp.pi * 1j * kx) ** 2 + (2 * xp.pi * 1j * ky) ** 2
    denom[:, 0, 0] = 1.0  # avoid division by zero

    # FFT over spatial axes only
    div_hat = xp.fft.fft2(div, axes=(1, 2))
    phi_hat = div_hat / denom
    phi_hat[:, 0, 0] = 0.0  # enforce zero-mean solution

    # Inverse FFT
    phase_unwrapped = xp.fft.ifft2(phi_hat, axes=(1, 2)).real

    if input_2d:
        phase_unwrapped = phase_unwrapped[0]
    return phase_unwrapped
