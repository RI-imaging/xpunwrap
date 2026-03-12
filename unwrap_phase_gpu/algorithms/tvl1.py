from .._ndarray_backend import xp


def algo_tvl1(
        phase_wrapped: xp.ndarray,
        axis=0,
        **kwargs
) -> xp.ndarray:
    """
    Total Variation / L1 Gradient Matching Unwrapping for a stack of 2D images.

    Parameters
    ----------
    phase_wrapped : xp.ndarray
        Shape (N, H, W) or (H, W)
    axis : int
        Stack axis
    kwargs :
        Passed to unwrap_2d_tvl1_gpu

    Returns
    -------
    unphase_wrapped : xp.ndarray

    References
    ----------
    .. [1] A. Chambolle and T. Pock, "A first-order primal-dual algorithm for
       convex problems with applications to imaging," J. Math. Imaging Vis.,
       vol. 40, no. 1, pp. 120-145, 2011.
    """
    if phase_wrapped.ndim == 2:
        return _unwrap_2d_tvl1_gpu(phase_wrapped, **kwargs)
    if phase_wrapped.ndim != 3:
        raise ValueError("phase_wrapped must have shape (H, W) or (N, H, W).")

    phase_wrapped = xp.moveaxis(phase_wrapped, axis, 0)

    out = xp.empty_like(phase_wrapped, dtype=xp.float32)

    for i in range(phase_wrapped.shape[0]):
        out[i] = _unwrap_2d_tvl1_gpu(phase_wrapped[i], **kwargs)

    return xp.moveaxis(out, 0, axis)


def _unwrap_2d_tvl1_gpu(
        wrapped,
        n_iters=200,
        tau=0.25,
        sigma=0.25,
):
    """
    TV-L1 phase unwrapping on GPU.

    Parameters
    ----------
    wrapped : xp.ndarray
        Wrapped phase in (-pi, pi]
    n_iters : int
        Number of primal-dual iterations
    tau, sigma : float
        Step sizes (tau * sigma <= 0.25 recommended)

    Returns
    -------
    phi : xp.ndarray
        Unwrapped phase

    References
    ----------
    .. [1] A. Chambolle and T. Pock, "A first-order primal-dual algorithm for
       convex problems with applications to imaging," J. Math. Imaging Vis.,
       vol. 40, no. 1, pp. 120-145, 2011.
    """

    wrapped = wrapped.astype(xp.float32)

    # Wrapped gradients (data term)
    gx_w, gy_w = _grad_forward(wrapped)
    gx_w = _wrap_phase(gx_w)
    gy_w = _wrap_phase(gy_w)

    # Primal variable
    phi = xp.zeros_like(wrapped)
    phi_bar = phi.copy()

    # Dual variables
    px = xp.zeros_like(wrapped)
    py = xp.zeros_like(wrapped)

    for _ in range(n_iters):
        # --- Dual update ---
        gx, gy = _grad_forward(phi_bar)

        px += sigma * (gx - gx_w)
        py += sigma * (gy - gy_w)

        # L1 proximal (projection onto |p| <= 1)
        norm = xp.maximum(1.0, xp.sqrt(px ** 2 + py ** 2))
        px /= norm
        py /= norm

        # --- Primal update ---
        div_p = _div_backward(px, py)
        phi_old = phi
        phi = phi + tau * div_p

        # Extrapolation
        phi_bar = 2 * phi - phi_old

    return phi


def _wrap_phase(x):
    """
    Wrap to (-pi, pi].

    Parameters
    ----------
    x : xp.ndarray
        Input phase values.

    Returns
    -------
    xp.ndarray
        Wrapped values in (-pi, pi].
    """
    return (x + xp.pi) % (xp.pi * 2) - xp.pi


def _grad_forward(u):
    """
    Forward differences with periodic wrap.

    Parameters
    ----------
    u : xp.ndarray
        Input array, shape (H, W).

    Returns
    -------
    gx, gy : xp.ndarray
        Forward gradients, shape (H, W).
    """
    gx = xp.roll(u, -1, axis=-1) - u
    gy = xp.roll(u, -1, axis=-2) - u
    return gx, gy


def _div_backward(px, py):
    """
    Backward divergence with periodic wrap.

    Parameters
    ----------
    px, py : xp.ndarray
        Dual variables, shape (H, W).

    Returns
    -------
    xp.ndarray
        Divergence, shape (H, W).
    """
    dx = px - xp.roll(px, 1, axis=-1)
    dy = py - xp.roll(py, 1, axis=-2)
    return dx + dy
