Least-Squares Poisson (algo_ls_poisson)
=======================================

This is the standard FFT-based least-squares phase unwrap. It computes wrapped
forward differences, forms their divergence, and solves a periodic Poisson
equation for the phase field. See Ghiglia and Pritt (1998).

API: :func:`xpunwrap.algo_ls_poisson<xpunwrap.algorithms.algo_ls_poisson>`

Derivation
----------

1. **Input**: a wrapped phase :math:`\phi_w` with values in :math:`[-\pi,\pi)`.
2. **Forward differences** (wrapped gradients):

   .. math::
      g_x = \phi_w((x+1)\bmod W,y) - \phi_w(x,y), \quad
      g_y = \phi_w(x,(y+1)\bmod H) - \phi_w(x,y)

3. **Wrap the gradients** into :math:`[-\pi,\pi)`:

   .. math::
      \mathrm{wrap}(u) = (u + \pi) \bmod (2\pi) - \pi

4. **Divergence** of wrapped gradients:

   .. math::
      \nabla \cdot g =
      \left(g_x(x,y) - g_x((x-1)\bmod W,y)\right) +
      \left(g_y(x,y) - g_y(x,(y-1)\bmod H)\right)

5. **Least-squares formulation** for the unwrapped phase :math:`\phi`:

   The discrete objective is:

   .. math::
      \phi^\star = \arg\min_{\phi}\sum_{x,y}\lvert \nabla \phi - g \rvert^2

   Its normal equation is the Poisson PDE:

   .. math::
      \nabla \cdot g = \nabla^2 \phi

   (This is a PDE, not an ODE.)


6. **FFT solve** in the frequency domain:

   .. math::
      \hat{\phi}(k_x,k_y) =
      \frac{\hat{\nabla \cdot g}(k_x,k_y)}
           {(2-2\cos(2\pi k_x)) + (2-2\cos(2\pi k_y))}

   Here :math:`k_x, k_y` are normalized frequencies from
   :math:`\mathrm{fftfreq}` (cycles/sample).

   The DC component is set to zero to enforce a zero-mean solution.

7. **Sign convention**:
   The shared FFT solver returns the positive-denominator solution, so the
   implementation multiplies the result by :math:`-1` to match the discrete
   Laplacian sign convention.

8. **Inverse FFT** yields :math:`\phi` in real space. If the input was 2D,
   the leading singleton dimension is removed.

.. note::

   **Boundary conditions and padding.**
   The FFT solve assumes the domain is *periodic*: the left/right and
   top/bottom edges are treated as connected. No zero-padding is applied
   before the FFT. If the wrapped phase is not periodic at the boundaries
   (which is the common case for experimental data), the solver may produce
   artifacts — ringing or slope errors — near the domain edges. To reduce
   their impact, crop the region of interest away from the edges, or use
   :func:`~xpunwrap.algorithms.algo_ls_weighted`, which can suppress
   discontinuous boundary regions via its border-weight mask.

.. _ls_poisson_refs:

References
----------
- D. C. Ghiglia and M. D. Pritt, "Two-Dimensional Phase Unwrapping: Theory,
  Algorithms, and Software," Wiley, 1998.
