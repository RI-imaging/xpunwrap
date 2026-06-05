Least-Squares with Periodic Gradients (algo_ls_poisson_pg)
=====================================================================

This variant uses the same least-squares Poisson solve as the standard method,
but it forces the wrapped gradients to be periodic before the FFT step. See
Ghiglia and Pritt (1998).

API: :func:`xpunwrap.algo_ls_poisson_pg<xpunwrap.algorithms.algo_ls_poisson_pg>`

Derivation
----------

1. **Input**: wrapped phase :math:`\phi_w` in :math:`[-\pi,\pi)`.
2. **Wrapped forward gradients**:

   .. math::
      g_x = \mathrm{wrap}(\phi_w(x+1,y) - \phi_w(x,y)), \quad
      g_y = \mathrm{wrap}(\phi_w(x,y+1) - \phi_w(x,y))

3. **Periodic enforcement**:
   The last column of :math:`g_x` is set to the first column, and the last
   row of :math:`g_y` is set to the first row, enforcing periodicity.

4. **Divergence**:

   .. math::
      f = \nabla \cdot g

   where the discrete periodic divergence used by the FFT solver is:

   .. math::
      \nabla \cdot g =
      \left(g_x(x,y) - g_x((x-1)\bmod W,y)\right) +
      \left(g_y(x,y) - g_y(x,(y-1)\bmod H)\right)

5. **Least-squares / Poisson relation**:

   The least-squares objective solved here is:

   .. math::
      \phi^\star = \arg\min_{\phi}\sum_{x,y}\lvert \nabla \phi - g \rvert^2

   with normal equation:

   .. math::
      \nabla^2 \phi = \nabla\cdot g = f

   (This is a PDE, not an ODE.)

6. **Periodic Poisson solve** in the Fourier domain:

   .. math::
      \hat{\phi}(k_x,k_y) =
      \frac{\hat{f}(k_x,k_y)}
           {(2-2\cos(2\pi k_x)) + (2-2\cos(2\pi k_y))}

   Here :math:`k_x, k_y` are normalized frequencies from
   :math:`\mathrm{fftfreq}` (cycles/sample).

   The DC term is set to zero to fix the global offset.

7. **Sign convention**:
   The implementation multiplies the result by :math:`-1` to match the
   internal gradient sign convention.

8. **Inverse FFT** returns the real-space unwrapped phase. If the input was
   2D, the leading singleton dimension is removed.

.. _ls_poisson_pg_refs:

References
----------
- D. C. Ghiglia and M. D. Pritt, "Two-Dimensional Phase Unwrapping: Theory,
  Algorithms, and Software," Wiley, 1998.
