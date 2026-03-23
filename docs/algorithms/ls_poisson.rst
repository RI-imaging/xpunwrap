Least-Squares Poisson (algo_ls_poisson)
=======================================

This algorithm turns phase unwrapping into a least-squares problem: it looks
for the smooth phase field whose gradients best match the wrapped gradients
measured from the data. In practice, it computes wrapped forward differences,
forms their divergence, and solves a Poisson equation to recover a globally
consistent phase (up to an arbitrary constant offset). This is the classic
FFT-based least-squares approach and is a good default when the wrapped phase
is reasonably clean and you want a fast, stable solution. [1]

API: :func:`unwrap_phase_gpu.algorithms.algo_ls_poisson`

Derivation
----------

1. **Input**: a wrapped phase :math:`\phi_w` with values in :math:`[-\pi,\pi)`.
2. **Forward differences** (wrapped gradients):

   .. math::
      g_x = \phi_w(x+1,y) - \phi_w(x,y), \quad
      g_y = \phi_w(x,y+1) - \phi_w(x,y)

3. **Wrap the gradients** into :math:`[-\pi,\pi)`:

   .. math::
      \mathrm{wrap}(u) = (u + \pi) \bmod (2\pi) - \pi

4. **Divergence** of wrapped gradients:

   .. math::
      \nabla \cdot g = \partial_x g_x + \partial_y g_y

5. **Poisson equation** for the unwrapped phase :math:`\phi`:

   .. math::
      \nabla^2 \phi = \nabla \cdot g

6. **FFT solve** in the frequency domain:

   .. math::
      \hat{\phi}(k_x,k_y) =
      \frac{\hat{\nabla \cdot g}(k_x,k_y)}
           {(2\pi i k_x)^2 + (2\pi i k_y)^2}

   The DC component is set to zero to enforce a zero-mean solution.

7. **Inverse FFT** yields :math:`\phi` in real space. If the input was 2D,
   the leading singleton dimension is removed.

.. _ls_poisson_refs:

References
----------
.. [1] D. C. Ghiglia and M. D. Pritt, "Two-Dimensional Phase Unwrapping:
   Theory, Algorithms, and Software," Wiley, 1998.
