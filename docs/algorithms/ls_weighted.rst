Weighted Least-Squares (algo_ls_weighted)
=========================================

This method extends the least-squares Poisson approach by down-weighting
locations that are likely to contain phase discontinuities. It detects large
wrapped gradients, treats them as unreliable, and assigns smaller weights so
they contribute less to the solution. The resulting weighted Poisson solve
produces a smoother, more robust unwrapped phase in the presence of sharp
jumps or noisy regions, at the cost of a slower iterative solve. [1] [2]

API: :func:`xpunwrap.algo_ls_weighted<xpunwrap.algorithms.algo_ls_weighted>`

Derivation
----------

1. **Input**: wrapped phase :math:`\phi_w` in :math:`[-\pi,\pi)`.
2. **Wrapped gradients**:

   .. math::
      g_x = \mathrm{wrap}(\phi_w(x+1,y) - \phi_w(x,y)), \quad
      g_y = \mathrm{wrap}(\phi_w(x,y+1) - \phi_w(x,y))

   In the current implementation, forward differences are used and the last
   row/column uses edge replication before wrapping.

3. **Gradient magnitude**:

   .. math::
      m = \sqrt{g_x^2 + g_y^2}

4. **Weights**:
   Set :math:`w = 0` where :math:`m` exceeds a threshold, otherwise :math:`w=1`.

5. **Weighted least-squares objective**:

   .. math::
      \phi^\star = \arg\min_{\phi}\sum_{x,y} w\,\lvert \nabla \phi - g \rvert^2

6. **Weighted divergence**:

   .. math::
      f = \nabla \cdot (w \, g)

7. **Weighted Poisson equation (normal equation)**:

   .. math::
      \nabla \cdot (w \nabla \phi) = f

   (This is a PDE, not an ODE.)

8. **Jacobi iterations**:
   The solver updates each pixel using its neighbors and the local weights:

   .. math::
      \phi^{k+1} =
      \frac{
        w_{x+}\phi_{x+} + w_{x-}\phi_{x-} +
        w_{y+}\phi_{y+} + w_{y-}\phi_{y-} - f
      }{w_{x+} + w_{x-} + w_{y+} + w_{y-}}

   The :math:`-f` term above reflects the implementation's discrete sign
   convention.

9. **Output**:
   After a fixed number of iterations, :math:`\phi` is returned. If the input
   was 2D, the leading singleton dimension is removed.

.. _ls_weighted_refs:

References
----------
.. [1] D. C. Ghiglia and L. A. Romero, "Robust two-dimensional weighted and
   unweighted phase unwrapping that uses fast transforms and iterative
   methods," J. Opt. Soc. Am. A, vol. 11, no. 1, pp. 107-117, 1994.
.. [2] D. C. Ghiglia and M. D. Pritt, "Two-Dimensional Phase Unwrapping:
   Theory, Algorithms, and Software," Wiley, 1998.
