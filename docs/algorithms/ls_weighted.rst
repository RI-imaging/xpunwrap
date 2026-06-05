Weighted Least-Squares (algo_ls_weighted)
=========================================

This is a simplified weighted least-squares unwrap. It masks large wrapped
gradients, then solves a weighted Poisson equation with Jacobi iterations.
The code here uses binary weights and periodic neighbour rolls, so it matches
the implementation in this repository rather than a full robust solver from
the literature. See Ghiglia and Romero (1994) and Ghiglia and Pritt (1998).

API: :func:`xpunwrap.algo_ls_weighted<xpunwrap.algorithms.algo_ls_weighted>`

Derivation
----------

1. **Input**: wrapped phase :math:`\phi_w` in :math:`[-\pi,\pi)`.
2. **Wrapped gradients**:

   .. math::
      g_x = \mathrm{wrap}(\phi_w(x+1,y) - \phi_w(x,y)), \quad
      g_y = \mathrm{wrap}(\phi_w(x,y+1) - \phi_w(x,y))

   In the code, forward differences are used and the last row/column is
   repeated before wrapping.

   The weighted divergence and Jacobi updates both use periodic neighbour
   access.

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
   The solver updates each pixel from its neighbours and local weights:

   .. math::
      \phi^{k+1} =
      \frac{
        w_{x+}\phi_{x+} + w_{x-}\phi_{x-} +
        w_{y+}\phi_{y+} + w_{y-}\phi_{y-} - f
      }{w_{x+} + w_{x-} + w_{y+} + w_{y-}}

   The :math:`-f` term above reflects the implementation's discrete sign
   convention.

   The update uses periodic neighbour rolls in code.

9. **Output**:
   After a fixed number of iterations, :math:`\phi` is returned. If the input
   was 2D, the leading singleton dimension is removed.

.. _ls_weighted_refs:

References
----------
- D. C. Ghiglia and L. A. Romero, "Robust two-dimensional weighted and
  unweighted phase unwrapping that uses fast transforms and iterative methods,"
  J. Opt. Soc. Am. A, vol. 11, no. 1, pp. 107-117, 1994.
- D. C. Ghiglia and M. D. Pritt, "Two-Dimensional Phase Unwrapping: Theory,
  Algorithms, and Software," Wiley, 1998.
