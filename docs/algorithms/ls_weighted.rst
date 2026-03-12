Weighted Least-Squares (ls_weighted)
====================================

This method solves a weighted Poisson equation to reduce the influence of
phase discontinuities detected by large gradient magnitudes.

Mathematics (line by line)
--------------------------

1. **Input**: wrapped phase :math:`\phi_w` in :math:`(-\pi,\pi]`.
2. **Wrapped gradients**:

   .. math::
      g_x = \mathrm{wrap}(\phi_w(x+1,y) - \phi_w(x,y)), \quad
      g_y = \mathrm{wrap}(\phi_w(x,y+1) - \phi_w(x,y))

3. **Gradient magnitude**:

   .. math::
      m = \sqrt{g_x^2 + g_y^2}

4. **Weights**:
   Set :math:`w = 0` where :math:`m` exceeds a threshold, otherwise :math:`w=1`.

5. **Weighted divergence**:

   .. math::
      f = \nabla \cdot (w \, g)

6. **Weighted Poisson equation**:

   .. math::
      \nabla \cdot (w \nabla \phi) = f

7. **Jacobi iterations**:
   The solver updates each pixel using its neighbors and the local weights:

   .. math::
      \phi^{k+1} =
      \frac{
        w_{x+}\phi_{x+} + w_{x-}\phi_{x-} +
        w_{y+}\phi_{y+} + w_{y-}\phi_{y-} - f
      }{w_{x+} + w_{x-} + w_{y+} + w_{y-}}

8. **Output**:
   After a fixed number of iterations, :math:`\phi` is returned. If the input
   was 2D, the leading singleton dimension is removed.
