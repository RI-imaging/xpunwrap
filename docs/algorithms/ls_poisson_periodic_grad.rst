Least-Squares with Periodic Gradients (ls_poisson_periodic_grad)
================================================================

This variant follows the same least-squares Poisson formulation as the
standard method, but explicitly enforces periodic boundary conditions on the
wrapped gradients before solving. That makes the formulation consistent with
periodic data or FFT-based solvers that assume wrap-around behavior at the
boundaries. The result is still a global least-squares solution, but one whose
gradients are periodic across the image edges. [1]

Derivation
----------

1. **Input**: wrapped phase :math:`\phi_w` in :math:`(-\pi,\pi]`.
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

5. **Periodic Poisson solve** in the Fourier domain:

   .. math::
      \hat{\phi}(k_x,k_y) =
      \frac{\hat{f}(k_x,k_y)}
           {(2-2\cos(2\pi k_x)) + (2-2\cos(2\pi k_y))}

   The DC term is set to zero to fix the global offset.

6. **Sign convention**:
   The implementation multiplies the result by :math:`-1` to match the
   internal gradient sign convention.

7. **Inverse FFT** returns the real-space unwrapped phase. If the input was
   2D, the leading singleton dimension is removed.

References
----------
.. [1] D. C. Ghiglia and M. D. Pritt, "Two-Dimensional Phase Unwrapping:
   Theory, Algorithms, and Software," Wiley, 1998.
