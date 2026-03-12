Least-Squares Poisson (ls_poisson)
==================================

This algorithm unwraps a 2D phase image (or a stack of images) by solving a
Poisson equation derived from wrapped gradients.

Mathematics (line by line)
--------------------------

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
