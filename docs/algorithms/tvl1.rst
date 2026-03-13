TV-L1 Primal-Dual (tvl1)
========================

This algorithm formulates unwrapping as an optimization problem that balances
data fidelity (matching wrapped gradients) with a total-variation prior that
promotes piecewise-smooth phase. It solves the resulting TV-L1 objective using
a fast first-order primal-dual method, applied independently to each slice in
a stack. The approach is robust to noise and preserves sharp features better
than simple least-squares methods. [1]

Derivation
----------

1. **Input**: wrapped phase :math:`\phi_w` in :math:`(-\pi,\pi]`.
2. **Wrapped gradients** of the data term:

   .. math::
      g_x^w = \mathrm{wrap}(\phi_w(x+1,y) - \phi_w(x,y)), \quad
      g_y^w = \mathrm{wrap}(\phi_w(x,y+1) - \phi_w(x,y))

3. **Initialize** the primal variable :math:`\phi` and dual variables
   :math:`p_x, p_y` to zeros. Set :math:`\bar{\phi} = \phi`.

4. **Dual update** (gradient step):

   .. math::
      p_x \leftarrow p_x + \sigma(\partial_x \bar{\phi} - g_x^w), \quad
      p_y \leftarrow p_y + \sigma(\partial_y \bar{\phi} - g_y^w)

5. **Projection** onto the unit ball (L1 proximal):

   .. math::
      (p_x, p_y) \leftarrow \frac{(p_x, p_y)}{\max(1, \sqrt{p_x^2+p_y^2})}

6. **Primal update** (divergence step):

   .. math::
      \phi \leftarrow \phi + \tau \, \nabla \cdot (p_x, p_y)

7. **Extrapolation**:

   .. math::
      \bar{\phi} \leftarrow 2\phi - \phi_{\text{old}}

8. **Repeat** for a fixed number of iterations. For 3D inputs, each slice is
   processed independently and reassembled into a stack.

References
----------
.. [1] A. Chambolle and T. Pock, "A first-order primal-dual algorithm for
   convex problems with applications to imaging," J. Math. Imaging Vis.,
   vol. 40, no. 1, pp. 120-145, 2011.
