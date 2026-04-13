"""
Step-by-step walkthrough of the Least-squares Poisson Periodic Gradient
algorithm.

The aim here is to visualise the steps of the algorithm with matplotlib
with example datasets, showing edge cases where phase unwrapping fails.
Where possible the scikit-image 2D phase unwrapping algorithm will be used
for comparison.

The example data is the data used by M. Gdeisat and F. Lilley in their
excellent "Two-Dimensional Phase Unwrapping Problem" guide.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple

import xpunwrap
from xpunwrap.algorithms.ls_poisson_pg import (
    wrapped_gradients_stack,
    enforce_periodic_gradients_stack,
    divergence_stack,
    poisson_solve_fft_stack,
)
from xpunwrap.algorithms._plane_utils import restore_mean_plane

xpunwrap.set_ndarray_backend('numpy')
xp = xpunwrap.get_ndarray_backend()

# load/create the example data, perhaps Fig 9 for good example, and Fig 13 as
# difficult example
def _generate_phase(nrx: int = 512, nry: int = 512)\
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create the continuous phase image f(x, y) = 20*exp(-0.25*(x^2+y^2)) + 2x + y."""
    tx = np.linspace(-3.0, 3.0, nrx)
    ty = np.linspace(-3.0, 3.0, nry)
    x, y = np.meshgrid(tx, ty)
    image = 20.0 * np.exp(-0.25 * (x**2 + y**2)) + 2.0 * x + y
    return x, y, image


def _wrap_phase(image: np.ndarray) -> np.ndarray:
    """Wrap phase to [-pi, pi] using angle of complex exponential."""
    return np.angle(np.exp(1j * image))


x, y, image = _generate_phase()
arr = _wrap_phase(image)
phase_wrapped = np.repeat(arr[None, ...], repeats=10, axis=0)

# the `algo_ls_poisson_pg` is broken up into the following steps:
#    initial dimensionality check
#    gx, gy = wrapped_gradients_stack(phase_wrapped)
#    gx, gy = enforce_periodic_gradients_stack(gx, gy)
#    rhs = divergence_stack(gx, gy)
#    phi = poisson_solve_fft_stack(rhs)
#    phase inversion

# Compute forward finite differences of the wrapped phase.
# Then wrap those gradients back into [-pi, pi) so they remain a valid
# representation of wrapped phase changes.
gx1, gy1 = wrapped_gradients_stack(phase_wrapped)
# Enforce periodic boundary conditions on the wrapped gradients so that the
# edges match up. This makes the subsequent FFT-based Poisson solve consistent
# with a periodic domain.
gx2, gy2 = enforce_periodic_gradients_stack(gx1, gy1)
# Take the divergence of those periodic wrapped gradients; this produces the
# right-hand side for the Poisson equation whose solution is the unwrapped phase.
rhs = divergence_stack(gx2, gy2)
# Solve the Poisson equation in the Fourier domain (batched over the stack);
# this yields an unwrapped phase up to a linear ramp.
phi = poisson_solve_fft_stack(rhs)
# Negate to match the sign convention used by the wrapped-gradient computation
# so that gradients of the solution align with the original wrapped gradients.
phi *= -1
phi_restored = restore_mean_plane(phi, phase_wrapped)

# plotting
fontsize = 12
nrows, ncols = 4, 2
fig = plt.figure(figsize=(8, 12))

ax1 = fig.add_subplot(nrows, ncols, 1)
ax1.set_title("Wrapped Phase Example", fontsize=fontsize)
im1 = ax1.imshow(phase_wrapped[1])
fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

ax2 = fig.add_subplot(nrows, ncols, 2, projection='3d')
ax2.set_title("Wrapped Phase Example 3D proj", fontsize=fontsize)
surf2 = ax2.plot_surface(x, y, phase_wrapped[1], cmap="viridis", linewidth=0, antialiased=True)
fig.colorbar(surf2, ax=ax2, fraction=0.046, pad=0.04, shrink=0.6)

ax3 = fig.add_subplot(nrows, ncols, 3)
ax3.set_title("Wrapped Gradients + Per. Grad. (x)", fontsize=fontsize)
im3 = ax3.imshow(gx2[1])
fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

ax4 = fig.add_subplot(nrows, ncols, 4)
ax4.set_title("Wrapped Gradients + Per. Grad. (y)", fontsize=fontsize)
im4 = ax4.imshow(gy2[1])
fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

ax5 = fig.add_subplot(nrows, ncols, 5)
ax5.set_title("Poisson RHS", fontsize=fontsize)
im5 = ax5.imshow(rhs[1])
fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

ax6 = fig.add_subplot(nrows, ncols, 6)
ax6.set_title("Unwrapped phase (with invertion)", fontsize=fontsize)
im6 = ax6.imshow(phi[1])
fig.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

ax7 = fig.add_subplot(nrows, ncols, 7)
ax7.set_title("Unwrapped phase (restore)", fontsize=fontsize)
im7 = ax7.imshow(phi_restored[1])
fig.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)

ax8 = fig.add_subplot(nrows, ncols, 8, projection='3d')
ax8.set_title("Unwrapped phase (restore) 3D proj", fontsize=fontsize)
surf8 = ax8.plot_surface(x, y, phi_restored[1], cmap="viridis", linewidth=0, antialiased=True)
fig.colorbar(surf8, ax=ax8, fraction=0.046, pad=0.04, shrink=0.6)

for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig("poisson_ls_walkthrough.png")
plt.show()
