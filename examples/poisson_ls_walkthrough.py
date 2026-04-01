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

import unwrap_phase_gpu as upg
from unwrap_phase_gpu.algorithms.ls_poisson_periodic_grad import (
    wrapped_gradients_stack,
    enforce_periodic_gradients_stack,
    divergence_stack,
    poisson_solve_fft_stack,
)

upg.set_ndarray_backend('numpy')
xp = upg.get_ndarray_backend()

# load/create the example data, perhaps Fig 9 for good example, and Fig 13 as
# difficult example
phase_wrapped = xp.ones(shape=(10, 512, 512), dtype=xp.float32)  #  todo:change

# the `algo_ls_poisson_periodic_grad` is broken up into the following steps:
#    initial dimensionality check
#    gx, gy = wrapped_gradients_stack(phase_wrapped)
#    gx, gy = enforce_periodic_gradients_stack(gx, gy)
#    rhs = divergence_stack(gx, gy)
#    phi = poisson_solve_fft_stack(rhs)
#    phase inversion

gx1, gy1 = wrapped_gradients_stack(phase_wrapped)
gx2, gy2 = enforce_periodic_gradients_stack(gx1, gy1)
rhs = divergence_stack(gx2, gy2)
phi = poisson_solve_fft_stack(rhs)
# invert in this case
phi *= -1

# plotting

nrows, ncols = 6, 2
fig = plt.figure(figsize=(6, 12))

ax1 = fig.add_subplot(nrows, ncols, 1, projection='3d')
ax1.imshow(phase_wrapped[1])

ax2 = fig.add_subplot(nrows, ncols, 2)
ax2.imshow(phase_wrapped[1])

ax3 = fig.add_subplot(nrows, ncols, 3)
ax3.imshow(gx1[1])

ax4 = fig.add_subplot(nrows, ncols, 4)
ax4.imshow(gy1[1])

ax5 = fig.add_subplot(nrows, ncols, 5)
ax5.imshow(gx2[1])

ax6 = fig.add_subplot(nrows, ncols, 6)
ax6.imshow(gy2[1])

for ax in fig.axes:
    ax.set_axis_off()

plt.tight_layout()
plt.show()