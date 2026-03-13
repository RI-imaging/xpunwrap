"""
Field retrieval (qpretrieve) and
phase unwrapping (unwrap_phase_gpu) on GPU.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import qpretrieve
import unwrap_phase_gpu as upg

# Force GPU backend for both libraries.
upg.set_ndarray_backend("cupy")
qpretrieve.set_ndarray_backend("cupy")
xp = upg.get_ndarray_backend()

edata = np.load("./data/hologram_cell.npz")
holo = qpretrieve.OffAxisHologram(data=edata["data"])
bg = qpretrieve.OffAxisHologram(data=edata["bg_data"])

holo.run_pipeline(filter_name="disk", filter_size=1 / 2, scale_to_filter=True)
bg.process_like(holo)
phase_wrapped = xp.asarray(holo.phase - bg.phase).astype(xp.float32)

# Unwrap the phase with all available algorithms
outputs = {}
for algo_name, algo in upg.algos_available().items():
    outputs[algo_name] = algo(phase_wrapped)

# plot the wrapped and unwrapped phases
plt.style.use("dark_background")
fig, axes = plt.subplots(2, 3, figsize=(8, 6))
fig.suptitle("Field Retrieval + Phase Unwrapping (GPU)", fontsize=18)
axes = axes.flatten(order="F")

axes[0].imshow(phase_wrapped.get()[0])
axes[0].set_title("Wrapped Phase")

for i, (algo_name, arr) in enumerate(outputs.items(), start=2):
    ax = axes[i]
    ax.imshow(arr.get()[0])
    ax.set_title(f"Unwrapped\n{algo_name}")

for ax in axes:
    ax.set_axis_off()

plt.tight_layout(w_pad=4.5)
plt.savefig("gpu_field_retr_phase_unwrapping.png")
# plt.show()
