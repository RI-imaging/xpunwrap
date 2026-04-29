"""
Field retrieval (qpretrieve) and phase unwrapping (xpunwrap) on GPU.
"""

import matplotlib.pyplot as plt
import numpy as np
import qpretrieve
import xpunwrap

# Force GPU backend for both libraries.
qpretrieve.set_ndarray_backend("cupy")
xpunwrap.set_ndarray_backend("cupy")
xp = xpunwrap.get_ndarray_backend()

fft_interface = qpretrieve.fourier.FFTFilterCupy
edata = np.load("./data/hologram_cell.npz")
holo = qpretrieve.OffAxisHologram(edata["data"], fft_interface)
bg = qpretrieve.OffAxisHologram(edata["bg_data"], fft_interface)

holo.run_pipeline(filter_name="disk", filter_size=1 / 2, scale_to_filter=True)
bg.process_like(holo)
phase_wrapped = xp.asarray(holo.phase - bg.phase).astype(xp.float32)

# Unwrap the phase with all available algorithms
outputs = {}
for algo_name, algo in xpunwrap.algos_available().items():
    outputs[algo_name] = algo(phase_wrapped)
skimage_out = outputs.get("algo_skimage_unwrap", None)
if skimage_out is not None:
    outputs_no_skimage = {k: v for k, v in outputs.items() if k != "algo_skimage_unwrap"}
else:
    outputs_no_skimage = outputs

# plot the wrapped and unwrapped phases
plt.style.use("dark_background")
fig, axes = plt.subplots(2, 3, figsize=(8, 6))
fig.suptitle("Field Retrieval + Phase Unwrapping (GPU)", fontsize=18)
axes = axes.flatten(order="F")

axes[0].imshow(phase_wrapped.get()[0])
axes[0].set_title("Wrapped Phase")

# With column-major flattening, bottom-right is index 5.
skimage_ax = axes[5]

# Fill remaining non-skimage algorithms first (slots 1..4).
plot_slots = [2, 3, 4]
for slot, (algo_name, arr) in zip(plot_slots, outputs_no_skimage.items()):
    ax = axes[slot]
    ax.imshow(arr.get()[0])
    ax.set_title(f"Unwrapped\n{algo_name}")

if skimage_out is not None:
    skimage_ax.imshow(skimage_out.get()[0])
    skimage_ax.set_title("Unwrapped (CPU-only)\nalgo_skimage_unwrap")
else:
    skimage_ax.text(0.5, 0.5, "skimage missing", ha="center", va="center")
    skimage_ax.set_title("Unwrapped\nalgo_skimage_unwrap")

for ax in axes:
    ax.set_axis_off()

plt.tight_layout(w_pad=4.5)
plt.savefig("gpu_field_retr_phase_unwrapping.png")
# plt.show()
