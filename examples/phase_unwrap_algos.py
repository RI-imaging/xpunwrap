"""Demonstrate each unwrapping algorithm."""

import matplotlib.pyplot as plt
import unwrap_phase_gpu as upg

plt.style.use('dark_background')
upg.set_ndarray_backend("cupy")
xp = upg.get_ndarray_backend()

# some fake wrapped phase data
rng = xp.random.default_rng(7)
phase_stack = rng.uniform(
    -xp.pi, xp.pi, size=(5, 64, 64)).astype(xp.float32)

algos_available = upg.algos_available()
outputs = {}
for algo_name, algo in algos_available.items():
    outputs[algo_name] = algo(phase_stack)

fig, axes = plt.subplots(2, 3, figsize=(8, 6))
fig.suptitle("Phase Unwrapping on the GPU with CuPy", fontsize=18)
axes = axes.flatten(order='F')

axes[0].imshow(phase_stack[0].get())
axes[0].set_title("Wrapped Phase")
for i, (algo_name, arr) in enumerate(outputs.items(), start=2):
    arr_cpu = arr.get()[0]
    axes[i].imshow(arr_cpu)
    axes[i].set_title(f"Unwrapped\n'{algo_name}'")

for ax in axes:
    ax.set_axis_off()

plt.tight_layout(w_pad=4.5)
# plt.show()
plt.savefig("phase_unwrap_algos.png")
