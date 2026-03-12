"""
Field retrieval (qpretrieve) and
phase unwrapping (unwrap_phase_gpu) on GPU.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import unwrap_phase_gpu as upg

# --- User settings (edit these in PyCharm) ---
DATA_PATH = Path(r"PATH\TO\off_axis_hologram.npy")
BG_PATH = None  # e.g. Path(r"PATH\TO\background.npy")
DATA_KEY = "data"  # used only for .npz
BG_KEY = "bg_data"  # used only for .npz

FILTER_NAME = "smooth disk"
FILTER_SIZE = 0.5
SCALE_TO_FILTER = False
SAVE_PATH = None  # e.g. "field_retr_unwrap.png"


def _load_hologram(path: Path, key: str | None) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        data = np.load(path)
        if key is None:
            raise ValueError("NPZ input requires a key.")
        return data[key]
    raise ValueError("Unsupported input format. Use .npy or .npz.")


def _to_numpy(arr):
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


# Force GPU backend for both libraries.
upg.set_ndarray_backend("cupy")
qpretrieve.set_ndarray_backend("cupy")
xp = upg.get_ndarray_backend()


# --- Load hologram(s) ---
if not DATA_PATH.exists():
    raise SystemExit(f"DATA_PATH does not exist: {DATA_PATH}")

holo_data = _load_hologram(DATA_PATH, DATA_KEY)
bg_data = _load_hologram(BG_PATH, BG_KEY) if BG_PATH else None

holo = qpretrieve.OffAxisHologram(data=xp.asarray(holo_data))
holo.run_pipeline(
    filter_name=FILTER_NAME,
    filter_size=FILTER_SIZE,
    scale_to_filter=SCALE_TO_FILTER,
)

if bg_data is not None:
    bg = qpretrieve.OffAxisHologram(data=xp.asarray(bg_data))
    bg.process_like(holo)
    phase_wrapped = (
        holo.get_data_with_input_layout(holo.phase)
        - bg.get_data_with_input_layout(bg.phase)
    )
else:
    phase_wrapped = holo.get_data_with_input_layout(holo.phase)

phase_wrapped = xp.asarray(phase_wrapped).astype(xp.float32)


# --- Unwrap with all algorithms ---
outputs = {}
for algo_name, algo in upg.algos_available().items():
    outputs[algo_name] = algo(phase_wrapped)


# --- Plot ---
plt.style.use("dark_background")
n_plots = 1 + len(outputs)
n_cols = 3
n_rows = (n_plots + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 3 * n_rows))
fig.suptitle("Field Retrieval + Phase Unwrapping (GPU)", fontsize=16)
axes = np.atleast_1d(axes).ravel(order="F")

axes[0].imshow(_to_numpy(phase_wrapped))
axes[0].set_title("Wrapped Phase")
axes[0].set_axis_off()

for i, (algo_name, arr) in enumerate(outputs.items(), start=1):
    ax = axes[i]
    ax.imshow(_to_numpy(arr))
    ax.set_title(f"Unwrapped\n{algo_name}")
    ax.set_axis_off()

for ax in axes[n_plots:]:
    ax.set_axis_off()

plt.tight_layout(w_pad=4.5)
if SAVE_PATH:
    plt.savefig(SAVE_PATH, dpi=150)
else:
    plt.show()
