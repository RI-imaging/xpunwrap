"""
Recreate the Figure 9(e–l) simulation from docs/research/twodimensionalphaseunwrapping.pdf.

Generates the synthetic continuous phase image, wraps it, and unwraps it via:
- Itoh rows→cols (first method)
- Itoh cols→rows (second method)
- Optional 2D unwrap using skimage.restoration.unwrap_phase (stand‑in for SRNCP)
- LS-Poisson (plain and periodic-gradient variants from unwrap_phase_gpu)

Run:
    python examples/unwrapping_simulation.py

Two figures are produced:
1) Itoh + optional skimage unwrap results.
2) LS-Poisson variants (same first row as the wrapped data).
"""

from __future__ import annotations

import math
import warnings
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import unwrap_phase_gpu as upg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d projection)


def _generate_phase(nrx: int = 512, nry: int = 512) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create the continuous phase image f(x, y) = 20*exp(-0.25*(x^2+y^2)) + 2x + y."""
    tx = np.linspace(-3.0, 3.0, nrx)
    ty = np.linspace(-3.0, 3.0, nry)
    x, y = np.meshgrid(tx, ty)
    image = 20.0 * np.exp(-0.25 * (x**2 + y**2)) + 2.0 * x + y
    return x, y, image


def _wrap_phase(image: np.ndarray) -> np.ndarray:
    """Wrap phase to [-pi, pi] using angle of complex exponential."""
    return np.angle(np.exp(1j * image))


def _unwrap_rows_then_cols(image_wrapped: np.ndarray) -> np.ndarray:
    """Itoh first method: unwrap rows, then columns."""
    out = np.unwrap(image_wrapped, axis=1)
    out = np.unwrap(out, axis=0)
    return out


def _unwrap_cols_then_rows(image_wrapped: np.ndarray) -> np.ndarray:
    """Itoh second method: unwrap columns, then rows."""
    out = np.unwrap(image_wrapped, axis=0)
    out = np.unwrap(out, axis=1)
    return out


def _unwrap_2d(image_wrapped: np.ndarray) -> np.ndarray | None:
    """Optional 2D unwrap using skimage (as a stand‑in for MATLAB SRNCP)."""
    try:
        algo = upg.algos_available()["algo_skimage_unwrap"]
    except Exception as exc:  # pragma: no cover - optional dependency
        warnings.warn(f"skimage not available, skipping 2D unwrap: {exc}")
        return None
    return algo(image_wrapped)


def _plot_intensity(
    fig: plt.Figure, ax: plt.Axes, x: np.ndarray, y: np.ndarray, image: np.ndarray, title: str
) -> None:
    im = ax.imshow(
        image,
        extent=(x.min(), x.max(), y.max(), y.min()),  # flip y to mirror MATLAB imagesc default
        origin="upper",
        cmap="viridis",
        aspect="equal",
    )
    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)


def _plot_surface(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    image: np.ndarray,
    title: str,
    elev: float = 70.0,
    clip: tuple[float, float] | None = None,
) -> None:
    # Match MATLAB surf: flip X, flip Z by rows so first row sits at the front.
    x_use = y[::-1, :]
    y_use = x
    z_use = image#[::-1, :]
    if clip is not None:
        z_use = np.clip(z_use, clip[0], clip[1])
    surf = ax.plot_surface(
        x_use,
        y_use,
        z_use,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        shade=True,
        axlim_clip=True,
    )
    ax.view_init(elev=elev, azim=-30)
    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.set_zlabel("Phase in radians")
    ax.set_title(title, fontsize=9)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    return surf


def _render_plot_grid(plots, x, y, figure_title: str, save_path: str | None = None) -> None:
    n_plots = len(plots)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)
    fig = plt.figure(figsize=(14, 4 * n_rows))
    fig.suptitle(figure_title, fontsize=16)
    gs = fig.add_gridspec(n_rows, n_cols, wspace=0.08, hspace=0.5)

    for idx, entry in enumerate(plots):
        title, kind, data, *rest = entry
        row, col = divmod(idx, n_cols)
        if kind == "surface":
            ax = fig.add_subplot(gs[row, col], projection="3d")
            surf = _plot_surface(
                ax,
                x,
                y,
                data,
                title,
                elev=rest[0] if rest else 30.0,
                clip=rest[1] if len(rest) > 1 else None,
            )
            # fig.colorbar(surf, ax=ax, fraction=0.04, pad=0.02, shrink=0.7)
        else:
            ax = fig.add_subplot(gs[row, col])
            _plot_intensity(fig, ax, x, y, data, title)

    # plt.tight_layout(h_pad=4)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def main(
    plot_itoh: bool = True,
    plot_poisson: bool = True,
    plot_2d_comparison: bool = True,
) -> None:
    # Ensure CPU numpy backend for the package algorithms
    upg.set_ndarray_backend("numpy")

    x, y, image = _generate_phase()

    image_wrapped = _wrap_phase(image)

    # Itoh and optional skimage unwraps
    image_unwrapped_1 = _unwrap_rows_then_cols(image_wrapped)
    image_unwrapped_2 = _unwrap_cols_then_rows(image_wrapped)
    image_unwrapped_2d = _unwrap_2d(image_wrapped)

    itoh_plots = [
        ("Wrapped image", "intensity", image_wrapped),
        ("Wrapped image plotted as surface", "surface", image_wrapped, 50.0, (-np.pi, np.pi)),
        ("Unwrapping Itoh algorithm: the first method", "intensity", image_unwrapped_1),
        ("Unwrapping Itoh unwrapper: the first method", "surface", image_unwrapped_1, 30.0),
        ("Unwrapping Itoh algorithm: the second method", "intensity", image_unwrapped_2),
        ("Unwrapping Itoh algorithm: the second method", "surface", image_unwrapped_2, 30.0),
    ]

    if image_unwrapped_2d is not None:
        itoh_plots.extend(
            [
                ("Unwrapping skimage", "intensity", image_unwrapped_2d),
                ("Unwrapping skimage (surface)", "surface", image_unwrapped_2d, 30.0),
            ]
        )
    else:
        warnings.warn("2D unwrap skipped; install scikit-image to reproduce MATLAB SRNCP plots.")

    if plot_itoh:
        _render_plot_grid(
            itoh_plots,
            x,
            y,
            "Itoh and skimage unwrapping",
            save_path="unwrapping_itoh.png",
        )

    # LS Poisson variants (first row reused: wrapped intensity + surface)
    restore_plane = True
    xp = upg.get_ndarray_backend()
    wrapped_stack = xp.asarray(image_wrapped[None, ...])
    ls_plain = upg.algo_ls_poisson(wrapped_stack, restore_plane)[0]
    ls_periodic = upg.algo_ls_poisson_periodic_grad(wrapped_stack, restore_plane)[0]
    ls_weighted = upg.algo_ls_weighted(wrapped_stack, restore_plane)[0]
    tvl1 = upg.algo_tvl1(wrapped_stack, restore_plane)[0]
    sk_unwrap = image_unwrapped_2d

    poisson_plots = [
        ("Wrapped image displayed as a visual intensity array", "intensity", image_wrapped),
        ("Wrapped image plotted as a surface", "surface", image_wrapped, 70.0, (-np.pi, np.pi)),
        ("Unwrapped phase map using LS-Poisson", "intensity", ls_plain),
        ("Unwrapped phase map using LS-Poisson (surface)", "surface", ls_plain, 30.0),
        ("Unwrapped phase map using LS-Poisson periodic-grad", "intensity", ls_periodic),
        ("Unwrapped phase map using LS-Poisson periodic-grad (surface)", "surface", ls_periodic, 30.0),
        ("Unwrapped phase map using LS-Weighted", "intensity", ls_weighted),
        ("Unwrapped phase map using LS-Weighted (surface)", "surface", ls_weighted, 30.0),
        ("Unwrapped phase map using TV-L1", "intensity", tvl1),
        ("Unwrapped phase map using TV-L1 (surface)", "surface", tvl1, 30.0),
    ]

    if image_unwrapped_2d is not None:
        poisson_plots.extend(
            [
                ("Unwrapped phase map using the 2D-SRNCP stand-in (skimage)", "intensity", image_unwrapped_2d),
                ("Unwrapped phase map using the 2D-SRNCP stand-in (surface)", "surface", image_unwrapped_2d, 30.0),
            ]
        )

    if plot_poisson:
        _render_plot_grid(
            poisson_plots,
            x,
            y,
            "Phase unwrapping comparison (Itoh, LS-Poisson variants, SRNCP stand-in)",
            save_path="unwrapping_comparison.png",
        )

    # 2D-only comparison and difference maps vs skimage
    if plot_2d_comparison and sk_unwrap is not None:
        diff_ls = ls_plain - sk_unwrap
        diff_ls_periodic = ls_periodic - sk_unwrap
        diff_ls_weighted = ls_weighted - sk_unwrap
        diff_tvl1 = tvl1 - sk_unwrap

        def rmse(a, b):
            return float(np.sqrt(np.mean((a - b) ** 2)))

        # Collect RMSEs for titles
        metrics = {
            "LS-Poisson": rmse(ls_plain, sk_unwrap),
            "LS-Poisson periodic-grad": rmse(ls_periodic, sk_unwrap),
            "LS-Weighted": rmse(ls_weighted, sk_unwrap),
            "TV-L1": rmse(tvl1, sk_unwrap),
        }

        # Use global vmin/vmax for unwrapped comparisons; and separate scale for diffs
        diff_stack = np.stack([diff_ls, diff_ls_periodic, diff_ls_weighted, diff_tvl1], axis=0)
        diff_abs = np.max(np.abs(diff_stack))

        algo_entries = [
            ("LS-Poisson", ls_plain, diff_ls, metrics["LS-Poisson"]),
            ("LS-Poisson periodic-grad", ls_periodic, diff_ls_periodic, metrics["LS-Poisson periodic-grad"]),
            ("LS-Weighted", ls_weighted, diff_ls_weighted, metrics["LS-Weighted"]),
            ("TV-L1", tvl1, diff_tvl1, metrics["TV-L1"]),
        ]
        algo_entries.sort(key=lambda t: t[3])  # sort by RMSE ascending

        two_d_plots = [
            ("Wrapped image", image_wrapped, None, None),
            ("Skimage unwrap", sk_unwrap, None, None),
        ]
        for name, img_algo, diff_img, rmse_val in algo_entries:
            two_d_plots.append((name, img_algo, None, None))
            two_d_plots.append((f"{name} - Skimage (RMSE={rmse_val:.3e})", diff_img, -diff_abs, diff_abs))

        n_plots = len(two_d_plots)
        n_cols = 2
        n_rows = math.ceil(n_plots / n_cols)
        fig = plt.figure(figsize=(14, 3 * n_rows), layout="constrained")
        gs = fig.add_gridspec(n_rows, n_cols, wspace=0.08, hspace=0.15)
        fig.suptitle("2D unwrapped comparisons and differences vs skimage", fontsize=11)

        for idx, (title, img, vmin, vmax) in enumerate(two_d_plots):
            r, c = divmod(idx, n_cols)
            ax = fig.add_subplot(gs[r, c])
            im = ax.imshow(
                img,
                extent=(x.min(), x.max(), y.max(), y.min()),
                origin="upper",
                cmap="viridis",
                aspect="equal",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("x axis")
            ax.set_ylabel("y axis")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

        fig.savefig("unwrapping_2d_comparison.png", dpi=150, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    main(plot_itoh=1,
         plot_poisson=1,
         plot_2d_comparison=1)
