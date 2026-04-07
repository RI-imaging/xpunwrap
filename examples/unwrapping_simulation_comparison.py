"""
Generate only the 2D unwrapped comparison/difference maps vs skimage.
Outputs: unwrapping_2d_comparison.png
"""

from unwrapping_simulation import main


if __name__ == "__main__":
    main(plot_itoh=False, plot_poisson=False, plot_2d_comparison=True)
