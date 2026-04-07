"""
Generate only the Itoh/skimage comparison figure from unwrapping_simulation.
Outputs: unwrapping_itoh.png
"""

from unwrapping_simulation import main


if __name__ == "__main__":
    main(plot_itoh=True, plot_poisson=False, plot_2d_comparison=False)
