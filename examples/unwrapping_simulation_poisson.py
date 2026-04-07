"""
Generate only the LS-Poisson variants figure from unwrapping_simulation.
Outputs: unwrapping_comparison.png
"""

from unwrapping_simulation import main


if __name__ == "__main__":
    main(plot_itoh=False, plot_poisson=True, plot_2d_comparison=False)
