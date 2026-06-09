"""Itoh and Skimage unwrapping

Generate only the Itoh/skimage comparison figure from unwrapping_simulation.
This graph is a reproduction of the graph in “Two-Dimensional Phase Unwrapping
Problem By Dr. Munther Gdeisat and Dr. Francis Lilley”.

Outputs: unwrapping_itoh.png
"""

from unwrapping_simulation import main


if __name__ == "__main__":
    main(plot_itoh=True, plot_poisson=False, plot_2d_comparison=False)
