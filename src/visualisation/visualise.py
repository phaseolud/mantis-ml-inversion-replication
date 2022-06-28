import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from src.data.utils import load_inversion_grid, create_grid_transformation_matrices


def plot_poloidal(poloidal_crosssection: np.array, ax=None, **kwargs):
    ax = ax or plt.gca()
    inversion_grid = load_inversion_grid()
    tri_grid = mpl.tri.Triangulation(
        inversion_grid["tri_x"][0][0][0],
        inversion_grid["tri_y"][0][0][0],
        inversion_grid["tri_nodes"][0][0],
    )

    # when we input a square inversion grid instead of a triangular one
    input_shape = poloidal_crosssection.shape
    if len(input_shape) == 4:
        assert input_shape[0] == 1  # batch shape 1
        _, square_to_tri_grid = create_grid_transformation_matrices((input_shape[1], input_shape[2]))
        poloidal_crosssection = square_to_tri_grid @ np.reshape(poloidal_crosssection, -1)

    tpc = ax.tripcolor(tri_grid, poloidal_crosssection, **kwargs)
    ax.axis("equal")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.patch.set_alpha(0.0)
    ax.set_xlim([0.62, 1.13])
    ax.set_ylim([-0.8, 0.2])

    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    return tpc
