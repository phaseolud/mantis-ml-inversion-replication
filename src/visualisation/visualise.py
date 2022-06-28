import matplotlib as mpl
import matplotlib.pyplot as plt

from src.data.utils import load_inversion_grid


def plot_poloidal(poloidal_crosssection, ax=None, **kwargs):
    ax = ax or plt.gca()
    inversion_grid = load_inversion_grid()
    tri_grid = mpl.tri.Triangulation(
        inversion_grid["tri_x"][0][0][0],
        inversion_grid["tri_y"][0][0][0],
        inversion_grid["tri_nodes"][0][0],
    )

    tpc = ax.tripcolor(tri_grid, poloidal_crosssection, **kwargs)
    ax.axis("equal")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.patch.set_alpha(0.0)
    ax.set_xlim([0.62, 1.13])
    ax.set_ylim([-0.8, 0.2])
    return tpc
