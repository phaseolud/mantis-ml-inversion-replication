import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from src.data.grid import load_tri_grid


def plot_poloidal_cs(poloidal_crosssection, ax=None, geometry_id: str = "65903_cam_10", **kwargs):
    ax = ax or plt.gca()
    inversion_grid = load_tri_grid(geometry_id)
    tri_grid = mpl.tri.Triangulation(
        inversion_grid['grid_verts'][:, 0],
        inversion_grid['grid_verts'][:, 1],
        inversion_grid["grid_cells"],
    )

    tpc = ax.tripcolor(tri_grid, poloidal_crosssection, rasterized=True, **kwargs)
    ax.axis("equal")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.patch.set_alpha(0.0)
    ax.set_xlim([0.62, 1.13])
    ax.set_ylim([-0.8, 0.2])
    return tpc
