from functools import lru_cache
from typing import Tuple

from scipy.sparse import csc_matrix, load_npz

import definitions
from scipy.io import loadmat
import numpy as np

from src.utils import disk_memory


@lru_cache
def load_inversion_grid():
    inversion_grid_file = definitions.DATA_DIR / "utils" / "inversion_grid.mat"
    inversion_grid = loadmat(str(inversion_grid_file))["inversion_grid"]
    return inversion_grid


@lru_cache
def load_geometry_matrix(shot_no: int):
    filename = f"{shot_no}.npz"
    geometry_matrix_file = (
        definitions.DATA_DIR / "utils" / "geometry_matrices" / filename
    )
    if not geometry_matrix_file.exists():
        raise IOError(f"Geometry matrix file not found for {filename=}")
    geometry_matrix = load_npz(geometry_matrix_file).astype(np.float32)
    return geometry_matrix


@disk_memory.cache
def create_grid_transformation_matrices(output_grid_shape: Tuple[int, int]):
    """
    Calculates the transformation matrices: square grid <--> triangular inversion grid by minimizing the
     distance between the individual points in both grids

    :returns tri_to_square_grid_mat, square_to_inv_grid_mat
    """
    grid_height, grid_width = output_grid_shape
    x_grid = np.arange(0, grid_width)
    y_grid = np.arange(0, grid_height)

    inversion_grid = load_inversion_grid()

    R_min = inversion_grid["R_mean"][0][0][0].min()
    R_max = inversion_grid["R_mean"][0][0][0].max()
    R_grid = R_min + x_grid / grid_width * (R_max - R_min)

    Z_min = inversion_grid["Z_mean"][0][0][0].min()
    Z_max = inversion_grid["Z_mean"][0][0][0].max()
    Z_grid = np.flip(Z_min + y_grid / grid_height * (Z_max - Z_min))

    inv_grid_Z_mean = inversion_grid["Z_mean"][0][0][0]
    inv_grid_R_mean = inversion_grid["R_mean"][0][0][0]

    sparse_idx_tri_to_square = np.zeros((grid_height * grid_width, 2), dtype=int)
    n_inv_elements = len(inv_grid_Z_mean)

    for h in range(grid_height):
        for w in range(grid_width):
            dist = np.sqrt(
                (R_grid[w] - inv_grid_R_mean) ** 2 + (Z_grid[h] - inv_grid_Z_mean) ** 2
            )
            min_dist_idx = np.argmin(dist)
            sparse_idx_tri_to_square[w + grid_width * h] = [w + grid_width * h, min_dist_idx]
    tri_to_square_grid_mat = csc_matrix(
        (np.ones((grid_height * grid_width,)), (sparse_idx_tri_to_square[:, 0], sparse_idx_tri_to_square[:, 1])),
        shape=(grid_width * grid_height, n_inv_elements),
        dtype=int,
    )

    sparse_idx_square_to_tri = np.zeros((n_inv_elements, 2), dtype=int)

    for i_element in range(n_inv_elements):
        dist = (
                np.square(inv_grid_Z_mean[i_element] - Z_grid)[:, None]
                + np.square(inv_grid_R_mean[i_element] - R_grid)[None, :]
        )
        min_dist_idx = np.argmin(dist, axis=None)
        sparse_idx_square_to_tri[i_element] = [i_element, min_dist_idx]

    square_to_tri_grid_mat = csc_matrix(
        (np.ones((n_inv_elements,)), (sparse_idx_square_to_tri[:, 0], sparse_idx_square_to_tri[:, 1])),
        shape=(n_inv_elements, grid_width * grid_height),
        dtype=int,
    )

    return tri_to_square_grid_mat, square_to_tri_grid_mat

