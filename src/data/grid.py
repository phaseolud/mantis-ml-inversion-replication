from typing import Tuple

from matplotlib import tri as mpl_tri
import numpy as np
from scipy.sparse import csc_matrix

import definitions


def create_grid_transformation_matrices(geometry_id: str, rect_grid_shape: Tuple[int, int], linear: bool = False):
    """
    Created the grid transformation matrices from and to the triangular grid and a rectangular grid
    returns: (rect_to_tri, tri_to_rect)
    """
    grid_transformations = GridTransformations(geometry_id, rect_grid_shape)
    if linear:
        return grid_transformations.rect_to_tri_grid_linear(), grid_transformations.tri_to_rect_grid_linear()
    else:
        return grid_transformations.rect_to_tri_grid_nearest(), grid_transformations.tri_to_rect_grid_nearest()


def load_tri_grid(geometry_id: str = "65903_cam_10"):
    inversion_grid_file = np.load(definitions.DATA_DIR / "utils" / "geometry_matrices" / f"{geometry_id}.npz")
    mean_grid_values = inversion_grid_file['grid_verts'][inversion_grid_file['grid_cells']].mean(axis=1).T
    inversion_grid = {
        'R_mean': mean_grid_values[0],
        'Z_mean': mean_grid_values[1],
        'grid_verts': inversion_grid_file['grid_verts'],
        'grid_cells': inversion_grid_file['grid_cells']
    }

    return inversion_grid


class GridTransformations:
    def __init__(self, geometry_id: str, rect_grid_shape: Tuple[int, int]):
        self.rect_height, self.rect_width = rect_grid_shape

        x_grid = np.arange(0, self.rect_width)
        y_grid = np.arange(0, self.rect_height)

        inversion_grid = load_tri_grid(geometry_id)

        self.Z_mean_values_tri_grid = inversion_grid['Z_mean']
        self.R_mean_values_tri_grid = inversion_grid['R_mean']

        R_min = self.R_mean_values_tri_grid.min()
        R_max = self.R_mean_values_tri_grid.max()
        R_step_size = (R_max - R_min) / self.rect_width
        self.R_grid_rect = R_min + x_grid * R_step_size

        Z_min = self.Z_mean_values_tri_grid.min()
        Z_max = self.Z_mean_values_tri_grid.max()
        Z_step_size = (Z_max - Z_min) / self.rect_height
        self.Z_grid_rect = np.flip((Z_min + y_grid * Z_step_size))

        self.R_mesh, self.Z_mesh = np.meshgrid(self.R_grid_rect, self.Z_grid_rect)

        self.N_tri = len(self.Z_mean_values_tri_grid)
        self.N_rect = self.rect_height * self.rect_width

        self.anchor_R_grid_rect = self.R_grid_rect[:-1] + R_step_size / 2
        self.anchor_Z_grid_rect = self.Z_grid_rect[:-1] - Z_step_size / 2

    def tri_to_rect_grid_nearest(self):
        sparse_idx_tri_to_rect = np.zeros((self.rect_height * self.rect_width, 2), dtype=int)
        should_contain = np.ones((self.rect_height * self.rect_width))
        for ih in range(self.rect_height):
            for iw in range(self.rect_width):
                dist = np.sqrt(
                    (self.R_grid_rect[iw] - self.R_mean_values_tri_grid) ** 2 +
                    (self.Z_grid_rect[ih] - self.Z_mean_values_tri_grid) ** 2
                )
                min_dist_idx = np.argmin(dist)
                sparse_idx_tri_to_rect[iw + self.rect_width * ih] = [iw + self.rect_width * ih, min_dist_idx]
                if dist[min_dist_idx] > 0.01:
                    should_contain[iw + self.rect_width * ih] = 0

        return csc_matrix(
            (should_contain,
             (sparse_idx_tri_to_rect[:, 0], sparse_idx_tri_to_rect[:, 1])),
            shape=(self.N_rect, self.N_tri),
            dtype=int,
        )

    def rect_to_tri_grid_nearest(self):
        sparse_idx_rect_to_tri = np.zeros((self.N_tri, 2), dtype=int)

        for i_element in range(self.N_tri):
            dist = (
                    np.square(self.Z_mean_values_tri_grid[i_element] - self.Z_grid_rect)[:, None]
                    + np.square(self.R_mean_values_tri_grid[i_element] - self.R_grid_rect)[None, :]
            )
            min_dist_idx = np.argmin(dist)
            sparse_idx_rect_to_tri[i_element] = [i_element, min_dist_idx]

        return csc_matrix(
            (np.ones((self.N_tri,)),
             (sparse_idx_rect_to_tri[:, 0], sparse_idx_rect_to_tri[:, 1])),
            shape=(self.N_tri, self.N_rect),
            dtype=int,
        )

    def tri_to_rect_grid_linear(self):
        M = np.zeros((self.N_rect, self.N_tri))

        mean_value_triangulation = mpl_tri.Triangulation(self.R_mean_values_tri_grid, self.Z_mean_values_tri_grid)
        tri_finder = mean_value_triangulation.get_trifinder()
        matching_triangles = tri_finder(self.R_mesh.ravel(), self.Z_mesh.ravel())
        matching_vertices = mean_value_triangulation.triangles[matching_triangles]
        r1, r2, r3 = mean_value_triangulation.x[matching_vertices].T
        z1, z2, z3 = mean_value_triangulation.y[matching_vertices].T
        barycentric_weights = self.get_barycentric_weights_triangle(r1, z1, r2, z2, r3, z3, self.R_mesh.ravel(),
                                                                    self.Z_mesh.ravel())

        for i, (vertex_indices, weights) in enumerate(zip(matching_vertices, barycentric_weights)):
            M[i, vertex_indices] = weights
        return csc_matrix(M, dtype=np.float32)

    def rect_to_tri_grid_linear(self):
        M = np.zeros((self.N_tri, self.N_rect))

        for i_tri_element in range(self.N_tri):
            Z_tri = self.Z_mean_values_tri_grid[i_tri_element]
            R_tri = self.R_mean_values_tri_grid[i_tri_element]
            dist_to_rect_anchor_points = (
                    np.square(Z_tri - self.anchor_Z_grid_rect)[:, None]
                    + np.square(R_tri - self.anchor_R_grid_rect)[None, :]
            )
            min_dist_flat_index = np.argmin(dist_to_rect_anchor_points, axis=None)

            i, j = np.unravel_index(min_dist_flat_index, dist_to_rect_anchor_points.shape)
            square_grid_interpolation_indices = np.ravel_multi_index(([i + 1, i, i + 1, i], [j, j, j + 1, j + 1]),
                                                                     (self.rect_height, self.rect_width))
            R1, Z1, R2, Z2 = self.anchor_indices_to_interpolation_points(i, j)

            weights = self.calculate_interpolation_weights_rect(R1, R2, Z1, Z2, R_tri, Z_tri)
            for w, idx in zip(weights, square_grid_interpolation_indices):
                M[i_tri_element, idx] = w
        return csc_matrix(M, dtype=np.float32)

    @staticmethod
    def calculate_interpolation_weights_rect(x1, x2, y1, y2, x, y):
        c = ((x2 - x1) * (y2 - y1))

        w11 = (x2 - x) * (y2 - y) / c
        w12 = (x2 - x) * (y - y1) / c
        w21 = (x - x1) * (y2 - y) / c
        w22 = (x - x1) * (y - y1) / c

        return w11, w12, w21, w22

    def anchor_indices_to_interpolation_points(self, i, j):
        R1, Z1 = self.R_grid_rect[j], self.Z_grid_rect[i + 1]
        R2, Z2 = self.R_grid_rect[j + 1], self.Z_grid_rect[i]
        return R1, Z1, R2, Z2

    @staticmethod
    def get_barycentric_weights_triangle(x1, y1, x2, y2, x3, y3, x, y):
        determinant = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        w1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / determinant
        w2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / determinant
        w3 = 1 - w1 - w2
        return np.array([w1, w2, w3]).T
