from typing import Tuple

import numpy as np
import tensorflow as tf

from src.data.utils import load_geometry_matrix, create_grid_transformation_matrices


class BatchSparseDenseMatmul(tf.keras.layers.Layer):
    """
    A layer for calculating a sparse-dense matrix multiplication for a batched input.
    """
    def __init__(self, M: tf.SparseTensor):
        super().__init__()
        self.M = M

    def inner_matmul(self, x):
        return tf.reshape(
            tf.sparse.sparse_dense_matmul(self.M, tf.reshape(x, (-1, 1))), (-1,)
        )

    def call(self, x_batched):
        return tf.map_fn(self.inner_matmul, x_batched)


def sparse_scipy_matrix_to_tf(sparse_matrix):
    sparse_matrix_coo = sparse_matrix.astype(np.float32).tocoo()
    indices = np.array([sparse_matrix_coo.row, sparse_matrix_coo.col]).transpose()
    return tf.SparseTensor(
        indices=indices,
        values=sparse_matrix_coo.data,
        dense_shape=sparse_matrix_coo.shape,
    )


def load_geometry_matrix_tf(shot_no: int):
    geometry_matrix = load_geometry_matrix(shot_no)
    return sparse_scipy_matrix_to_tf(geometry_matrix)


def create_grid_transformation_matrices_tf(output_grid_shape: Tuple[int, int]) -> Tuple[tf.SparseTensor, tf.SparseTensor]:
    """
    Calculates the transformation matrices: square grid <--> triangular inversion grid by minimizing the
    distance between the individual points in both grids. The returned matrices are sparse TF tensors.

    :returns: tri_to_square_grid_mat, square_to_inv_grid_mat
    """
    tri_to_square_grid, square_to_tri_grid = create_grid_transformation_matrices(output_grid_shape)
    tri_to_square_grid_tf = sparse_scipy_matrix_to_tf(tri_to_square_grid)
    square_to_tri_grid_tf = sparse_scipy_matrix_to_tf(square_to_tri_grid)

    return tri_to_square_grid_tf, square_to_tri_grid_tf
