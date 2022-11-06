from typing import Tuple

import numpy as np
import tensorflow as tf

from src.data.grid import create_grid_transformation_matrices
from src.data.utils import load_geometry_matrix


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

    def get_config(self):
        config = super().get_config()
        config.update({
            'M shape': self.M.shape
        })
        return config


def sparse_scipy_matrix_to_tf(sparse_matrix):
    sparse_matrix_coo = sparse_matrix.astype(np.float32).tocoo()
    indices = np.array([sparse_matrix_coo.row, sparse_matrix_coo.col]).transpose()
    return tf.SparseTensor(
        indices=indices,
        values=sparse_matrix_coo.data,
        dense_shape=sparse_matrix_coo.shape,
    )


def load_geometry_matrix_tf(geometry_id: str):
    geometry_matrix = load_geometry_matrix(geometry_id)
    return sparse_scipy_matrix_to_tf(geometry_matrix)


def create_grid_transformation_matrices_tf(geometry_id: str, output_grid_shape: Tuple[int, int]) -> Tuple[tf.SparseTensor, tf.SparseTensor]:
    """
    Calculates the transformation matrices: square grid <--> triangular inversion grid by minimizing the
    distance between the individual points in both grids. The returned matrices are sparse TF tensors.

    :returns: rect_to_tri_tf, tri_to_rect_tf
    """
    rect_to_tri, tri_to_rect = create_grid_transformation_matrices(geometry_id, output_grid_shape)
    tri_to_rect_tf = sparse_scipy_matrix_to_tf(tri_to_rect)
    rect_to_tri_tf = sparse_scipy_matrix_to_tf(rect_to_tri)

    return rect_to_tri_tf, tri_to_rect_tf
