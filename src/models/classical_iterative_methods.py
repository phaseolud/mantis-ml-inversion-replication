import tensorflow as tf
import numpy as np
from scipy.sparse import diags
from src.config.config import ModelConfig
from src.data.utils import load_geometry_matrix
from src.models.utils import sparse_scipy_matrix_to_tf, create_grid_transformation_matrices_tf, BatchSparseDenseMatmul


def build_classical_iterative_model(model_config: ModelConfig, F_x: tf.Tensor, F_y: tf.SparseTensor):
    rect_to_tri, _ = create_grid_transformation_matrices_tf(model_config.geometry_id, model_config.output_shape)
    inputs = tf.keras.layers.Input(shape=model_config.input_shape + (1,))
    inputs_flat = tf.keras.layers.Flatten()(inputs)

    x_hat = tf.zeros(F_x.shape[0])
    Fy = BatchSparseDenseMatmul(F_y)(inputs_flat)

    outputs_flat = iterative_solver_tf(F_x, Fy, x_hat, model_config.n_iterations)
    outputs = tf.keras.layers.Reshape(model_config.output_shape + (1,))(
        BatchSparseDenseMatmul(rect_to_tri)(outputs_flat)
    )

    model = tf.keras.models.Model(inputs, outputs)
    return model


def build_sirt_model(model_config: ModelConfig):
    mu = tf.Variable(model_config.mu, dtype=tf.float32)
    F_x, F_y = create_sirt_variables_tf(model_config.geometry_id)
    return build_classical_iterative_model(model_config, mu * F_x, mu * F_y)


def build_gd_model(model_config: ModelConfig):
    mu = tf.Variable(model_config.mu, dtype=tf.float32)
    F_x, F_y = create_gd_variables_tf(model_config.geometry_id)
    return build_classical_iterative_model(model_config, mu * F_x, mu * F_y)


def iterative_solver_tf(F_x: tf.Tensor, Fy: tf.Tensor, x_hat: tf.Tensor, n_iterations: int) -> tf.Tensor:
    """
    Approximate a solution for the inverse problem using a gradient descent like algorithm.
    If using the basic gradient descent algorithm:
        F_x = mu G^T G \n
        Fy = F_y @ y, where F_y = mu G^T \n
    If using the SIRT algorithm:
        F_x = mu C G^T R G \n
        Fy = F_y @ y, where F_y = mu C G^T R \n
    where F_y is the back projection operator
    """
    for i in range(n_iterations):
        x_hat = x_hat + (Fy - tf.linalg.matvec(F_x, x_hat))

        # constrain the solution (emissivity) to positive values only
        x_hat = tf.keras.activations.relu(x_hat)
    return x_hat


def create_sirt_variables_tf(geometry_id: str):
    """
    Calculates the solution estimate operator and the back projection operator for the SIRT algorithm for a specific
    geometry matrix (shot_number). The operators are transformed to tf.Tensor types. We use a separate function for the
    tf variables, because loading from disk_cache resulted in problems before for tf variables.
    :returns: F_x = C G^T R G (dense),  F_y = C G^T R (sparse)
    """
    F_x_np, F_y_np = create_sirt_variables(geometry_id)
    F_y = sparse_scipy_matrix_to_tf(F_y_np)
    F_x = tf.constant(F_x_np)
    return F_x, F_y


def create_gd_variables_tf(geometry_id: str):
    """
    Calculates the solution estimate operator and the back projection operator for the gradient descent algorithm for
    a specific geometry matrix (shot_number). The operators are transformed to tf.Tensor types. We use a separate
    function for the tf variables, because loading from disk_cache resulted in problems before for tf variables.

    :returns: F_x = G^T G (dense),  F_y = G^T (sparse)
    """
    F_x_np, F_y_np = create_gd_variables(geometry_id)
    F_x = tf.constant(F_x_np)
    F_y = sparse_scipy_matrix_to_tf(F_y_np)
    return F_x, F_y


def create_sirt_variables(geometry_id: str):
    """
    Calculates the solution estimate operator and the back projection operator for the SIRT algorithm for a specific
    geometry matrix (shot_number).

    :returns: F_x = C G^T R G (dense),  F_y = C G^T R (sparse)
    """
    geometry_matrix = load_geometry_matrix(geometry_id)

    # The geometry matrix can contain full zero rows/columns, so ignore the divide by zero, and remove the nan-values
    with np.errstate(divide='ignore'):
        R = diags(1 / np.squeeze(np.sum(geometry_matrix, axis=1).A))
        C = diags(1 / np.squeeze(np.sum(geometry_matrix, axis=0).A))

    F_y = np.nan_to_num(C @ geometry_matrix.transpose() @ R)
    F_x = np.nan_to_num(C @ geometry_matrix.transpose() @ R @ geometry_matrix).todense().A
    return F_x, F_y


def create_gd_variables(geometry_id: str):
    """
    Calculates the solution estimate operator and the back projection operator for the gradient descent algorithm for
    a specific geometry matrix (shot_number).
    :returns: F_x = G^T G (dense),  F_y = G^T (sparse)
    """
    geometry_matrix = load_geometry_matrix(geometry_id)
    F_x = (geometry_matrix.transpose() @ geometry_matrix).todense().A
    F_y = geometry_matrix.transpose()

    return F_x, F_y
