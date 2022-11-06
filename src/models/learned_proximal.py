from src.config.config import ModelConfig
import tensorflow as tf
import tensorflow_addons as tfa
from src.models.classical_iterative_methods import create_sirt_variables_tf
from src.models.unet import UnetBase
from src.models.utils import create_grid_transformation_matrices_tf, BatchSparseDenseMatmul


def build_unfolded_proximal_learned_sirt_model(model_config: ModelConfig):
    inputs = tf.keras.layers.Input(model_config.input_shape_ext)
    inputs_flat = tf.keras.layers.Flatten()(inputs)

    F_x, F_y = create_sirt_variables_tf(model_config.geometry_id)
    rect_to_tri, tri_to_rect = create_grid_transformation_matrices_tf(model_config.geometry_id, model_config.output_shape)

    Fy = BatchSparseDenseMatmul(F_y)(inputs_flat)
    Fy_square_grid_flat = BatchSparseDenseMatmul(tri_to_rect)(Fy)

    x_hat = tf.zeros_like(Fy_square_grid_flat)
    intermediate_outputs = []
    for i in range(model_config.n_iterations):
        # model based sirt layer
        x_hat = SquareLSirtBlock(F_x, tri_to_rect, rect_to_tri, model_config.mu)(x_hat, Fy_square_grid_flat)

        # learned proximal
        x_hat = proximal_operator_unet(model_config)(x_hat)
        x_hat = tfa.layers.InstanceNormalization()(x_hat)
        intermediate_outputs.append(x_hat)
        x_hat = tf.keras.layers.Flatten()(x_hat)

    outputs = tf.keras.layers.Reshape(model_config.output_shape_ext)(x_hat)

    if model_config.unfolded_intermediate_output_loss:
        outputs = [outputs] + intermediate_outputs
    return tf.keras.models.Model(inputs, outputs)


class SquareLSirtBlock(tf.keras.layers.Layer):
    def __init__(self, F_x: tf.Tensor, tri_to_square: tf.SparseTensor, square_to_tri: tf.SparseTensor, mu_init: float):
        super().__init__()
        self.F_x = F_x
        self.tri_to_square = tri_to_square
        self.square_to_tri = square_to_tri
        self.mu = tf.Variable(mu_init, trainable=True, dtype=tf.float32)

    def call(self, x: tf.Tensor, Fy: tf.SparseTensor) -> tf.Tensor:
        x_t = BatchSparseDenseMatmul(self.square_to_tri)(x)
        Fx_t = tf.linalg.matvec(self.F_x, x_t)
        Fx = BatchSparseDenseMatmul(self.tri_to_square)(Fx_t)
        z = x + self.mu * (Fy - Fx)
        return tf.keras.activations.relu(z)  # apply ReLU activation to constrain to positive emissivities only


def proximal_operator_unet(model_config: ModelConfig):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Reshape(model_config.output_shape_ext),
            UnetBase(
                model_config.encoder_filters,
                model_config.decoder_filters,
                model_config.bottleneck_filters,
                model_config.activation_function,
            ),
            tf.keras.layers.Conv2D(
                1,
                (3, 3),
                activation=model_config.final_activation_function,
                padding="same",
            )
        ]
    )
