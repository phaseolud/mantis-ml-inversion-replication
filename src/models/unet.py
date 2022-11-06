from typing import List

import tensorflow as tf
import tensorflow_addons as tfa

from src.config.config import ModelConfig
from src.models.classical_iterative_methods import create_sirt_variables_tf
from src.models.utils import BatchSparseDenseMatmul, create_grid_transformation_matrices_tf


def build_sirt_informed_unet_model(model_config: ModelConfig):
    inputs = tf.keras.layers.Input(model_config.input_shape + (1,))
    inputs_flat = tf.keras.layers.Flatten()(inputs)

    # back project measured image
    _, F_y = create_sirt_variables_tf(model_config.geometry_id)
    _, tri_to_rect = create_grid_transformation_matrices_tf(model_config.geometry_id, model_config.output_shape)

    Fy = BatchSparseDenseMatmul(F_y)(inputs_flat)
    Fy_square_grid_flat = BatchSparseDenseMatmul(tri_to_rect)(Fy)
    Fy_square = tf.keras.layers.Reshape(model_config.output_shape + (1,))(Fy_square_grid_flat)

    unet_outputs = UnetBase(model_config.encoder_filters, model_config.decoder_filters,
                    model_config.bottleneck_filters, model_config.activation_function)(Fy_square)

    outputs = tf.keras.layers.Conv2D(1, (3, 3), padding="same", activation=model_config.final_activation_function,
                                     name="inversion")(unet_outputs)

    return tf.keras.models.Model(inputs, outputs)


class UnetBase(tf.keras.models.Model):
    def __init__(self, encode_filters: List[int], decode_filters: List[int], bottleneck_filters: List[int], activation_function: str):
        super().__init__()

        self.encoder_filters = encode_filters
        self.bottleneck_filters = bottleneck_filters
        self.decoder_filters = decode_filters

        self.encoder = Encoder(encode_filters, activation_function)
        self.bottleneck = Bottleneck(bottleneck_filters, activation_function)
        self.decoder = Decoder(decode_filters, activation_function)

    def call(self, x):
        x, skips = self.encoder(x)
        skips_rev = skips[::-1]
        x = self.bottleneck(x)
        x = self.decoder(x, skips_rev)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "encode_filters": self.encoder_filters,
                "decode_filters": self.decoder_filters,
                "bottleneck_filters": self.bottleneck_filters,
            }
        )
        return config


class Encoder(tf.keras.layers.Layer):
    def __init__(self, filters: List[int], activation: str):
        super().__init__()
        self.filters = filters
        self.encoder_blocks = [
            EncodeBlock(f, activation) for f in filters
        ]

    def call(self, x: tf.Tensor) -> (tf.Tensor, List[tf.Tensor]):
        skips = []
        for e_block in self.encoder_blocks:
            x, skip = e_block(x)
            skips.append(skip)
        return x, skips

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config


class Decoder(tf.keras.layers.Layer):
    def __init__(self, filters: List[int], activation: str):
        super().__init__()
        self.filters = filters
        self.decoder_blocks = [
            DecodeBlock(f, activation) for f in filters
        ]

    def call(self, x: tf.Tensor, skips: List[tf.Tensor]):
        for d_block, skip in zip(self.decoder_blocks, skips):
            x = d_block(x, skip)
        return x


class Bottleneck(tf.keras.layers.Layer):
    def __init__(self, filters: List[int], activation: str):
        super().__init__()
        self.filters = filters
        self.blocks = [
            tf.keras.layers.Conv2D(f, (3, 3), padding="same", activation=activation)
            for f in filters
        ]

    def call(self, x):
        for b_block in self.blocks:
            x = b_block(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config


class EncodeBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, activation: str):
        super().__init__()
        self.filters = filters

        self.conv1 = tf.keras.layers.Conv2D(filters, (3, 3), padding="same", activation=activation)
        self.conv2 = tf.keras.layers.Conv2D(filters, (3, 3), padding="same", activation=activation)
        self.norm = tfa.layers.InstanceNormalization()
        self.max_pooling = tf.keras.layers.MaxPooling2D((2, 2))

    def call(self, x: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = skip = self.norm(x)
        x = self.max_pooling(x)
        return x, skip

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config


class DecodeBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, activation: str):
        super().__init__()
        self.filters = filters

        self.conv1 = tf.keras.layers.Conv2D(filters, (3, 3), padding="same", activation=activation)
        self.conv2 = tf.keras.layers.Conv2D(filters, (3, 3), padding="same", activation=activation)
        self.norm = tfa.layers.InstanceNormalization()
        self.upsampling = tf.keras.layers.UpSampling2D((2, 2))
        self.concatenate = tf.keras.layers.Concatenate()

    def call(self, x: tf.Tensor, skip: tf.Tensor) -> tf.Tensor:
        x = self.upsampling(x)
        x = self.concatenate([x, skip])
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.norm(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config
