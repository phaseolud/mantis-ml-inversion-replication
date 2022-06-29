import os
from pathlib import Path
from typing import Tuple

import tensorflow as tf

import definitions


def load_data(folder_path: str) -> Tuple[tf.Tensor, tf.Tensor]:
    inversion_filename = folder_path + os.sep + "inversion.png"
    cam_img_filename = folder_path + os.sep + "cam_img.png"
    scaling_cam_filename = folder_path + os.sep + "scaling_cam.txt"
    scaling_inv_filename = folder_path + os.sep + "scaling_inv.txt"

    cam_img_file = tf.io.read_file(cam_img_filename)
    cam_img = tf.io.decode_png(cam_img_file)

    scaling_cam = tf.strings.to_number(tf.io.read_file(scaling_cam_filename))
    cam_img_scaled = tf.cast(cam_img, tf.float32)

    inversion_file = tf.io.read_file(inversion_filename)
    inversion = tf.cast(tf.io.decode_png(inversion_file), tf.float32)

    scaling_inv = tf.strings.to_number(tf.io.read_file(scaling_inv_filename))
    inversion_scaled = tf.cast(inversion, tf.float32) * (scaling_cam / scaling_inv)

    return cam_img_scaled, inversion_scaled


def normalize_ds(dataset: tf.data.Dataset) -> tf.data.Dataset:
    def normalize_zero_one(x, y):
        return x / 255.0, y / 255.0

    return dataset.map(
        lambda x, y: normalize_zero_one(x, y),
        num_parallel_calls=tf.data.AUTOTUNE
    )


def list_datasets(split: str, path: Path):
    """
    List all the folders for the mantis dataset
    """
    ds = tf.data.Dataset.list_files(str(path / split / "*"), shuffle=(split == "train"))
    return ds
