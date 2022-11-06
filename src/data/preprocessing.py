import os
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

import definitions
from src.config.config import DatasetConfig
from src.data.utils import load_geometry_matrix


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


def observable_mask_from_gm(geometry_id: str):
    gm = load_geometry_matrix(geometry_id)
    return np.logical_not(np.isclose(np.squeeze(gm.sum(0).A), 0))


def add_noise(dataset: tf.data.Dataset, data_config: DatasetConfig) -> tf.data.Dataset:
    """
    Adds noise to the dataset.

    Args:
        dataset: The dataset to add noise to.
        data_config: The data config.

    Returns:
        The dataset with noise added.
    """
    if "constant" in data_config["noise"]:
        dataset = add_constant_noise(
            dataset, data_config, data_config["noise"]["constant"]["factor"]
        )

    if "gaussian" in data_config["noise"]:
        dataset = add_gaussian_noise(
            dataset, data_config, data_config["noise"]["gaussian"]["stddev"]
        )

    return dataset


def add_constant_noise(dataset: tf.data.Dataset, data_config: DatasetConfig, factor: float) -> tf.data.Dataset:
    """
    Adds constant noise to the dataset.

    Args:
        dataset: The dataset to add noise to.
        factor: The factor to multiply the data by.

    Returns:
        The dataset with constant noise (=offset) added.
    """
    port_mask = load_viewport_mask(data_config.geometry_id)
    return dataset.map(
        lambda x, y: (x + tf.where(port_mask, factor * tf.reduce_max(x), 0), y)
    )


def add_gaussian_noise(dataset: tf.data.Dataset, data_config: DatasetConfig, stddev: float) -> tf.data.Dataset:
    """
    Adds gaussian noise to the dataset.

    Args:
        dataset: The dataset to add noise to.
        stddev: The standard deviation of the gaussian noise.

    Returns:
        The dataset with gaussian noise added.
    """
    port_mask = load_viewport_mask(data_config.geometry_id)
    return dataset.map(
        lambda x, y: (
            x + tf.where(port_mask, tf.random.normal(tf.shape(x), stddev=stddev), 0),
            y,
        )
    )


@lru_cache
def load_viewport_mask(geometry_id: str):
    mask_file = definitions.DATA_DIR / "utils" / f"{geometry_id}_mask.npy"
    mask = np.load(mask_file)
    return mask


def mask_unobservable_output(dataset: tf.data.Dataset, data_config: DatasetConfig, mask) -> tf.data.Dataset:
    """
    Mask out the unobservable part by settings these values to zero. For evaluation and training
    """
    if data_config.mask_unobservable_output:
        return dataset.map(lambda x, y: (x, y * mask))
    return dataset

def load_ds(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """
    Load the dataset from a dataset containing a list of files.
    """
    return dataset.map(load)


def load(dirname) -> Tuple[tf.Tensor, tf.Tensor]:
    inversion_filename = dirname + os.sep + "inversion.png"
    cam_img_filename = dirname + os.sep + "cam_img.png"
    scaling_cam_filename = dirname + os.sep + "scaling_cam.txt"
    scaling_inv_filename = dirname + os.sep + "scaling_inv.txt"

    cam_img_file = tf.io.read_file(cam_img_filename)
    cam_img = tf.io.decode_png(cam_img_file, dtype=tf.uint16)

    scaling_cam = tf.strings.to_number(tf.io.read_file(scaling_cam_filename))
    cam_img_scaled = tf.cast(cam_img, tf.float32) / 255.0

    inversion_file = tf.io.read_file(inversion_filename)
    inversion = tf.cast(tf.io.decode_png(inversion_file, dtype=tf.uint16), tf.float32)

    scaling_inv = tf.strings.to_number(tf.io.read_file(scaling_inv_filename))
    inversion_scaled = tf.cast(inversion, tf.float32) * (scaling_cam / scaling_inv) / 255.0

    return cam_img_scaled, inversion_scaled
