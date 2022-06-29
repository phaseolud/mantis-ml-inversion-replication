from src.config.config import DatasetConfig
import tensorflow as tf
from typing import List
from src.data.preprocessing import load_data, list_datasets, normalize_ds


def load_datasets(data_config: DatasetConfig) -> List[tf.data.Dataset]:
    """
    Loads the preprocessed train, validation and test datasets
    :return: train_ds, val_ds, test_ds
    """
    datasets = [list_datasets(split, data_config.full_path) for split in ["train", "validation", "test"]]
    datasets = [ds.map(load_data) for ds in datasets]
    datasets = [normalize_ds(ds) for ds in datasets]
    datasets = [ds.batch(data_config.batch_size) for ds in datasets]
    datasets = [ds.prefetch(tf.data.AUTOTUNE) for ds in datasets]

    return datasets
