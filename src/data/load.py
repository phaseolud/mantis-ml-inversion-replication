from typing import List

import tensorflow as tf

from src.config.config import DatasetConfig
from src.data.grid import create_grid_transformation_matrices
from src.data.preprocessing import list_datasets, normalize_ds, mask_unobservable_output, observable_mask_from_gm, \
    add_noise, load_ds


def load_datasets(data_config: DatasetConfig) -> List[tf.data.Dataset]:
    """
    Loads the dataset as tf.Dataset and apply all preprocessing functions
    """

    datasets = [
        list_datasets(split, data_config.path)
        for split in ["train", "test", "validation"]
    ]

    datasets = [load_ds(ds) for ds in datasets]
    datasets = [normalize_ds(ds) for ds in datasets]

    if data_config.mask_unobservable_output:
        mask_tri = observable_mask_from_gm(data_config.geometry_id)
        _, tri_to_rect = create_grid_transformation_matrices(data_config.geometry_id,
                                                             data_config.output_rect_grid_shape)
        mask = (tri_to_rect @ mask_tri).reshape(data_config.output_rect_grid_shape)[..., None]

        datasets = [mask_unobservable_output(ds, data_config, mask) for ds in datasets]

    datasets = [add_noise(ds, data_config) for ds in datasets]
    datasets = [ds.batch(data_config.batch_size) for ds in datasets]
    datasets = [ds.prefetch(tf.data.AUTOTUNE) for ds in datasets]

    return datasets
