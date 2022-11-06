from typing import List, Optional, Tuple
from pathlib import Path
from pydantic import BaseModel
import tensorflow as tf
import definitions


class BaseConfigModel(BaseModel):
    def __getitem__(self, item):
        return getattr(self, item)


class DatasetConfig(BaseConfigModel):
    geometry_id: str
    batch_size: int = 32
    output_rect_grid_shape: Tuple[int, int] = (256, 256)
    mask_unobservable_output: bool = True
    noise: dict = {}
    copy_input_to_output: bool = False
    @property
    def path(self) -> Path:
        return Path("processed") / self.geometry_id

    @property
    def full_path(self) -> Path:
        return definitions.DATA_DIR / self.path


class ModelConfig(BaseConfigModel):
    name: str
    geometry_id: str

    encoder_filters: List[int] = [64, 64, 64, 128, 128, 128]
    decoder_filters: List[int] = [128, 128, 128, 64, 64, 64]
    bottleneck_filters: List[int] = [128]

    activation_function: str = "relu"
    final_activation_function: str = "relu"

    mu: float = 1.99
    unfolded_intermediate_output_loss: bool = False
    n_iterations: Optional[int] = None
    input_shape: Tuple[int, int] = None
    output_shape: Tuple[int, int] = None

    @property
    def input_shape_ext(self) -> Tuple[int, int, int]:
        return self.input_shape + (1,)

    @property
    def output_shape_ext(self) -> Tuple[int, int, int]:
        return self.output_shape + (1,)

    def set_shape_from_ds(self, dataset: tf.data.Dataset):
        sample_image, sample_inversion = dataset.__iter__().next()
        self.input_shape = (sample_image.shape[1], sample_image.shape[2])
        self.output_shape = (sample_inversion.shape[1], sample_inversion.shape[2])


class TrainingConfig(BaseConfigModel):
    epochs: int = 20
    optimizer: str = "adam"
    optimizer_params: dict = {}
    loss_function: str = "mse"
    cycle_loss: bool = False


class Config(BaseConfigModel):
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig = TrainingConfig()
