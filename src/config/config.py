from typing import List, Optional, Dict, Tuple, Union
from pathlib import Path
from pydantic import BaseModel

import definitions


class BaseConfigModel(BaseModel):
    def __getitem__(self, item):
        return getattr(self, item)


class DatasetConfig(BaseConfigModel):
    name: str
    shot_no: int
    batch_size: int = 32
    shuffle_buffer: int = 1000
    data_scaling: str = "zero_one"
    noise: dict = {}
    copy_input_to_output: bool = False

    @property
    def path(self) -> Path:
        return Path("processed") / "mantis" / self.name / str(self.shot_no)

    @property
    def full_path(self) -> Path:
        return definitions.DATA_DIR / self.path


class ModelConfig(BaseConfigModel):
    name: str
    shot_no: int

    encoder_filters: List[int] = [64, 64, 64, 128, 128, 128]
    decoder_filters: List[int] = [128, 128, 128, 64, 64, 64]
    bottleneck_filters: List[int] = [128]

    activation_function: str = "relu"
    final_activation_function: str = "relu"

    double_conv: bool = False
    unfolded_intermediate_output_loss: bool = False

    normalization_layer: str = "instance_normalization"

    n_iterations_unfolding: int = 5


class TrainingConfig(BaseConfigModel):
    epochs: int = 20
    optimizer: str = "adam"
    optimizer_params: dict = {}
    loss_function: str = "mse"
    cycle_loss: bool = False
    loss_weights: Optional[Union[List[float], Dict[str, float]]] = None
    additional_callbacks: List[Tuple[str, Dict]] = []
    staged_training: bool = False
    staged_epochs_per_unfold: List[int] = []


class Config(BaseConfigModel):
    project: str
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig = TrainingConfig()
