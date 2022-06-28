from src.config.config import ModelConfig
import tensorflow as tf
from src.models.classical_iterative_methods import build_gd_model, build_sirt_model
from src.models.unet import build_sirt_informed_unet_model
model_map = {
    "gradient descent": build_gd_model,
    "sirt": build_sirt_model,
    "unet sirt": build_sirt_informed_unet_model,
    "learned proximal sirt": ...
}


def load_model(model_config: ModelConfig) -> tf.keras.models.Model:
    if model_config.name not in model_map.keys():
        raise IndexError(f"The provided model name ({model_config.name} is not available. Choose "
                         f"a model from {model_map.keys()}")

    return model_map[model_config.name](model_config)

