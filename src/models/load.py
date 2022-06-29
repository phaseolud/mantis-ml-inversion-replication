from src.config.config import ModelConfig, TrainingConfig
import tensorflow as tf
from src.models.classical_iterative_methods import build_gd_model, build_sirt_model
from src.models.learned_proximal import build_unfolded_proximal_learned_sirt_model
from src.models.unet import build_sirt_informed_unet_model

model_map = {
    "gradient descent": build_gd_model,
    "sirt": build_sirt_model,
    "unet sirt": build_sirt_informed_unet_model,
    "unfolded proximal sirt": build_unfolded_proximal_learned_sirt_model
}


def load_model(model_config: ModelConfig) -> tf.keras.models.Model:
    if model_config.name not in model_map.keys():
        raise IndexError(f"The provided model name ({model_config.name} is not available. Choose "
                         f"a model from {model_map.keys()}")

    return model_map[model_config.name](model_config)


def compile_model(model: tf.keras.models.Model, training_config: TrainingConfig) -> tf.keras.models.Model:
    optimizer_config = {'class_name': training_config.optimizer, 'config': training_config.optimizer_params}
    optimizer = tf.keras.optimizers.get(optimizer_config)
    model.compile(optimizer=optimizer, loss=training_config.loss_function, metrics=['mae', 'mse'])
    return model
