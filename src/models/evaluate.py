from datetime import datetime

import tensorflow as tf
import definitions
from src.config.config import Config
from src.config.load import load_json_config_from_path
from src.data.load import load_datasets
from src.models.classical_iterative_methods import build_sirt_model
from src.models.load import load_model, compile_model


def evaluate_model_from_id(model_id: str):
    model, config = load_model_and_config_from_id(model_id)
    _, _, test_ds = load_datasets(config.dataset)
    print(f"Evaluating model {config.model.name} with {model.count_params()} parameters")
    model.evaluate(test_ds)
    evaluate_inference_time(model, test_ds, model_id)


def evaluate_inference_time(model: tf.keras.models.Model, dataset: tf.data.Dataset, model_id):
    profiler_options = tf.profiler.experimental.ProfilerOptions(host_tracer_level=3, python_tracer_level=1, device_tracer_level=1)
    profile_dir = definitions.LOGS_DIR / model_id / "profile"

    tf.profiler.experimental.start(str(profile_dir), profiler_options)
    model.predict(dataset.repeat(), steps=50)
    tf.profiler.experimental.stop()

    print("You can find the timing results of this model by using the following command from the root project directory:\n",
          f"tensorboard --logdir=logs/{model_id}/profile")


def load_model_and_config_from_id(model_id: str) -> (tf.keras.models.Model, Config):
    """
    Loads the model from the model and logs folder. The model_id corresponds to the datetime string that is used for the directory names.
    """
    model_saved_weights_path = definitions.MODELS_DIR / model_id
    logs_path = definitions.LOGS_DIR / model_id

    if not model_saved_weights_path.exists():
        raise FileNotFoundError(f"The saved weights for the model can not be found in {model_saved_weights_path}")
    if not logs_path.exists():
        raise FileNotFoundError(f"The logs can not be found for this model id. The logs are used to load the configuration of the model.")

    config = load_json_config_from_path(logs_path / "config.json")
    model = load_model(config.model)
    model = compile_model(model, config.training)

    checkpoint_path = tf.train.latest_checkpoint(model_saved_weights_path)
    model.load_weights(checkpoint_path).expect_partial()
    return model, config


def evaluate_sirt_model(config: Config):
    _, _, test_ds = load_datasets(config.dataset)
    config.model.set_shape_from_ds(test_ds)

    sirt_model = build_sirt_model(config.model)
    sirt_model = compile_model(sirt_model, config.training)

    print(f"Evaluating SIRT model with k={config.model.n_iterations} iterations and mu={config.model.mu}.")
    model_id = datetime.now().strftime("%Y%m%d-%H%M")
    sirt_model.evaluate(test_ds)

    evaluate_inference_time(sirt_model, test_ds, model_id)