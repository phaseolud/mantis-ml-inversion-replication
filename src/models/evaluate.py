import tensorflow as tf
import definitions
from src.config.config import Config
from src.config.load import load_json_config_from_path
from src.data.load import load_datasets
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
    model.predict(dataset.repeat(), steps=100)
    tf.profiler.experimental.stop()


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
