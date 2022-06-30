import datetime
import tensorflow as tf
import definitions
from src.config.config import Config
from src.data.load import load_datasets
from src.models.load import load_model, compile_model
import json


def train_from_config(config: Config):
    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    log_dir = definitions.ROOT_DIR / "logs" / time_str
    models_checkpoint_path = definitions.ROOT_DIR / "models" / time_str / time_str

    train_ds, val_ds, _ = load_datasets(config.dataset)
    config.model.set_shape_from_ds(train_ds)
    model = load_model(config.model)
    model = compile_model(model, config.training)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(models_checkpoint_path, save_weights_only=True, save_best_only=True)
    model.fit(train_ds, epochs=config.training.epochs, validation_data=val_ds, callbacks=[tensorboard_callback, model_checkpoint_callback])

    with open(log_dir / 'config.json', 'w') as file:
        json.dump(config.dict(), file)
