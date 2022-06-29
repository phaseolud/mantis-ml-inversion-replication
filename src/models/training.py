import datetime
import tensorflow as tf
import definitions
from src.config.config import Config
from src.data.load import load_datasets
from src.models.load import load_model, compile_model


def train_from_config(config: Config):
    log_dir = definitions.ROOT_DIR / "logs" / datetime.datetime.now().strftime("%Y%m%d-%H%M")
    models_checkpoint_dir = definitions.ROOT_DIR / "models" / datetime.datetime.now().strftime("%Y%m%d-%H%M")

    train_ds, val_ds, _ = load_datasets(config.dataset)
    config.model.set_shape_from_ds(train_ds)
    model = load_model(config.model)
    model = compile_model(model, config.training)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(models_checkpoint_dir, save_weights_only=True, save_best_only=True)
    model.fit(train_ds, epochs=config.training.epochs, validation_data=val_ds, callbacks=[tensorboard_callback, model_checkpoint_callback])
