from pathlib import Path
from typing import Optional

import typer

import definitions
from src.config.config import ModelConfig, DatasetConfig, Config
from src.config.load import load_yaml_config_from_path
from src.data.generate import generate_all_datasets
from src.models.evaluate import evaluate_model_from_id
from src.models.training import train_from_config
from src.models.evaluate import evaluate_sirt_model
app = typer.Typer()


@app.command()
def train(config_filepath: Path = definitions.ROOT_DIR / "config.yaml"):
    """
    Train a neural network to estimate the tomographic reconstruction. The used dataset, model and other configurations are defined in
    config.yaml. You can find more information about the default configuration in src/config/config.py.
    :param config_filepath: The configuration file (yaml) which determines the (hyper)parameters used for training
    """
    config = load_yaml_config_from_path(config_filepath)
    train_from_config(config)


@app.command()
def generate_dataset(shot_no: int, count: int):
    """
    Generate a synthetic dataset to train the networks on. The shot_no is used to determine which geometry matrix to use, and count defines
    the number of training samples to create. The number of validation and test samples are 20% of the number of training samples.
    """
    generate_all_datasets(shot_no, count)


@app.command()
def evaluate(model_id: str):
    """
    Evaluate the MSE and MAE for a trained model, identified by the model_id. The model id is a datetime string, used as a folder name in
    the logs and models directory.
    """
    evaluate_model_from_id(model_id)


@app.command()
def evaluate_sirt(config_path: Path):
    """
    Evaluate the SIRT algorithm for a certain configuration.yaml file. In this configuration file you can specify the number of iterations,
    the step size (mu) and the dataset properties (like batch size).
    """
    config = load_yaml_config_from_path(config_path)
    evaluate_sirt_model(config)


if __name__ == "__main__":
    app()
