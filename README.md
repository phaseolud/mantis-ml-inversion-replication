# Machine Learning Accelerated Tomographic Reconstruction for Multispectral Imaging on TCV

This package contains code to reproduce results for the machine learning based tomographic reconstruction.
First clone the repository to your machine, and create a virtual environment (and activate the venv).
```shell
python -m venv venv
```

Then install the required packages, and the code of this project (or used the requirements-lock.txt):
```shell
pip install -r requirements.txt
pip install -e .
```

To get started with this project you need two additional data files: the inversion grid and a geometry matrix.
Copy the inversion grid file to `./data/utils/inversion_grid.mat` and copy the geometry matrix for a specific shot to `./data/utils/geometry_matrices/{shot_no}.npz`.

Now you can run the tutorials in the `notebooks` directory, or use the command line interface.
To get started with the command line interface generate some data with:
```shell
python cli.py generate-dataset {shot_no} {number_of_training_samples}
```

Then you can edit the `config.yaml` file as you wish, but the default one in the repository should work as well.
To train a model with the parameters defined in the configuration file, simply run
```shell
python cli.py train
```
or when using a different configuration file use (with ? indicating an optional parameter)
```shell
python cli.py train {?your_configuration_file.yaml}
```

You can find the inversion grid and the geometry matrix for shot number 65903 on [this google drive](https://drive.google.com/drive/folders/1hxuSuCPjPOhwNOfia9q8m1M7VJkKuTa1?usp=sharing).


## Reproduction
The code in this repository is very similar to the code used to generate the figures and results in the halfway paper.
Configurations that should yield similar results for both the informed U-net and the deep unfolded network can both be found in the
`configurations` directory. 25000 training samples had been generated for the training.

You can evaluate the MSE and MAE of a trained model as:
```shell
python cli.py evaluate {training/model_id}
```
where the `{training/model_id}` can be found in the `logs` or `models` folder, being a datetime string.
You can analyse the inference time by starting tensorboard in `logs/{model_id}/profile` and then selecting
the Profile tab in the top bar.

[//]: # (We can get the same metrics for the SIRT algorithm as)

[//]: # (```shell)

[//]: # (python cli.py evaluate-sirt {?n_iterations} {?step_size})

[//]: # (```)


