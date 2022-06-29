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
or when using a different configuration file use
```shell
python cli.py train {your_configuration_file.yaml}
```

You can find the inversion grid and the geometry matrix for shot number 65903 on [this google drive](https://drive.google.com/drive/folders/1hxuSuCPjPOhwNOfia9q8m1M7VJkKuTa1?usp=sharing).
