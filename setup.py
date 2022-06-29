from setuptools import find_packages, setup
from pathlib import Path

setup(
    name="mantis_ml_inversion",
    packages=find_packages(),
    version="0.1.0",
    description="Machine Learning Accelerated Tomographic Reconstruction for Imaging data in a Tokamak.",
    author="Loek van Leeuwen",
    license="MIT"
)

# create the empty directories
(Path().absolute() / "data" / "utils" / "geometry_matrices").mkdir(exist_ok=True, parents=True)
(Path().absolute() / "logs").mkdir(exist_ok=True)
(Path().absolute() / "models").mkdir(exist_ok=True)
(Path().absolute() / ".cache").mkdir(exist_ok=True)