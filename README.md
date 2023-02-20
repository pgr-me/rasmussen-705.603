# rasmussen-705.603

## Portfolio for AI-Enabled Systems

## Assignment source code

Each assignment is organized and its Dockerfile and Python code is provided in its own directory in the repository. `Assignment4` is used as an example; other assignments follow the same pattern.
* `Dockerfile`, which the user can run to create an image and run code.
* `Assignment4.py`, which is the main Python script and which may import other local Python modules.
* `Assignment4.ipynb`, which is a Jupyter notebook that the user can use to interactively run code in the `.py` file.
* `readme.md`, which provides an overview of the assignment's purpose and instructions for running associated code.

## Data directory

Input data files for each assignment are provided in the [data](data/) directory in this repository. Each assignment has its own directory in [data](data/) and each assignment has an input directory, `raw`, and an output directory, `processed`. Some assignments may include intermediate outputs, which are saved in `interim`. Inputs are immutable - they're never changed. Code in this repository is idempotent, meaning if the user runs the same code again outputs will always be the same (notwithstanding differences owing to intended stochasticity / randomness in the code).

## DockerHub

Rather than clone the repository to run the code, the user can opt to pull Docker images, tagged by assignment, from the [associated DockerHub repo](https://hub.docker.com/repository/docker/pgrjhu/705.603/general). Pull the appropriately tagged image to run the desired assignment program. For instance, to run code for Assignment 4, execute `$ docker pull pgrjhu/705.603:module4-1.1`.

Additional instructions for running an image are provided in the [DockerHub repo README](https://hub.docker.com/repository/docker/pgrjhu/705.603/general).
