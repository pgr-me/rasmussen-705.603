# Assignment 9: Lookup-table- and deep-learning-based reinforcement learning

## Project description

This program provides code to train and analyze two implementations of a reinforcement learner that plays blackjack. The
first implementation uses a lookup-table to map state-action pairs to rewards and the second uses a neural network to
map state-action paris to rewards.

## Assignment structure

The assignment is sub-divided into the following parts:Neo4J and MongoDB folders. Each has its own set of scripts and
Dockerfiles to reproduce the work in a container.

* `SageMaker Tutorial.ipynb`: Provides a walkthrough based on an AWS SageMaker tutorial for how to train, test, and
  analyze an XGBoost classifier.
* `Local Blackjack.ipynb`: Based on a notebook provided in Module 10, this notebook trains and analyzes two
  reinforcement learner implementations: look-table and deep learning-based.
* `main.py`: Provides a script version of `Local Blackjack.ipynb`.
* `SageMaker Blackjack.ipynb`: Implements the `Local Blackjack.ipynb` in AWS SageMaker.

File and module organization - after creating and populating data directories - is as follows:

```
.
|-- Dockerfile
|-- Local\ Blackjack.ipynb
|-- SageMaker\ Tutorial.ipynb
|-- data
|   |-- interim
|   |-- processed
|   |   |-- local_notebook_lookup_total_rewards_over_time.png
|   |   |-- local_notebook_model_checkpoint.pth
|   |   |-- local_notebook_rewards_over_time.png
|   |   |-- local_notebook_total_cash_over_time.png
|   |   |-- local_py_dl_rewards_over_time.png
|   |   |-- local_py_dl_total_cash_over_time.png
|   |   |-- local_py_lookup_total_rewards_over_time.png
|   |   `-- local_py_lookup_total_rewards_over_time_no_exploration.png
|   `-- raw
|-- main.py
|-- qlearning
|   |-- __init__.py
|   |-- agent.py
|   |-- experience.py
|   |-- lookup_rl.py
|   |-- qnet.py
|   `-- train.py
`-- requirements.txt
```

## Docker

### Local runs

Rather than clone the repository to run the code, the user can opt to pull Docker image for this assignment
from [the DockerHub repo](https://hub.docker.com/repository/docker/pgrjhu/705.603/general).

Pull the image: `$ docker pull pgrjhu/705.603:a9`

Instantiate the container:

```
docker run \
-p 8888:8888 \
-dit \
--name a9 \
pgrjhu/705.603:a9
```

To run the SageMaker notebook, refer to the `SageMaker Tutorial.ipynb` to get signed up to AWS SageMaker. Then,
run `SageMaker Tutorial.ipynb` in SageMaker. Finally, you'll be ready to run this `SageMaker Blackjack.ipynb`.

Additional instructions for running an image are provided in
the [DockerHub repo README](https://hub.docker.com/repository/docker/pgrjhu/705.603/general).


