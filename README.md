# open-sigmoid

Open Source codebase of SIGMOID, the Scalable Infrastructure for Generic Model Optimization on Inhomogeneous Datasets.

## Description

**SIGMOID** stands for **S**calable  **I**nfrastructure for **G**eneric **M**odel  **O**ptimization on **I**nhomogeneous  **D**atasets. It is an infrastructure in the sense that is is not a _single_ computer program but rather a _collection_ of them. The main goal of `sigmoid` is to provide scalabitility to an already existing model. In short, this means

- Making it possible to train a arbitrary model using as much data as possible without changing the model at all.
- Provide the output product in a form-factor that suits large-scale HPC compute infrastructure.
- Accomplish the above with zero Human intervention.

## High-level overview

### Data-driven model scaling

A key distinction between `sigmoid` and already existing solutions is that `sigmoid` relies on the training data itself to provide scalability. We call this method "data-driven model scaling" (D2MS).

`sigmoid` attempts to achieve D2MS by combining self-supervised Deep Learning methods and unsupervised clustering algorithms to detect underlying data partitions in the dataset; loosely speaking, a partition is a subset of the data where every all elements are similar to one another.

`sigmoid` then trains an arbitrary number of models in a way that makes every model become specialized (fine-tuned) for data coming from one particular partition. This way, no instance of the model gets to "see" the entire dataset.

Finally, after the training process, `sigmoid` provides the user with a "pool" of models (the specialists) and a "routing" model (a switch). Inference then comes down to feeding new data to the switch, which redirects the data to the respective specialist to perform the actual inference.

![High level flow-diagram of `sigmoid`](/assets/figures/sigmoid_flow_diagram.png)

## Installation

`sigmoid` is written in Python, so to install it from source need a Python Environment (recommended to use `pyenv`) and `poetry`.

```:shell
pip install poetry
poetry install --only main
```