# open-sigmoid
Open Source codebase of SIGMOID, the Scalable Infrastructure for Generic Model Optimization on Inhomogeneous Datasets

# Description

**SIGMOID** stands for **S**calable  **I**nfrastructure for **G**eneric **M**odel  **O**ptimization on **I**nhomogeneous  **D**atasets. It is an infrastructure in the sense that is is not a _single_ computer program but rather a collection of them. The main goal of `sigmoid` is to add scalabitility to an already existing model. In particular, to be trained with as much data as possible and then made available using a compute-efficient architecture.

A key distinction between `sigmoid` and already existing solutions is that `sigmoid` relies on data-driven scalability by exploiting the underlying grouping of data, that is, partitions of the dataset where every data-point is similar to one another. `sigmoid` then learns how to route data accordingly to train multiple instances of the original model partition-wise, that is, no model instance gets to "see" the entire dataset.

<p align="center">
  <img src="https://github.com/mindsdb/open-sigmoid/blob/staging/assets/figures/sigmoid_flow_diagram.png?raw=true alt="High level flow-diagram of sigmoid"/>
</p>
