# JAX TPU Embedding (JAX SparseCore)

[![Build and test](https://github.com/jax-ml/jax-tpu-embedding/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/jax-ml/jax-tpu-embedding/actions/workflows/build_and_test.yml)
[![PyPI version](https://img.shields.io/pypi/v/jax-tpu-embedding)](https://pypi.org/project/jax-tpu-embedding/)
[![Documentation Status](https://readthedocs.org/projects/jax-tpu-embedding/badge/?version=latest)](https://jax-tpu-embedding.readthedocs.io/en/latest/?badge=latest)

[**Quick Overview**](https://jax-tpu-embedding.readthedocs.io/en/latest/overview.html)
| [**Install guide**](#installation)
| [**Reference docs**](https://jax-tpu-embedding.readthedocs.io/en/latest/)

## What is JAX TPU Embedding?

JAX SparseCore provides support for leveraging the SparseCore accelerators
present in TPU generations starting with TPU v5.

SparseCores are specialized tiled processors engineered for high-performance
acceleration of workloads that involve irregular, sparse memory access and
computation. While they excel at tasks like embedding lookups (common in deep
learning recommendation models), their capabilities extend to accelerating a
variety of other dynamic and sparse workloads on large datasets stored in High
Bandwidth Memory (HBM).

This is a research project, not an official Google product. Expect sharp edges.
Please help by trying it out, reporting bugs, and letting us know what you
think!

## Installation

You can install JAX TPU Embedding from PyPI:

```bash
pip install jax-tpu-embedding
```

*Note: To use TPU acceleration, you must run in an environment with access to TPU v5+ hardware and have the appropriate `jax` and `jaxlib` TPU releases installed (see the [JAX installation guide](https://github.com/google/jax#installation)).*

## Documentation

For detailed guides, specifications, and tutorials, see the [JAX TPU Embedding Documentation](https://jax-tpu-embedding.readthedocs.io/en/latest/).

*   [Quick Overview for Users](https://jax-tpu-embedding.readthedocs.io/en/latest/overview.html)
*   [Embedding Specification](https://jax-tpu-embedding.readthedocs.io/en/latest/embedding.html)
*   [Input Processing](https://jax-tpu-embedding.readthedocs.io/en/latest/input_processing.html)
*   [Tutorials (Flax & Primitives)](https://jax-tpu-embedding.readthedocs.io/en/latest/#tutorials)
