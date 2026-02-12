# A simple Shakespeare model using the JAX SparseCore API

## About

This directory contains two versions of the simple Shakespeare model that run a
distributed training with the embedding layer on SparseCore and a dense tower on
TensorCore. It is implemented using `jit` + `shard_map`.
