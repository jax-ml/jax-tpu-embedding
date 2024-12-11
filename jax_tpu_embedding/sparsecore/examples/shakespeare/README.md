# A simple Shakespeare model using the Jax on SC API

## About

This directory contains two versions of the simple Shakespeare model that run a
distributed training with the embedding layer on SparseCore and a dense tower on
TensorCore. One is implemented using `pmap()` and the other uses `jit` +
`shard_map`.
