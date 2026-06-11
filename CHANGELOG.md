# Changelog

<!--

Changelog follow the https://keepachangelog.com/ standard (at least the headers)

This allow to:

* auto-parsing release notes during the automated releases from github-action:
  https://github.com/marketplace/actions/pypi-github-auto-release
* Have clickable headers in the rendered markdown

To release a new version (e.g. from `1.0.0` -> `2.0.0`):

* Create a new `# [2.0.0] - YYYY-MM-DD` header and add the current
  `[Unreleased]` notes.
* At the end of the file:
  * Define the new link url:
  `[2.0.0]: https://github.com/jax-ml/jax-tpu-embedding/compare/v1.0.0...v2.0.0`
  * Update the `[Unreleased]` url: `v1.0.0...HEAD` -> `v2.0.0...HEAD`

-->

## [Unreleased]

## [0.1.0] - 2026-06-11

* Initial release of `jax-tpu-embedding`, providing TPU SparseCore (embedding acceleration) support for JAX.
  * **Hardware-Accelerated Embeddings**: Native TPU SparseCore primitives with Flax (Linen and NNX) integration.
  * **Optimized Preprocessing**: High-throughput C++ preprocessing on host CPU.
  * **Fused Optimizers**: Parameter updates fused into the backward pass on SparseCore (Adagrad, SGD, Adam, FTRL, LaProp) with Optax integration.
  * **Memory Stability & Scalability**: Minibatching and Feedback Directed Optimization (FDO) to prevent OOMs on skewed batches.
  * **Performance Optimizations**: Automatic table stacking, load-balanced partitioning, and embedding pipelining.
  * **Checkpointing**: Orbax and TensorStore integration with serving-ready weight export.

[Unreleased]: https://github.com/jax-ml/jax-tpu-embedding/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jax-ml/jax-tpu-embedding/releases/tag/v0.1.0
