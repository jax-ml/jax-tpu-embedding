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

## [0.2.0] - 2026-09-09

### Added

- **Offline Checkpoint Converter**: Added an offline checkpoint conversion
  tool (`convert_cross_topology_checkpoint` / `:offline_converter`) to resize
  and migrate checkpoints across TPU topologies on CPU
  ([#793](https://github.com/jax-ml/jax-tpu-embedding/pull/793)).
- **Flax NNX Support**: Generalized the offline checkpoint converter to
  support Flax NNX models
  ([#828](https://github.com/jax-ml/jax-tpu-embedding/pull/828)).
- **Unstacking Without Target Topology**: Support unstacking and unsharding
  checkpoints directly into per-table representations without requiring
  restacking for a target topology
  ([#851](https://github.com/jax-ml/jax-tpu-embedding/pull/851)).
- **Example**: Added an end-to-end example demonstrating cross-topology
  checkpoint restoration (e.g., 2x2x2 checkpoint into a 2x2x1 model)
  ([#789](https://github.com/jax-ml/jax-tpu-embedding/pull/789)).
- **Configurable Activation Memory Limit**: Added configurable
  `activation_mem_bytes_limit` in SparseCore table stacking to customize
  memory thresholds and reduce kernel launch overhead without SRAM/VMEM OOMs
  ([#864](https://github.com/jax-ml/jax-tpu-embedding/pull/864)).
- **Quantization Config Support**: Added `quantization_config` support in
  SparseCore table stacking, grouping tables by quantization configuration and
  forwarding settings to lower properly to hardware
  ([#868](https://github.com/jax-ml/jax-tpu-embedding/pull/868)).
- **Multi-Axis Sharding**: Added support for multi-axis sharding and carry
  dtype invariance for SparseCore embeddings
  ([#895](https://github.com/jax-ml/jax-tpu-embedding/pull/895)).
- **1D Embedding Table Support**: Supported 1D embedding tables (embedding
  dimension = 1) across SparseCore forward pass and optimizer gradient update
  primitives ([#819](https://github.com/jax-ml/jax-tpu-embedding/pull/819),
  [#852](https://github.com/jax-ml/jax-tpu-embedding/pull/852),
  [#856](https://github.com/jax-ml/jax-tpu-embedding/pull/856)).
- **Automatic Input Squeezing**: Automatically detect and squeeze 2D inputs of
  shape `[x, 1]` to `[x]`, gracefully handling inputs with trailing singleton
  dimensions ([#819](https://github.com/jax-ml/jax-tpu-embedding/pull/819)).
- **64-bit Global Batch Sizes**: Supported `int64_t` global batch sizes
  alongside `int32_t` local batch sizes in preprocessing
  ([#799](https://github.com/jax-ml/jax-tpu-embedding/pull/799)).
- **`SparseCoreEmbed` API**: Expose a public `mesh` property and added
  `PartitionNames` and layout initializer types
  ([#896](https://github.com/jax-ml/jax-tpu-embedding/pull/896),
  [#897](https://github.com/jax-ml/jax-tpu-embedding/pull/897)).
- **Buffer Sizing Guidelines & Helpers**: Added buffer size estimation helpers
  and documented calculations for `max_ids_per_partition`,
  `max_unique_ids_per_partition`, and `required_buffer_size_per_sc`
  ([#857](https://github.com/jax-ml/jax-tpu-embedding/pull/857),
  [#891](https://github.com/jax-ml/jax-tpu-embedding/pull/891)).
- **Memory Contract Docs**: Documented the uninitialized trailing COO buffer
  memory contract
  ([#901](https://github.com/jax-ml/jax-tpu-embedding/pull/901)).

### Changed

- **JAX Upgrade**: Bumped minimum required JAX version to `>= 0.10.1`
  ([#821](https://github.com/jax-ml/jax-tpu-embedding/pull/821),
  [#882](https://github.com/jax-ml/jax-tpu-embedding/pull/882)).
- **Cloud & Cross-Filesystem Paths**: Migrated path manipulation in checkpoint
  and FDO utilities to `etils.epath` for seamless cloud and local filesystem
  compatibility
  ([#834](https://github.com/jax-ml/jax-tpu-embedding/pull/834)).
- **`StackedTableSpec.replace()`**: Made `StackedTableSpec` inherit from
  `PyTreeNode` to provide a type-safe `.replace()` method
  ([#831](https://github.com/jax-ml/jax-tpu-embedding/pull/831)).
- **StableHLO Custom Optimizers & Clipping**: Migrated SparseCore custom
  optimizer primitives from jaxpr to StableHLO lowering, with gradient
  clipping support in `SparseDenseMatmulGradOptimizerUpdate`
  ([#875](https://github.com/jax-ml/jax-tpu-embedding/pull/875),
  [#877](https://github.com/jax-ml/jax-tpu-embedding/pull/877),
  [#879](https://github.com/jax-ml/jax-tpu-embedding/pull/879)).
- **Optimizer Slot Variables**: Refactored optimizer slot variables to
  `NamedTuple` parameterized by generic `TypeVar`s, preserving field names
  across Orbax checkpoints and improving typing ergonomics
  ([#825](https://github.com/jax-ml/jax-tpu-embedding/pull/825),
  [#827](https://github.com/jax-ml/jax-tpu-embedding/pull/827)).
- **Hardware Device Mapping**: Updated `NUM_SC_PER_DEVICE_MAP` to support
  additional TPU generations and topologies
  ([#795](https://github.com/jax-ml/jax-tpu-embedding/pull/795)).
- **Documentation & Guides**: Updated developer and nightly installation
  documentation ([#791](https://github.com/jax-ml/jax-tpu-embedding/pull/791),
  [#812](https://github.com/jax-ml/jax-tpu-embedding/pull/812),
  [#846](https://github.com/jax-ml/jax-tpu-embedding/pull/846)).

### Removed

- **Python 3.10 & 3.11**: Dropped support for Python 3.10 and 3.11;
  Python 3.12+ is now required across `jax_tpu_embedding`
  ([#807](https://github.com/jax-ml/jax-tpu-embedding/pull/807)).

### Fixed

- **Registration Order Preservation**: Preserved table registration order
  during target feature spec reconstruction (`_recompute_target_specs`),
  preventing stack naming/layout mismatches during Orbax restoration
  ([#853](https://github.com/jax-ml/jax-tpu-embedding/pull/853)).
- **Validation & GIL Bug Fix**: Added parameter validation for devices,
  topologies, and input batches in C++ preprocessing (raising
  `InvalidArgumentError` instead of crashing), and fixed a GIL release bug
  during Python error handling
  ([#797](https://github.com/jax-ml/jax-tpu-embedding/pull/797)).
- **Device Count & Dtype Fix**: Cast SparseCore tables to `float32` and fixed
  `global_device_count` calculation
  ([#796](https://github.com/jax-ml/jax-tpu-embedding/pull/796)).
- **Pipelining Tracing Fix**: Allowed abstract evaluation (tracing with
  `jax.core.ShapedArray`) of `embedding_pipelining_utils.get_initial_state`
  ([#884](https://github.com/jax-ml/jax-tpu-embedding/pull/884)).

## [0.1.0] - 2026-06-11

Initial release of `jax-tpu-embedding`, providing TPU SparseCore (embedding
acceleration) support for JAX.

### Added

- **Hardware-Accelerated Embeddings**: Native TPU SparseCore primitives with
  Flax (Linen and NNX) integration.
- **Optimized Preprocessing**: High-throughput C++ preprocessing on host CPU.
- **Fused Optimizers**: Parameter updates fused into the backward pass on
  SparseCore (Adagrad, SGD, Adam, FTRL, LaProp) with Optax integration.
- **Memory Stability & Scalability**: Minibatching and Feedback Directed
  Optimization (FDO) to prevent OOMs on skewed batches.
- **Performance Optimizations**: Automatic table stacking, load-balanced
  partitioning, and embedding pipelining.
- **Checkpointing**: Orbax and TensorStore integration with serving-ready weight
  export.

[Unreleased]: https://github.com/jax-ml/jax-tpu-embedding/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/jax-ml/jax-tpu-embedding/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jax-ml/jax-tpu-embedding/releases/tag/v0.1.0
