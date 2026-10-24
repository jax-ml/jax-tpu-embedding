# Copyright 2024 The JAX SC Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reference implementation for Megacore SparseCore input preprocessing.

This module provides a readable, non-performant reference implementation of the
Megacore (chip-level) CSR input preprocessing logic for SparseCore. Unlike the
per-SparseCore CSR format in `input_preprocessing.py`, Megacore CSR merges all
local SparseCores on a device into a single CSR buffer with one row pointer per
destination device.

This is intended for testing, verification, and documentation purposes and does
not follow the parallelized patterns used in the C++ implementation.
"""

# This reference implementation reuses private helpers of the sibling
# `input_preprocessing` module in the same package.
# pylint: disable=protected-access

import collections
from collections.abc import Sequence

import jax
from jax import numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core import input_preprocessing
from jax_tpu_embedding.sparsecore.lib.core import pybind_input_preprocessing
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np

MinibatchingMode = pybind_input_preprocessing.MinibatchingMode
FeatureBatch = input_preprocessing.FeatureBatch
WeightBatch = input_preprocessing.WeightBatch
Sample = input_preprocessing.Sample
SampleWeight = input_preprocessing.SampleWeight

# (linearized_emb_id, sample_id)
MegacoreCoordinate = tuple[int, int]
# Sorted list of ((linearized_emb_id, sample_id), gain) for one global_sc_id
BucketData = list[tuple[MegacoreCoordinate, float]]
# Mapping from global_sc_id to its sorted COO entries
BucketMap = dict[int, BucketData]

_MIN_MEGACORE_COO_BUFFER_SIZE = 256
_MEGACORE_COO_BUFFER_ALIGNMENT = 128
_MAX_MINIBATCHING_BUCKETS = 64
_MEGACORE_PADDING_ID = np.iinfo(np.int32).max


def _preprocess_batch_to_megacore_buckets(
    features: FeatureBatch,
    features_weights: WeightBatch,
    num_scs: int,
    num_sc_per_device: int,
    rows_per_sc: int,
) -> BucketMap:
  """Partitions, deduplicates, and sorts a batch into per-global-SC buckets.

  Args:
    features: Input features for the current device batch.
    features_weights: Input weights corresponding to each feature.
    num_scs: Total number of global SparseCores across all devices.
    num_sc_per_device: Number of SparseCores per device.
    rows_per_sc: Vocabulary rows per SparseCore shard, used to offset each
      destination SparseCore's local embedding IDs within the chip-level table.

  Returns:
    A mapping from `global_sc_id` to a list of
    `((linearized_emb_id, sample_id), gain)` tuples sorted by
    `(linearized_emb_id, sample_id)`.
  """
  # buckets[global_sc_id][(linearized_emb_id, sample_id)] -> accumulated gain
  buckets: dict[int, dict[MegacoreCoordinate, float]] = collections.defaultdict(
      lambda: collections.defaultdict(float)
  )

  feat_seq: Sequence[Sample] = input_preprocessing._to_sequence(features)
  weight_seq: Sequence[SampleWeight] = input_preprocessing._to_sequence(
      features_weights
  )

  for sample_id, (sample_feat, sample_weight) in enumerate(
      zip(feat_seq, weight_seq, strict=True)
  ):
    for col_id, weight in zip(sample_feat, sample_weight, strict=True):
      global_sc_id = int(col_id) % num_scs
      dest_sc = global_sc_id % num_sc_per_device
      emb_id = int(col_id) // num_scs
      linearized_emb_id = dest_sc * rows_per_sc + emb_id
      buckets[global_sc_id][(linearized_emb_id, sample_id)] += float(weight)

  return {
      global_sc_id: sorted(deduped_data.items())
      for global_sc_id, deduped_data in buckets.items()
  }


def _megacore_coo_buffer_size(
    max_ids_per_partition: int,
    global_device_count: int,
    num_sc_per_device: int,
    hbm_word_size_in_4b: int,
    minibatching_mode: MinibatchingMode,
    suggested_coo_buffer_size_per_device: int | None = None,
) -> int:
  """Computes the Megacore COO buffer size per device."""
  num_scs = global_device_count * num_sc_per_device
  theoretical_max = (
      input_preprocessing._round_up(max_ids_per_partition, hbm_word_size_in_4b)
      * num_sc_per_device
      * num_scs
      * (
          _MAX_MINIBATCHING_BUCKETS
          if minibatching_mode != MinibatchingMode.DISABLED
          else 1
      )
  )
  if suggested_coo_buffer_size_per_device is not None:
    base_size = input_preprocessing._round_up(
        suggested_coo_buffer_size_per_device,
        hbm_word_size_in_4b * num_sc_per_device,
    )
    if base_size > theoretical_max:
      raise ValueError(
          f"suggested_coo_buffer_size_per_device ({base_size}) exceeds"
          f" theoretical_max ({theoretical_max})."
      )
  else:
    base_size = theoretical_max

  aligned_with_device_padding = (
      input_preprocessing._round_up(base_size, hbm_word_size_in_4b)
      + global_device_count * hbm_word_size_in_4b
  )
  return max(
      _MIN_MEGACORE_COO_BUFFER_SIZE,
      input_preprocessing._round_up(
          aligned_with_device_padding, _MEGACORE_COO_BUFFER_ALIGNMENT
      ),
  )


def _pack_buckets_to_megacore_csr(
    buckets: BucketMap,
    global_device_count: int,
    num_sc_per_device: int,
    hbm_word_size_in_4b: int,
    coo_buffer_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Packs per-global-SC buckets into chip-level Megacore CSR buffers."""
  row_pointers: list[int] = []
  embedding_ids = np.zeros(coo_buffer_size, dtype=np.int32)
  sample_ids = np.zeros(coo_buffer_size, dtype=np.int32)
  gains = np.zeros(coo_buffer_size, dtype=np.float32)

  coo_index = 0
  for device_id in range(global_device_count):
    for dest_sc in range(num_sc_per_device):
      global_sc_id = device_id * num_sc_per_device + dest_sc
      for (linearized_emb_id, sample_id), gain in buckets.get(global_sc_id, []):
        if coo_index >= coo_buffer_size:
          break
        embedding_ids[coo_index] = linearized_emb_id
        sample_ids[coo_index] = sample_id
        gains[coo_index] = gain
        coo_index += 1

    row_pointers.append(coo_index)
    while coo_index % hbm_word_size_in_4b != 0 and coo_index < coo_buffer_size:
      embedding_ids[coo_index] = _MEGACORE_PADDING_ID
      sample_ids[coo_index] = _MEGACORE_PADDING_ID
      gains[coo_index] = 0.0
      coo_index += 1

  row_pointers_size = input_preprocessing._round_up(
      global_device_count, hbm_word_size_in_4b
  )
  row_pointers.extend([coo_index] * (row_pointers_size - global_device_count))

  return (
      jnp.asarray(row_pointers, dtype=jnp.int32),
      jnp.asarray(embedding_ids),
      jnp.asarray(sample_ids),
      jnp.asarray(gains),
  )


def preprocess_sparse_dense_matmul_input(
    features: FeatureBatch,
    features_weights: WeightBatch,
    mesh: jax.sharding.Mesh,
    *,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    rows_per_sc: int = 0,
    num_sc_per_device: int = -1,
    hbm_word_size_in_4b: int = -1,
    sharding_strategy: str = "MOD",
    minibatching_mode: MinibatchingMode | str = MinibatchingMode.DISABLED,
    suggested_coo_buffer_size_per_device: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Preprocesses input into Megacore (chip-level) SparseCore CSR format.

  Args:
    features: Input feature array or ragged list for a single device.
    features_weights: Input weights corresponding to each feature.
    mesh: JAX sharding mesh containing global and local device info.
    max_ids_per_partition: Maximum number of IDs per SparseCore partition.
    max_unique_ids_per_partition: Maximum number of unique IDs per SparseCore
      partition.
    rows_per_sc: Number of vocabulary rows per SparseCore shard (`max_col_id +
      1` in `FeatureMetadataInStack`). Required to be positive when
      `num_sc_per_device > 1`.
    num_sc_per_device: Number of SparseCores per device.
    hbm_word_size_in_4b: The HBM word size in 4-byte words.
    sharding_strategy: Embedding table sharding strategy (only "MOD" supported).
    minibatching_mode: The minibatching mode (`DISABLED` or `DEVICE`; `HOST` is
      not supported for Megacore CSR).
    suggested_coo_buffer_size_per_device: Optional explicit COO buffer size per
      device before Megacore alignment.

  Returns:
    A tuple `(row_pointers, embedding_ids, sample_ids, gains)` forming the
    Megacore CSR wrapped COO input.
  """
  if isinstance(minibatching_mode, str):
    minibatching_mode = MinibatchingMode(minibatching_mode)
  if minibatching_mode == MinibatchingMode.HOST:
    raise ValueError(
        "Megacore CSR does not support host minibatching; use"
        " MinibatchingMode.DEVICE."
    )
  if max_ids_per_partition <= 0:
    raise ValueError(
        f"max_ids_per_partition must be positive, got {max_ids_per_partition}."
    )
  features_ndim = input_preprocessing._resolve_feature_input_ndim(features)
  if features_ndim not in (1, 2):
    raise ValueError(f"features must be 1D or 2D, got {features_ndim}D.")
  if len(features) != len(features_weights):
    raise ValueError("features and features_weights must have the same length.")
  if sharding_strategy != "MOD":
    raise ValueError("Currently only MOD sharding strategy is supported")

  global_device_count = len(mesh.devices)
  if global_device_count <= 0:
    raise ValueError("global_device_count must be positive.")

  num_sc_per_device = (
      num_sc_per_device
      if num_sc_per_device > 0
      else utils.num_sparsecores_per_device(mesh.devices.item(0))
  )
  hbm_word_size_in_4b = (
      hbm_word_size_in_4b
      if hbm_word_size_in_4b > 0
      else utils.hbm_word_size_in_4b(mesh.devices.item(0))
  )
  if num_sc_per_device > 1 and rows_per_sc <= 0:
    raise ValueError(
        "rows_per_sc must be positive when num_sc_per_device > 1, got"
        f" {rows_per_sc}."
    )
  num_scs = num_sc_per_device * global_device_count

  if minibatching_mode == MinibatchingMode.DISABLED:
    partitions = input_preprocessing._preprocess_batch_to_partitions(
        features, features_weights, num_scs, num_sc_per_device
    )
    input_preprocessing._validate_partition_map(
        partitions, max_ids_per_partition, max_unique_ids_per_partition
    )

  buckets = _preprocess_batch_to_megacore_buckets(
      features, features_weights, num_scs, num_sc_per_device, rows_per_sc
  )
  coo_buffer_size = _megacore_coo_buffer_size(
      max_ids_per_partition=max_ids_per_partition,
      global_device_count=global_device_count,
      num_sc_per_device=num_sc_per_device,
      hbm_word_size_in_4b=hbm_word_size_in_4b,
      minibatching_mode=minibatching_mode,
      suggested_coo_buffer_size_per_device=suggested_coo_buffer_size_per_device,
  )
  return _pack_buckets_to_megacore_csr(
      buckets=buckets,
      global_device_count=global_device_count,
      num_sc_per_device=num_sc_per_device,
      hbm_word_size_in_4b=hbm_word_size_in_4b,
      coo_buffer_size=coo_buffer_size,
  )
