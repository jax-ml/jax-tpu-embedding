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
"""Reference implementation for SparseCore input preprocessing.

This module provides a readable, non-performant reference implementation of the
input preprocessing logic for SparseCore. It handles partitioning, sorting,
deduplication, and packing into CSR-formatted buffers.

This is intended for testing and verification purposes and does not necessarily
follow the most performant patterns used in production.
"""

import collections
from collections.abc import Sequence

import jax
from jax import numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core import constants
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np


ArrayLike = jnp.ndarray | np.ndarray

# (local_sc_id, global_sc_id)
PartitionKey = tuple[int, int]
# (local_col_id, local_row_id)
LocalCoordinate = tuple[int, int]
# List of (Coordinate, weight/gain)
PartitionData = list[tuple[LocalCoordinate, float]]
# Mapping from partition key to its data
PartitionMap = dict[PartitionKey, PartitionData]

Sample = Sequence[int]
SampleWeight = Sequence[float]

Feature2D = ArrayLike
Weight2D = ArrayLike

FeatureBatch = Feature2D | Sequence[Sample]
WeightBatch = Weight2D | Sequence[SampleWeight]

MinibatchedFeatures = Sequence[FeatureBatch]
MinibatchedWeights = Sequence[WeightBatch]


################################################################################
# Helper Functions
################################################################################


def _round_up(value: int, round_to: int) -> int:
  """Rounds up a value to the nearest multiple of `round_to`."""
  return value + (-value) % round_to


def _validate_partition_map(
    partition_map: PartitionMap,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
) -> None:
  """Validates that the partition does not exceed the maximum number of IDs or unique IDs."""
  for partition_data in partition_map.values():
    unique_ids = set(col_id for ((col_id, _), _) in partition_data)

    if len(partition_data) > max_ids_per_partition:
      raise ValueError(
          f"Partition has too many ids. {len(partition_data)} >"
          f" {max_ids_per_partition}."
      )

    if len(unique_ids) > max_unique_ids_per_partition:
      raise ValueError(
          f"Partition has too many unique ids. {len(unique_ids)} >"
          f" {max_unique_ids_per_partition}."
      )


def _preprocess_batch_to_partitions(
    features: FeatureBatch,
    features_weights: WeightBatch,
    num_scs: int,
    num_sc_per_device: int,
) -> PartitionMap:
  """Core logic: converts a single batch to structured partitions.

  Args:
    features: Input features for the current batch.
    features_weights: Input weights corresponding to each feature.
    num_scs: Total number of global SparseCores.
    num_sc_per_device: Number of SparseCores per device.

  Returns:
    A mapping from (local_sc_id, global_sc_id) to a list of
    ((local_col_id, local_row_id), gain) tuples, sorted by (col, row).
  """
  ##############################################################################
  # Step 1: Partitioning and Implicit Deduping
  ##############################################################################
  batch_size_per_mb = len(features)
  batch_size_per_sc = batch_size_per_mb // num_sc_per_device

  # partitions[(local_sc, global_sc)] -> defaultdict(float)
  # where keys are (col, row)
  partitions = collections.defaultdict(lambda: collections.defaultdict(float))

  for row_id, (sample_feat, sample_weight) in enumerate(
      zip(features, features_weights, strict=True)
  ):
    local_sc_id = row_id // batch_size_per_sc
    local_row_id = row_id % batch_size_per_sc
    for col_id, weight in zip(sample_feat, sample_weight, strict=True):
      global_sc_id = int(col_id) % num_scs
      local_col_id = int(col_id) // num_scs
      # Accumulate gain for the same (col, row) in this partition.
      # fmt: off
      partitions[(local_sc_id, global_sc_id)][(local_col_id, local_row_id)] += float(weight)
      # fmt: on

  ##############################################################################
  # Step 2: Partition-wise Sorting
  ##############################################################################
  return {
      key: sorted(deduped_data.items())
      for key, deduped_data in partitions.items()
  }


def _coo_buffer_tensor_size(
    max_ids_per_partition: int, num_scs: int, num_scs_per_device: int
) -> int:
  """Returns the size of the COO buffer tensor.

  Args:
    max_ids_per_partition: The maximum number of ids per SparseCore partition.
    num_scs: The number of global SparseCores.
    num_scs_per_device: The number of SparseCores per chip.

  Returns:
    The size of the COO buffer tensor.
  """
  return _round_up(max_ids_per_partition, 8) * num_scs_per_device * num_scs


def _pack_partitions_to_csr(
    all_minibatch_partitions: Sequence[PartitionMap],
    num_scs: int,
    num_sc_per_device: int,
    max_ids_per_partition: int,
    *,
    enable_minibatching: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """CSR Packer: pads and flattens partitions into final CSR buffers.

  Args:
    all_minibatch_partitions: List of partition mappings for each minibatch.
    num_scs: Total number of global SparseCores.
    num_sc_per_device: Number of SparseCores per device.
    max_ids_per_partition: Maximum number of ids per SparseCore partition. This
      value is used to determine the size of the static buffer of embedding,
      sample IDs and gains.
    enable_minibatching: Whether or not minibatching is enabled.

  Returns:
    A tuple (row_pointers, col_ids, row_ids, gains) forming the CSR wrapped COO
    input, where:
      row_pointers: Row pointers which point to the index for each sparse core
      partition.
      col_ids: Embedding ids.
      row_ids: Sample ids for each embedding id.
      gains: The weights for each embedding id.
  """
  ##############################################################################
  # Step 3: CSR Buffer Initialization
  ##############################################################################
  num_minibatches = len(all_minibatch_partitions)

  coo_buffer_size = (
      _coo_buffer_tensor_size(max_ids_per_partition, num_scs, num_sc_per_device)
      * num_minibatches
  )

  row_pointers = []
  embedding_ids = np.full(
      coo_buffer_size, constants.PADDING_VALUE, dtype=np.int32
  )
  sample_ids = np.full(coo_buffer_size, constants.PADDING_VALUE, dtype=np.int32)
  gains = np.full(coo_buffer_size, np.nan, dtype=np.float32)

  coo_buffer_size_per_sc = coo_buffer_size // num_sc_per_device

  ##############################################################################
  # Step 4: CSR Serialization and Alignment
  ##############################################################################
  coo_index = 0

  # Minibatching uses global index, otherwise we slice the COO buffer per SC.
  def _get_base_coo_index(local_sc_id: int) -> int:
    if enable_minibatching:
      return 0
    else:
      return local_sc_id * coo_buffer_size_per_sc

  def _get_coo_beginning_index(local_sc_id: int) -> int:
    if enable_minibatching:
      return _round_up(coo_index, 8)
    else:
      return _get_base_coo_index(local_sc_id)

  for local_sc_id in range(num_sc_per_device):
    base_coo_index = _get_base_coo_index(local_sc_id)
    coo_index = _get_coo_beginning_index(local_sc_id)

    for data_by_partition in all_minibatch_partitions:
      for global_sc_id in range(num_scs):
        partition_data = data_by_partition.get((local_sc_id, global_sc_id), [])

        # Populate COO data.
        for (col_id, row_id), gain in partition_data:
          embedding_ids[coo_index] = col_id
          sample_ids[coo_index] = row_id
          gains[coo_index] = gain
          coo_index += 1

        # Record row pointer (end of data).
        row_pointers.append(coo_index - base_coo_index)

        # Align for next partition.
        coo_index = _round_up(coo_index, 8)

      # Pad row pointers to max(num_scs, 8) per minibatch.
      row_pointers.extend([coo_index - base_coo_index] * max(0, 8 - num_scs))

  return (
      jnp.asarray(row_pointers, dtype=jnp.int32),
      jnp.asarray(embedding_ids),
      jnp.asarray(sample_ids),
      jnp.asarray(gains),
  )


################################################################################
# Main Entry Points
################################################################################


def preprocess_sparse_dense_matmul_input(
    features: FeatureBatch | MinibatchedFeatures,
    features_weights: WeightBatch | MinibatchedWeights,
    mesh: jax.sharding.Mesh,
    *,
    max_ids_per_partition: int = 64,
    max_unique_ids_per_partition: int = 64,
    num_sc_per_device: int = -1,
    sharding_strategy: str = "MOD",
    enable_minibatching: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Preprocesses standard input into SparseCore CSR wrapped COO format.

  Args:
    features: Input feature array or ragged list. In case of minibatching, it
      should be a list of feature arrays.
    features_weights: Input weights corresponding to each feature. In case of
      minibatching, it should be a list of weights arrays.
    mesh: JAX sharding mesh containing global and local device info.
    max_ids_per_partition: Maximum number of ids per SparseCore partition.
    max_unique_ids_per_partition: Maximum number of unique ids per SparseCore
      partition.
    num_sc_per_device: Number of sparse cores per device.
    sharding_strategy: Embedding table sharding strategy (only "MOD" supported).
    enable_minibatching: Whether or not minibatching is enabled.

  Returns:
    A tuple (row_pointers, col_ids, row_ids, gains) forming the CSR wrapped COO
    input, where:
      row_pointers: Row pointers which point to the index for each sparse core
      partition.
      col_ids: Embedding ids.
      row_ids: Sample ids for each embedding id.
      gains: The weights for each embedding id.
  """
  if max_ids_per_partition <= 0:
    raise ValueError(
        f"max_ids_per_partition must be positive, got {max_ids_per_partition}."
    )
  try:
    features_ndim = np.ndim(features)
  except ValueError:
    features_ndim = 1
  if not enable_minibatching and features_ndim not in (1, 2):
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
  num_scs = num_sc_per_device * global_device_count
  if not enable_minibatching:
    features = [features]
    features_weights = [features_weights]

  ##############################################################################
  # Step 1: Preprocess Minibatch to Partition Map
  ##############################################################################
  all_partitions: list[PartitionMap] = [
      _preprocess_batch_to_partitions(
          mb_feat, mb_weight, num_scs, num_sc_per_device
      )
      for mb_feat, mb_weight in zip(features, features_weights, strict=True)
  ]
  for partitions in all_partitions:
    _validate_partition_map(
        partitions, max_ids_per_partition, max_unique_ids_per_partition
    )

  ##############################################################################
  # Step 2: Pack to CSR
  ##############################################################################
  return _pack_partitions_to_csr(
      all_partitions,
      num_scs,
      num_sc_per_device,
      max_ids_per_partition,
      enable_minibatching=enable_minibatching,
  )
