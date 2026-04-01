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


def _pack_partitions_to_csr(
    all_minibatch_partitions: Sequence[PartitionMap],
    num_minibatches: int,
    num_scs: int,
    num_sc_per_device: int,
    max_ids_per_partition: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """CSR Packer: pads and flattens partitions into final CSR buffers.

  Args:
    all_minibatch_partitions: List of partition mappings for each minibatch.
    num_minibatches: Total number of minibatches.
    num_scs: Total number of global SparseCores.
    num_sc_per_device: Number of SparseCores per device.
    max_ids_per_partition: Maximum number of ids per SparseCore partition. This
      value is used to determine the size of the static buffer of embedding,
      sample IDs and gains.

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
  row_pointers_size_per_sc = max(num_scs * num_minibatches, 8)

  coo_buffer_size = (
      _coo_buffer_tensor_size(max_ids_per_partition, num_scs, num_sc_per_device)
      * num_minibatches
  )

  lhs_row_pointers_by_sc_id = np.zeros(
      (num_sc_per_device, row_pointers_size_per_sc), np.int32
  )
  lhs_local_embedding_ids = np.full(
      coo_buffer_size, constants.PADDING_VALUE, np.int32
  )
  lhs_local_sample_ids = np.full(
      coo_buffer_size, constants.PADDING_VALUE, np.int32
  )
  lhs_gains = np.full(coo_buffer_size, np.nan, np.float32)

  ids_per_sc_buffer = coo_buffer_size // num_sc_per_device

  ##############################################################################
  # Step 4: CSR Serialization and Alignment
  ##############################################################################
  for local_sc_id in range(num_sc_per_device):
    sc_offset = local_sc_id * ids_per_sc_buffer
    current_ids = 0
    ptr_idx = 0

    for data_by_partition in all_minibatch_partitions:
      for global_sc_id in range(num_scs):
        partition_data = data_by_partition.get((local_sc_id, global_sc_id), [])

        for (col_id, row_id), gain in partition_data:
          lhs_local_embedding_ids[sc_offset + current_ids] = col_id
          lhs_local_sample_ids[sc_offset + current_ids] = row_id
          lhs_gains[sc_offset + current_ids] = gain
          current_ids += 1

        # Record point (end of data for this partition)
        lhs_row_pointers_by_sc_id[local_sc_id, ptr_idx] = current_ids
        ptr_idx += 1

        # Post-pad this partition to 8-word multiple for the next partition's
        # start.
        while current_ids % 8 != 0:
          current_ids += 1

    # Pad remaining pointers to match buffer size
    if ptr_idx < row_pointers_size_per_sc:
      lhs_row_pointers_by_sc_id[local_sc_id, ptr_idx:] = current_ids

  return (
      jnp.asarray(lhs_row_pointers_by_sc_id.reshape(-1)),
      jnp.asarray(lhs_local_embedding_ids),
      jnp.asarray(lhs_local_sample_ids),
      jnp.asarray(lhs_gains),
  )


################################################################################
# Main Entry Points
################################################################################


def preprocess_sparse_dense_matmul_input(
    features: FeatureBatch,
    features_weights: WeightBatch,
    mesh: jax.sharding.Mesh,
    *,
    max_ids_per_partition: int = 64,
    num_sc_per_device: int = -1,
    sharding_strategy: str = "MOD",
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Preprocesses standard input into SparseCore CSR wrapped COO format.

  Args:
    features: Input feature array or ragged list.
    features_weights: Input weights corresponding to each feature.
    mesh: JAX sharding mesh containing global and local device info.
    max_ids_per_partition: Maximum number of ids per SparseCore partition.
    num_sc_per_device: Number of sparse cores per device.
    sharding_strategy: Embedding table sharding strategy (only "MOD" supported).

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
  num_scs = num_sc_per_device * global_device_count

  ##############################################################################
  # Step 1: Preprocess Minibatch to Partition Map
  ##############################################################################
  # Standard case: exactly one minibatch.
  partitions: PartitionMap = _preprocess_batch_to_partitions(
      features, features_weights, num_scs, num_sc_per_device
  )

  ##############################################################################
  # Step 2: Pack to CSR
  ##############################################################################
  return _pack_partitions_to_csr(
      [partitions], 1, num_scs, num_sc_per_device, max_ids_per_partition
  )


def preprocess_sparse_dense_matmul_input_minibatched(
    features: MinibatchedFeatures,
    features_weights: MinibatchedWeights,
    mesh: jax.sharding.Mesh,
    *,
    max_ids_per_partition: int = 64,
    num_sc_per_device: int = -1,
    sharding_strategy: str = "MOD",
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Preprocesses minibatched input into SparseCore CSR wrapped COO format.

  Args:
    features: List of batches, each either a 2D array or a ragged list.
    features_weights: Nested list of weights matching `features` structure.
    mesh: JAX sharding mesh containing global and local device info.
    max_ids_per_partition: Maximum number of ids per SparseCore partition.
    num_sc_per_device: Number of sparse cores per device.
    sharding_strategy: Embedding table sharding strategy (only "MOD" supported).

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
  if len(features) != len(features_weights):
    raise ValueError("features and features_weights must have the same length.")
  if sharding_strategy != "MOD":
    raise ValueError("Currently only MOD sharding strategy is supported")

  num_minibatches = len(features)
  global_device_count = len(mesh.devices)
  if global_device_count <= 0:
    raise ValueError("global_device_count must be positive.")

  num_sc_per_device = (
      num_sc_per_device
      if num_sc_per_device > 0
      else utils.num_sparsecores_per_device(mesh.devices.item(0))
  )
  num_scs = num_sc_per_device * global_device_count

  ##############################################################################
  # Step 1: Preprocess Each Minibatch to Partition Maps
  ##############################################################################
  # Process each minibatch independently.
  all_partitions: list[PartitionMap] = [
      _preprocess_batch_to_partitions(
          mb_feat, mb_weight, num_scs, num_sc_per_device
      )
      for mb_feat, mb_weight in zip(features, features_weights, strict=True)
  ]

  ##############################################################################
  # Step 2: Pack All Minibatches to CSR
  ##############################################################################
  return _pack_partitions_to_csr(
      all_partitions,
      num_minibatches,
      num_scs,
      num_sc_per_device,
      max_ids_per_partition,
  )


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
  # Round up the value of `max_ids_per_partition` to nearest multiple of 8.
  max_ids_per_partition = max_ids_per_partition + (-max_ids_per_partition) % 8
  return max_ids_per_partition * num_scs_per_device * num_scs
