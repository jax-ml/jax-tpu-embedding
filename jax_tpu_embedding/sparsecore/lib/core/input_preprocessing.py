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
"""SparseCore input preprocessing functions."""

from typing import Tuple

import jax
from jax import numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core import constants
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np

ArrayLike = jnp.ndarray | np.ndarray


def preprocess_sparse_dense_matmul_input(
    features: ArrayLike,
    features_weights: ArrayLike,
    mesh: jax.sharding.Mesh,
    max_ids_per_partition: int = 64,
    num_sc_per_device: int = -1,
    sharding_strategy: str = "MOD",
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Preprocesses the input for the sparse-dense matmul.

  Args:
    features: Input feature array. The input feature array must either be a 2D
      array or a 1D numpy array of lists (in cases of ragged tensors). In the 2D
      array case, the first dimension is the batch size and the second dimension
      is the feature length. In the 1D case, the first dimension is the batch
      size and the individual nested arrays are the training samples.
    features_weights: Input weights corresponding to each feature. The shape of
      the features_weights has to be the exact same as the input features.
    mesh: An instance of the jax.sharding.Mesh object. This contains the global
      devices info and local devices info which we need to do the preprocessing.
    max_ids_per_partition: Maximum number of ids per SparseCore partition. This
      value is used to determine the size of the static buffer of embedding,
      sample IDs and gains.
    num_sc_per_device: Number of sparse cores per device.
    sharding_strategy: The sharding strategy to use to shard the embedding
      table.

  Returns:
    A tuple of four arrays forms the csr wrapped coo input:
      row_pointers: Row pointers which points to index for each sparse core
      partition.
      col_ids: Embedding ids.
      row_ids: Sample ids for each embedding ids.
      gains: The weight for each embedding id.

  Raises:
    NotImplementedError
  """
  if sharding_strategy != "MOD":
    raise ValueError("Currently only MOD sharding strategy is supported")

  if features.shape != features_weights.shape:
    raise ValueError("features and weights must have the same shape.")

  if np.ndim(features) != 2 and np.ndim(features) != 1:
    raise ValueError("features must be a 2D array or a 1D numpy array.")

  if max_ids_per_partition <= 0:
    raise ValueError("max_ids_per_partition must be positive.")

  global_device_count = len(mesh.devices)
  if global_device_count <= 0:
    raise ValueError("global_device_count must be positive.")

  if num_sc_per_device <= 0:
    num_sc_per_device = utils.num_sparsecores_per_device(mesh.devices.item(0))

  # Global number of sparse cores.
  num_scs = num_sc_per_device * global_device_count

  # `features` is the batch for a single device. The batch_size for each core is
  # then the length of features divided by the number of cores.
  batch_size_per_sc = features.shape[0] // num_sc_per_device
  row_pointers_size_per_sc = max(num_scs, 8)
  coo_buffer_size = _coo_buffer_tensor_size(
      max_ids_per_partition, num_scs, num_sc_per_device
  )

  ##############################################################################
  # Step 1: Convert ragged tensor to COO tensor.
  ##############################################################################
  coo_tensor = []
  for row_id, (sample_input_tensor, sample_input_weights) in enumerate(
      zip(features, features_weights)
  ):
    if len(sample_input_tensor) != len(sample_input_weights):
      raise ValueError(
          "features and weights must have the same shape. Detected mismatched"
          f" shapes at row {row_id}."
      )

    for col_id, weight in zip(sample_input_tensor, sample_input_weights):
      coo_tensor.append((row_id, col_id, weight))

  ##############################################################################
  # Step 2: Order COO tensor by the sample SC allocation.
  ##############################################################################

  # Define a function to map a COO tensor element to its SC allocation.
  sample_to_sc_id = lambda s: (
      s[0] // batch_size_per_sc,  # Local SC ID of the training sample.
      s[1] % num_scs,  # Global SC ID of the column.
  )
  coo_tensor.sort(
      key=lambda v: (
          *sample_to_sc_id(v),
          v[1],  # Column ID.
          v[0],  # Training Sample ID.
          v[2],  # Weight/Gain.
      )
  )

  ##############################################################################
  # Step 3: Compute the row pointers for each group of ids.
  ##############################################################################
  lhs_row_pointers_by_sc_id = np.empty(
      (num_sc_per_device, row_pointers_size_per_sc),
      np.int32,
  )
  padded_coo_tensor_by_sc_id = np.empty(
      (
          num_sc_per_device,
          coo_buffer_size // num_sc_per_device,
      ),
      object,
  )

  # Calculate row pointers for each partition.
  coo_tensor_index = 0
  prev_row_id = -1
  prev_col_id = -1
  for local_sc_id in range(num_sc_per_device):
    lhs_row_index = 0
    padded_coo_tensor_index = 0

    # Prepare COO tensor for each global SC.
    for global_sc_id in range(num_scs):
      sc_id_tuple = (local_sc_id, global_sc_id)

      # Grab all the samples for the current local and global SC.
      while (
          coo_tensor_index < len(coo_tensor)
          and sample_to_sc_id(coo_tensor[coo_tensor_index]) == sc_id_tuple
      ):
        row_id, col_id, gain = coo_tensor[coo_tensor_index]
        coo_tensor_index += 1

        if row_id == prev_row_id and col_id == prev_col_id:
          # If the row ids and col ids are both same as the previous one,
          # dedup the id by adding the gains.
          padded_coo_tensor_index -= 1
          gain += padded_coo_tensor_by_sc_id[local_sc_id][
              padded_coo_tensor_index
          ][2]

        # Append the sample to the COO tensor.
        padded_coo_tensor_by_sc_id[local_sc_id][padded_coo_tensor_index] = (
            row_id % batch_size_per_sc,
            col_id // num_scs,
            gain,
        )
        padded_coo_tensor_index += 1

        prev_row_id = row_id
        prev_col_id = col_id

      # Commit the current row pointer.
      lhs_row_pointers_by_sc_id[local_sc_id][
          lhs_row_index
      ] = padded_coo_tensor_index
      lhs_row_index += 1

      # Add padding to coo tensor to make sure it is rounded up by 8
      while padded_coo_tensor_index % 8 != 0:
        padded_coo_tensor_by_sc_id[local_sc_id][padded_coo_tensor_index] = (
            constants.PADDING_VALUE,
            constants.PADDING_VALUE,
            np.nan,
        )
        padded_coo_tensor_index += 1

    # Pad the rest of the row pointer buffer for the current chip.
    while lhs_row_index < row_pointers_size_per_sc:
      lhs_row_pointers_by_sc_id[local_sc_id][
          lhs_row_index
      ] = padded_coo_tensor_index
      lhs_row_index += 1

    # Pad the rest of the COO tensor values for the current chip.
    while padded_coo_tensor_index < padded_coo_tensor_by_sc_id.shape[1]:
      padded_coo_tensor_by_sc_id[local_sc_id][padded_coo_tensor_index] = (
          constants.PADDING_VALUE,
          constants.PADDING_VALUE,
          np.nan,
      )
      padded_coo_tensor_index += 1

  ##############################################################################
  # Step 4: Reassemble the final results.
  ##############################################################################
  lhs_row_pointers = lhs_row_pointers_by_sc_id.reshape(-1)
  padded_coo_tensor = padded_coo_tensor_by_sc_id.reshape(coo_buffer_size)

  lhs_local_sample_ids = np.zeros((coo_buffer_size,), np.int32)
  lhs_local_embedding_ids = np.zeros((coo_buffer_size,), np.int32)
  lhs_gains = np.zeros((coo_buffer_size,), np.float32)

  for coo_index, (row_id, col_id, gain) in enumerate(padded_coo_tensor):
    lhs_local_sample_ids[coo_index] = row_id
    lhs_local_embedding_ids[coo_index] = col_id
    lhs_gains[coo_index] = gain

  return (
      jnp.asarray(lhs_row_pointers),
      jnp.asarray(lhs_local_embedding_ids),
      jnp.asarray(lhs_local_sample_ids),
      jnp.asarray(lhs_gains),
  )


# TODO: b/343986969 - Optimize the size of the tensor.
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
