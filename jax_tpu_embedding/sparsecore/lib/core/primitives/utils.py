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
"""Utils for sparsecore grad primitives."""

from typing import Any

import numpy as np


def ensure_dtype(check: Any, expected_type: Any, object_name: str):
  if check.dtype != expected_type:
    raise ValueError(
        f"{object_name} must have type {expected_type!r}, got {check.dtype!r}"
    )


def ensure_dim(check: Any, expected_dim: int, object_name: str):
  if len(check.shape) != expected_dim:
    raise ValueError(
        f"{object_name} must have dim {expected_dim!r}, got {check.shape!r}"
    )


def validate_abstract_eval_params(
    lhs_row_pointers: np.ndarray,
    lhs_local_embedding_ids: np.ndarray,
    lhs_local_sample_ids: np.ndarray,
    lhs_gains: np.ndarray,
    num_minibatches_per_physical_sparse_core: np.int32,
    embedding_table: np.ndarray,
    activations_grad: np.ndarray,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str,
    sharding_strategy: int,
):
  """Validate parameters common to all sparsecore grad primitives."""
  ensure_dtype(lhs_row_pointers, np.int32, "lhs_row_pointers")
  ensure_dtype(lhs_local_sample_ids, np.int32, "lhs_local_sample_ids")
  ensure_dtype(lhs_local_embedding_ids, np.int32, "lhs_local_embedding_ids")
  ensure_dtype(lhs_gains, np.float32, "lhs_gains")
  ensure_dtype(
      num_minibatches_per_physical_sparse_core,
      np.int32,
      "num_minibatches_per_physical_sparse_core",
  )
  ensure_dtype(embedding_table, np.float32, "embedding_table")
  ensure_dtype(activations_grad, np.float32, "activations_grad")
  ensure_dim(lhs_row_pointers, 1, "lhs_row_pointers")
  ensure_dim(embedding_table, 2, "embedding_table")
  ensure_dim(
      num_minibatches_per_physical_sparse_core,
      0,
      "num_minibatches_per_physical_sparse_core",
  )
  ensure_dim(activations_grad, 2, "activations_grad")
  if (
      lhs_local_sample_ids.shape != lhs_local_embedding_ids.shape
      or lhs_gains.shape != lhs_local_embedding_ids.shape
      or len(lhs_local_sample_ids.shape) != 1
  ):
    raise ValueError(
        "LHS sample IDs, embedding IDs, and gains must all have "
        f"equal rank 1 shapes, got shapes {lhs_local_sample_ids.shape}, "
        f"{lhs_local_embedding_ids.shape} and {lhs_gains.shape}"
    )
  if embedding_table.shape[-1] != activations_grad.shape[-1]:
    raise ValueError(
        "embedding_table and activations_grad must have equal feature (minor)"
        f" dimensions, got {embedding_table.shape}, {activations_grad.shape}"
    )

  if sharding_strategy != 1:
    raise ValueError(
        f"sharding_strategy must be MOD (1), got {sharding_strategy}"
    )

  if max_ids_per_partition <= 0:
    raise ValueError(
        f"max_ids_per_partition must be positive, got {max_ids_per_partition}"
    )

  if max_unique_ids_per_partition <= 0:
    raise ValueError(
        "max_unique_ids_per_partition must be positive, got"
        f" {max_unique_ids_per_partition}"
    )
  if not computation_name:
    raise ValueError("computation_name must be non-empty")
