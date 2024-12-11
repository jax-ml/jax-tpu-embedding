# Copyright 2024 JAX SC Authors.
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

"""Implement support for SparseDenseMatmulWithMinibatchingOp."""

import functools
import json

from jax import core
from jax._src import dispatch
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax.interpreters import mlir
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core import constants
import numpy as np

# Define the sparse dense matmul primitive.
tpu_sparse_dense_matmul_csr_with_mini_batching_primitive = core.Primitive(
    "sparse_dense_matmul_csr_with_mini_batching"
)


# Define the impl function for the sparse dense matmul primitive.
tpu_sparse_dense_matmul_csr_with_mini_batching_primitive.def_impl(
    functools.partial(
        dispatch.apply_primitive,
        tpu_sparse_dense_matmul_csr_with_mini_batching_primitive,
    )
)


# Define the abstract eval function for the sparse dense matmul primitive.
def _tpu_sparse_dense_matmul_csr_with_mini_batching_abstract_eval(
    lhs_row_pointers: jnp.ndarray,
    lhs_local_embedding_ids: jnp.ndarray,
    lhs_local_sample_ids: jnp.ndarray,
    lhs_gains: jnp.ndarray,
    num_minibatches_per_physical_sparse_core: np.int32,
    embedding_table: jnp.ndarray,
    *_,
    device_batch_size: int,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    sharding_strategy: int = 1,
):
  """Abstract eval for sdmm_csr."""
  del num_minibatches_per_physical_sparse_core

  if lhs_row_pointers.dtype != np.int32:
    raise ValueError(
        f"lhs_row_pointers must have type int32, got {lhs_row_pointers.dtype}"
    )

  if lhs_local_sample_ids.dtype != np.int32:
    raise ValueError(
        "lhs_local_sample_ids must have type int32, got"
        f" {lhs_local_sample_ids.dtype}"
    )

  if lhs_local_embedding_ids.dtype != np.int32:
    raise ValueError(
        "lhs_local_embedding_ids must have type uint32, got"
        f" {lhs_local_embedding_ids.dtype}"
    )

  if lhs_gains.dtype != np.float32:
    raise ValueError(f"lhs_gains must have type float32, got {lhs_gains.dtype}")

  if embedding_table.dtype != np.float32:
    raise ValueError(
        f"embedding_table must have type float32, got {embedding_table.dtype}"
    )

  if len(lhs_row_pointers.shape) != 1:
    raise ValueError(
        f"lhs_row_pointers must have rank 1, got {lhs_row_pointers.shape}"
    )

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

  if len(embedding_table.shape) != 2:
    raise ValueError(
        f"embedding_table must have rank 2, got {embedding_table.shape}"
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

  return core.ShapedArray(
      (device_batch_size, embedding_table.shape[1]),
      dtype=jnp.float32,
  )


tpu_sparse_dense_matmul_csr_with_mini_batching_primitive.def_abstract_eval(
    _tpu_sparse_dense_matmul_csr_with_mini_batching_abstract_eval
)


# Define the mlir lowering rule for the sparse dense matmul primitive.
def _tpu_sparse_dense_matmul_csr_with_mini_batching_lowering(
    ctx,
    lhs_row_pointers: np.ndarray,
    lhs_local_embedding_ids: np.ndarray,
    lhs_local_sample_ids: np.ndarray,
    lhs_gains: np.ndarray,
    num_minibatches_per_physical_sparse_core: np.int32,
    embedding_table: np.ndarray,
    *,
    device_batch_size: int,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    sharding_strategy: int = 1,
) -> jnp.ndarray:
  """Lowering for tpu_sparse_dense_matmul_csr."""
  (out_aval,) = ctx.avals_out

  constant_op = hlo.constant(ir.DenseElementsAttr.get(np.float32(0.0)))
  activation_init = hlo.broadcast(
      constant_op,
      mlir.dense_int_array(
          [device_batch_size, embedding_table.type.get_dim_size(1)]  # pylint: disable=attribute-error
      ),
  )

  sdmm_csr_config = {
      "max_ids_per_partition": max_ids_per_partition,
      "max_unique_ids_per_partition": max_unique_ids_per_partition,
      "sharding_strategy": sharding_strategy,
      "pad_value": constants.PADDING_VALUE,
  }
  backend_config = json.dumps({
      "sparse_dense_matmul_config": sdmm_csr_config,
      "device_type": "DEVICE_TYPE_SPARSECORE",
  })

  op = mlir.custom_call(
      "SparseDenseMatmulWithMinibatchingOp",
      result_types=[mlir.aval_to_ir_type(out_aval)],
      operands=[
          lhs_row_pointers,
          lhs_local_embedding_ids,
          lhs_local_sample_ids,
          lhs_gains,
          num_minibatches_per_physical_sparse_core,
          embedding_table,
          activation_init,
      ],
      backend_config=backend_config,
  )
  return op.results


mlir.register_lowering(
    tpu_sparse_dense_matmul_csr_with_mini_batching_primitive,
    _tpu_sparse_dense_matmul_csr_with_mini_batching_lowering,
)
