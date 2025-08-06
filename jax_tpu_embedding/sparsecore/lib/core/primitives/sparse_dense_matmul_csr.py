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
"""Primitive tpu_sparse_dense_matmul_csr."""

import functools
import json

import jax
from jax import core
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
import jax.extend as jex
from jax.interpreters import mlir
from jax.interpreters import xla
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core import constants
import numpy as np

# Define the sparse dense matmul primitive.
tpu_sparse_dense_matmul_csr_primitive = jex.core.Primitive(
    "sparse_dense_matmul_csr"
)


# Define the impl function for the sparse dense matmul primitive.
tpu_sparse_dense_matmul_csr_primitive.def_impl(
    functools.partial(
        xla.apply_primitive, tpu_sparse_dense_matmul_csr_primitive
    )
)


# Define the abstract eval function for the sparse dense matmul primitive.
def _tpu_sparse_dense_matmul_csr_abstract_eval(
    lhs_row_pointers: jnp.ndarray,
    lhs_local_embedding_ids: jnp.ndarray,
    lhs_local_sample_ids: jnp.ndarray,
    lhs_gains: jnp.ndarray,
    num_minibatches_per_sparse_core: np.int32,
    embedding_table: jnp.ndarray,
    *,
    device_batch_size: int,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    sharding_strategy: int = 1,
    quantization_config: tuple[float, float, int] | None = None,
    # NOMUTANTS -- unused param for abstract eval.
    minibatches: bool = False,
):
  """Abstract eval for sdmm_csr."""

  del minibatches
  del num_minibatches_per_sparse_core

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

  if quantization_config is not None:
    quantization_min_value, quantization_max_value, quantization_num_buckets = (
        quantization_config
    )
    if quantization_num_buckets < 2:
      raise ValueError(
          "quantization_num_buckets must be at least 2, got"
          f" {quantization_num_buckets}"
      )

    if quantization_min_value >= quantization_max_value:
      raise ValueError(
          "quantization_min_valuemust be less than quantization_max_value,"
          f" got {quantization_min_value} and {quantization_max_value}"
      )

  return core.ShapedArray(
      (device_batch_size, embedding_table.shape[1]),
      dtype=jnp.float32,
  )


tpu_sparse_dense_matmul_csr_primitive.def_abstract_eval(
    _tpu_sparse_dense_matmul_csr_abstract_eval
)


# Define the mlir lowering rule for the sparse dense matmul primitive.
def _tpu_sparse_dense_matmul_csr_lowering(
    ctx,
    lhs_row_pointers: np.ndarray,
    lhs_local_embedding_ids: np.ndarray,
    lhs_local_sample_ids: np.ndarray,
    lhs_gains: np.ndarray,
    num_minibatches_per_sparse_core: np.int32,
    embedding_table: np.ndarray,
    *,
    device_batch_size: int,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    sharding_strategy: int = 1,
    quantization_config: tuple[float, float, int] | None = None,
    minibatches: bool = False,
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
  # Add quantization params only when enabled
  if quantization_config is not None:
    q_min, q_max, q_buckets = quantization_config
    sdmm_csr_config["quantization_config"] = {
        "min_value": q_min,
        "max_value": q_max,
        "num_buckets": q_buckets,
    }
  backend_config = json.dumps({
      "sparse_dense_matmul_config": sdmm_csr_config,
      "device_type": "DEVICE_TYPE_SPARSECORE",
  })
  operands = (
      [
          lhs_row_pointers,
          lhs_local_embedding_ids,
          lhs_local_sample_ids,
          lhs_gains,
      ]
      + ([num_minibatches_per_sparse_core] if minibatches else [])
      + [
          embedding_table,
          activation_init,
      ]
  )

  if minibatches:  # Buffer contains minibatches
    call_target = "SparseDenseMatmulWithMinibatchingOp"
  else:  # Buffer contains per SC buffer
    # We still have tests that use this format.
    call_target = "SparseDenseMatmulOp"

  return jax.ffi.ffi_lowering(
      call_target,
      result_types=[mlir.aval_to_ir_type(out_aval)],
      api_version=1,
      backend_config=backend_config,
      skip_ffi_layout_processing=True,
  )(
      ctx, *operands
  )  # type: ignore


mlir.register_lowering(
    tpu_sparse_dense_matmul_csr_primitive, _tpu_sparse_dense_matmul_csr_lowering
)
