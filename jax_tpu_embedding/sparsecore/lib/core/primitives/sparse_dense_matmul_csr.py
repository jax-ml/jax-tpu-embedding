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
from jax_tpu_embedding.sparsecore.lib.core.primitives import utils
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
    quantization_config: tuple[float, float, int] | None = None,
    # NOMUTANTS -- unused param for abstract eval.
    enable_minibatching: bool = False,
):
  """Abstract eval for sdmm_csr."""

  del enable_minibatching

  utils.validate_abstract_eval_params(
      lhs_row_pointers,
      lhs_local_embedding_ids,
      lhs_local_sample_ids,
      lhs_gains,
      num_minibatches_per_physical_sparse_core,
      embedding_table,
      activations_grad=np.zeros(
          (device_batch_size, embedding_table.shape[1]), np.float32
      ),  # Not used in the forward pass.
      max_ids_per_partition=max_ids_per_partition,
      max_unique_ids_per_partition=max_unique_ids_per_partition,
      computation_name="fwd-pass",  # Not used in the forward pass.
      sharding_strategy=sharding_strategy,
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
    num_minibatches_per_physical_sparse_core: np.int32,
    embedding_table: np.ndarray,
    *,
    device_batch_size: int,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    sharding_strategy: int = 1,
    quantization_config: tuple[float, float, int] | None = None,
    enable_minibatching: bool = False,
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
      + (
          [num_minibatches_per_physical_sparse_core]
          if enable_minibatching
          else []
      )
      + [
          embedding_table,
          activation_init,
      ]
  )

  if enable_minibatching:  # Buffer contains minibatches
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
