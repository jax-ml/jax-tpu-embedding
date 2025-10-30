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
"""Primitive local_tpu_sparse_dense_matmul."""

import functools
import json

import jax
from jax import core
import jax.extend as jex
from jax.extend.mlir import ir
from jax.extend.mlir.dialects import stablehlo as hlo
from jax.interpreters import mlir
from jax.interpreters import xla
import jax.numpy as jnp
import numpy as np

# Define the local sparse dense matmul primitive.
tpu_local_sparse_dense_matmul_primitive = jex.core.Primitive(
    "local_sparse_dense_matmul"
)


# Define the impl function for the sparse dense matmul primitive.
tpu_local_sparse_dense_matmul_primitive.def_impl(
    functools.partial(
        xla.apply_primitive, tpu_local_sparse_dense_matmul_primitive
    )
)


# Define the abstract eval function for the sparse dense matmul primitive.
def _tpu_local_sparse_dense_matmul_abstract_eval(
    lhs_local_embedding_ids: jnp.ndarray,
    lhs_local_sample_ids: jnp.ndarray,
    lhs_gains: jnp.ndarray,
    embedding_table: jnp.ndarray,
    *_,
    device_batch_size: int,
):
  """Abstract eval for sdmm."""

  if lhs_local_sample_ids.dtype != np.int32:
    raise ValueError(
        "lhs_local_sample_ids must have type int32, got"
        f" {lhs_local_sample_ids.dtype}"
    )

  if lhs_local_embedding_ids.dtype != np.int32:
    raise ValueError(
        "lhs_local_embedding_ids must have type int32, got"
        f" {lhs_local_embedding_ids.dtype}"
    )

  if lhs_gains.dtype != np.float32:
    raise ValueError(f"lhs_gains must have type float32, got {lhs_gains.dtype}")

  if embedding_table.dtype != np.float32:
    raise ValueError(
        f"embedding_table must have type float32, got {embedding_table.dtype}"
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

  return core.ShapedArray(
      (device_batch_size, embedding_table.shape[1]),
      dtype=jnp.float32,
  )


tpu_local_sparse_dense_matmul_primitive.def_abstract_eval(
    _tpu_local_sparse_dense_matmul_abstract_eval
)


# Define the mlir lowering rule for the local sparse dense matmul primitive.
def _tpu_local_sparse_dense_matmul_lowering(
    ctx,
    lhs_local_embedding_ids: np.ndarray,
    lhs_local_sample_ids: np.ndarray,
    lhs_gains: np.ndarray,
    embedding_table: np.ndarray,
    *,
    device_batch_size: int,
) -> jnp.ndarray:
  """Lowering for tpu_sparse_dense_matmul."""
  (out_aval,) = ctx.avals_out

  constant_op = hlo.constant(ir.DenseElementsAttr.get(np.float32(0.0)))
  activation_init = hlo.broadcast(
      constant_op,
      mlir.dense_int_array(
          [device_batch_size, embedding_table.type.get_dim_size(1)]  # pylint: disable=attribute-error
      ),
  )

  backend_config = json.dumps({
      "device_type": "DEVICE_TYPE_SPARSECORE",
  })

  operands = [
      embedding_table,
      lhs_local_embedding_ids,
      lhs_local_sample_ids,
      lhs_gains,
      activation_init,
  ]

  return jax.ffi.ffi_lowering(
      "SparseDenseMatmulLocalOp",
      result_types=[mlir.aval_to_ir_type(out_aval)],
      api_version=1,
      backend_config=backend_config,
      skip_ffi_layout_processing=True,
  )(
      ctx, *operands
  )  # type: ignore


mlir.register_lowering(
    tpu_local_sparse_dense_matmul_primitive,
    _tpu_local_sparse_dense_matmul_lowering,
)
