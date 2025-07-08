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
"""Primitive local_tpu_sparse_dense_matmul_csr."""

import functools

from jax import core
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
import jax.extend as jex
from jax.interpreters import mlir
from jax.interpreters import xla
import jax.numpy as jnp
import numpy as np

# Define the local sparse dense matmul primitive.
tpu_local_sparse_dense_matmul_csr_primitive = jex.core.Primitive(
    "local_sparse_dense_matmul_csr"
)


# Define the impl function for the sparse dense matmul primitive.
tpu_local_sparse_dense_matmul_csr_primitive.def_impl(
    functools.partial(
        xla.apply_primitive, tpu_local_sparse_dense_matmul_csr_primitive
    )
)


# Define the abstract eval function for the sparse dense matmul primitive.
def _tpu_local_sparse_dense_matmul_csr_abstract_eval(
    lhs_local_embedding_ids: jnp.ndarray,
    lhs_local_sample_ids: jnp.ndarray,
    lhs_gains: jnp.ndarray,
    embedding_table: jnp.ndarray,
    *_,
    device_batch_size: int,
):
  """Abstract eval for sdmm_csr."""

  if lhs_local_sample_ids.dtype != np.int32:
    raise ValueError(
        "lhs_local_sample_ids must have type int32, got"
        f" {lhs_local_sample_ids.dtype}"
    )

  if lhs_local_embedding_ids.dtype != np.int64:
    raise ValueError(
        "lhs_local_embedding_ids must have type int64, got"
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


tpu_local_sparse_dense_matmul_csr_primitive.def_abstract_eval(
    _tpu_local_sparse_dense_matmul_csr_abstract_eval
)


# Define the mlir lowering rule for the local sparse dense matmul primitive.
def _tpu_local_sparse_dense_matmul_csr_lowering(
    ctx,
    lhs_local_embedding_ids: np.ndarray,
    lhs_local_sample_ids: np.ndarray,
    lhs_gains: np.ndarray,
    embedding_table: np.ndarray,
    *,
    device_batch_size: int,
) -> jnp.ndarray:
  """Lowering for tpu_sparse_dense_matmul_csr."""
  (out_aval,) = ctx.avals_out
  f32 = ir.F32Type.get()
  i32 = ir.IntegerType.get_signless(32)
  i64 = ir.IntegerType.get_signless(64)

  num_ids = lhs_local_embedding_ids.type.get_dim_size(0)
  feat_width = embedding_table.type.get_dim_size(1)

  constant_op = hlo.constant(ir.DenseElementsAttr.get(np.float32(0.0)))
  activation_init = hlo.broadcast(
      constant_op, mlir.dense_int_array([device_batch_size, feat_width])
  )

  ids_2d_type = ir.RankedTensorType.get([num_ids, 1], i64)
  ids_2d = hlo.reshape(ids_2d_type, lhs_local_embedding_ids)

  gather_dnums = hlo.GatherDimensionNumbers.get(
      # operand dim 0 is sliced, dim 1 is carried through
      collapsed_slice_dims=[0],
      offset_dims=[1],
      start_index_map=[0],
      index_vector_dim=1,
      operand_batching_dims=[],
      start_indices_batching_dims=[],
  )

  slice_sizes = mlir.dense_int_array([1, feat_width])
  embedded_rows = hlo.gather(
      embedding_table,
      ids_2d,
      gather_dnums,
      slice_sizes,
  )

  # Broadcast the gains to the embedding table shape
  gains_bd = hlo.broadcast_in_dim(
      ir.RankedTensorType.get([num_ids, feat_width], f32),
      lhs_gains,
      mlir.dense_int_array([0]),
  )

  # Multiply the gains with the embedded rows
  scaled_rows = hlo.multiply(gains_bd, embedded_rows)
  scaled_rows.owner.attributes["mhlo.frontend_attributes"] = ir.DictAttr.get(
      {"_xla_compute_type": ir.StringAttr.get("sparse")}
  )

  sids_2d_type = ir.RankedTensorType.get([num_ids, 1], i32)
  sids_2d = hlo.reshape(sids_2d_type, lhs_local_sample_ids)

  scatter_dnums = hlo.ScatterDimensionNumbers.get(
      update_window_dims=[1],
      inserted_window_dims=[0],
      scattered_dims_to_operand_dims=[0],
      index_vector_dim=1,
      input_batching_dims=[],
      scatter_indices_batching_dims=[],
  )

  scatter_result = hlo.scatter(
      (mlir.aval_to_ir_type(out_aval),),
      [activation_init],
      sids_2d,
      [scaled_rows],
      scatter_dnums,
  )
  scatter_op = scatter_result.owner

  # Reduction computation for scatter: (x, y) -> x + y
  scalar_f32 = ir.RankedTensorType.get([], f32)
  block = scatter_op.regions[0].blocks.append(scalar_f32, scalar_f32)
  with ir.InsertionPoint(block):
    summed = hlo.add(block.arguments[0], block.arguments[1])
    hlo.return_([summed])

  return scatter_op.results


mlir.register_lowering(
    tpu_local_sparse_dense_matmul_csr_primitive,
    _tpu_local_sparse_dense_matmul_csr_lowering,
)
