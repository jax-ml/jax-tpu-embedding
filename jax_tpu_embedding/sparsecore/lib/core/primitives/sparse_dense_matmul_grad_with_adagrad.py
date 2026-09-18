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
"""Adagrad optimizer for sparse dense matmul backward pass.

This implements the Jax primitive for the Adagrad optimizer for the sparse dense
matmul backward pass, as a custom call to the
SparseDenseMatmulGradOpWithOptimizerUpdate op. This op takes the preprocessed
input tensors, embedding table, accumulator, the grad and the learning rate as
inputs and returns the updated embedding table and accumulator.
"""

import functools
import json
from typing import Sequence
from typing import Tuple

import jax
from jax import core
import jax.extend as jex
from jax.extend.mlir import ir
from jax.extend.mlir.dialects import func as func_dialect
from jax.extend.mlir.dialects import stablehlo as hlo
from jax.interpreters import mlir
from jax.interpreters import xla
from jax_tpu_embedding.sparsecore.lib.core import constants
from jax_tpu_embedding.sparsecore.lib.core.primitives import utils
import numpy as np


tpu_sparse_dense_matmul_grad_with_adagrad_primitive = jex.core.Primitive(
    "sparse_dense_matmul_grad_with_adagrad_primitive",
)

tpu_sparse_dense_matmul_grad_with_adagrad_primitive.multiple_results = True


tpu_sparse_dense_matmul_grad_with_adagrad_primitive.def_impl(
    functools.partial(
        xla.apply_primitive,
        tpu_sparse_dense_matmul_grad_with_adagrad_primitive,
    )
)


def _sum_features(val: ir.Value, result_type: ir.RankedTensorType) -> ir.Value:
  """Sums a [1, N] row across the feature axis, producing a [1] row value."""
  f32 = ir.F32Type.get()
  scalar_type = ir.RankedTensorType.get([], f32)
  zero = hlo.constant(
      ir.DenseElementsAttr.get_splat(scalar_type, ir.FloatAttr.get(f32, 0.0))
  )
  reduce_op = hlo.ReduceOp(
      [result_type], [val], [zero], ir.DenseI64ArrayAttr.get([1])
  )
  reducer = reduce_op.regions[0].blocks.append(scalar_type, scalar_type)
  with ir.InsertionPoint(reducer):
    hlo.return_([hlo.add(reducer.arguments[0], reducer.arguments[1])])
  return reduce_op.result


def _tpu_sparse_dense_matmul_grad_with_adagrad_abstract_eval(
    lhs_row_pointers: core.ShapedArray,
    lhs_local_embedding_ids: core.ShapedArray,
    lhs_local_sample_ids: core.ShapedArray,
    lhs_gains: core.ShapedArray,
    num_minibatches_per_physical_sparse_core: core.ShapedArray,
    embedding_table: core.ShapedArray,
    accumulator: core.ShapedArray,
    activations_grad: core.ShapedArray,
    learning_rate: np.float32,
    *_,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "adagrad_optimizer_update",
    sharding_strategy: int = 1,
    # NOMUTANTS -- unused param for abstract eval.
    enable_minibatching: bool = False,
    min_value: float | None = None,
    max_value: float | None = None,
) -> Tuple[core.ShapedArray, core.ShapedArray]:
  """Abstract eval for sparse_dense_matmul_adagrad."""
  del enable_minibatching

  utils.validate_abstract_eval_params(
      lhs_row_pointers,
      lhs_local_embedding_ids,
      lhs_local_sample_ids,
      lhs_gains,
      num_minibatches_per_physical_sparse_core,
      embedding_table,
      activations_grad,
      max_ids_per_partition,
      max_unique_ids_per_partition,
      computation_name,
      sharding_strategy,
      min_value,
      max_value,
  )

  utils.ensure_dtype(accumulator, np.float32, "accumulator")
  utils.ensure_dtype(learning_rate, np.float32, "learning_rate")

  # The accumulator either mirrors the table element-wise, or holds a single
  # value per table row (row-wise Adagrad).
  if accumulator.shape not in (
      embedding_table.shape,
      embedding_table.shape[:1],
  ):
    raise ValueError(
        "accumulator must have the same shape as embedding_table"
        f" {embedding_table.shape} or be row-wise"
        f" {embedding_table.shape[:1]}, got {accumulator.shape}"
    )

  return embedding_table, accumulator


tpu_sparse_dense_matmul_grad_with_adagrad_primitive.def_abstract_eval(
    _tpu_sparse_dense_matmul_grad_with_adagrad_abstract_eval
)


def _tpu_sparse_dense_matmul_grad_with_adagrad_lowering(
    ctx: mlir.LoweringRuleContext,
    lhs_row_pointers: ir.BlockArgument,
    lhs_local_embedding_ids: ir.BlockArgument,
    lhs_local_sample_ids: ir.BlockArgument,
    lhs_gains: ir.BlockArgument,
    num_minibatches_per_physical_sparse_core: ir.BlockArgument,
    embedding_table: ir.BlockArgument,
    accumulator: ir.BlockArgument,
    activations_grad: ir.BlockArgument,
    learning_rate: ir.BlockArgument,
    *,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "adgrad_optimizer_update",
    sharding_strategy: int = 1,
    enable_minibatching: bool = False,
    min_value: float | None = None,
    max_value: float | None = None,
) -> Tuple[Sequence[ir.Value], Sequence[ir.Value]]:
  """Lowering for sparse_dense_matmul_grad_with_adagrad."""
  sdmm_sgd_config = {
      "max_ids_per_partition": max_ids_per_partition,
      "max_unique_ids_per_partition": max_unique_ids_per_partition,
      "pad_value": constants.PADDING_VALUE,
      "sharding_strategy": sharding_strategy,
      "num_slot_variables": 1,
      "num_hyperparameters": 1,
  }
  backend_config = json.dumps({
      "sparse_dense_matmul_config": sdmm_sgd_config,
      "device_type": "DEVICE_TYPE_SPARSECORE",
  })

  optimizer_update_computation_name = computation_name

  grad_row_type = utils.get_row_type(activations_grad)
  table_row_type = utils.get_row_type(embedding_table)
  accumulator_row_type = utils.get_row_type(accumulator)
  row_wise = accumulator_row_type != grad_row_type

  optimizer_update = func_dialect.FuncOp(
      computation_name,
      (
          [
              grad_row_type,
              table_row_type,
              accumulator_row_type,
              grad_row_type,
          ],
          [
              ir.TupleType.get_tuple([
                  table_row_type,
                  accumulator_row_type,
              ]),
          ],
      ),
      ip=ctx.module_context.ip,
      visibility="private",
  )
  if row_wise:
    # The accumulator row is shaped differently from the table row. Marking the
    # update as a SparseCore computation keeps XLA's main-thread pipeline from
    # normalizing it to a single shape before the optimizer-update decomposer
    # runs.
    optimizer_update.attributes["execution_thread"] = ir.StringAttr.get(
        "sparsecore"
    )

  entry_block = optimizer_update.add_entry_block()
  with ir.InsertionPoint(entry_block):
    grad, table, accumulator_row, learning_rate_row = entry_block.arguments
    grad_squared = hlo.multiply(grad, grad)
    if row_wise:
      # Row-wise Adagrad: a single accumulator value per table row, so the
      # squared gradient is summed across the feature axis and the resulting
      # scale is broadcast back over the row.
      new_accumulator = hlo.add(
          accumulator_row, _sum_features(grad_squared, accumulator_row_type)
      )
      scale = hlo.broadcast_in_dim(
          grad_row_type, hlo.sqrt(new_accumulator), [0]
      )
    else:
      # new_accumulator = accumulator + grad * grad
      new_accumulator = hlo.add(accumulator_row, grad_squared)
      scale = hlo.sqrt(new_accumulator)
    # updated_embedding_table = (learning_rate * grad) / sqrt(new_accumulator)
    updated_embedding_table = hlo.subtract(
        table,
        hlo.divide(hlo.multiply(learning_rate_row, grad), scale),
    )
    updated_embedding_table_clipped = utils.maybe_clip_params(
        updated_embedding_table, min_value, max_value
    )
    updated_embedding_tables = hlo.tuple(
        [updated_embedding_table_clipped, new_accumulator]
    )
    func_dialect.ReturnOp([updated_embedding_tables])

  operands = [
      lhs_row_pointers,
      lhs_local_embedding_ids,
      lhs_local_sample_ids,
      lhs_gains,
  ]

  # b/436897459 - Unify argument order.
  if enable_minibatching:
    call_target = "SparseDenseMatmulGradOptimizerUpdateWithMinibatchingOp"
    operands += [
        num_minibatches_per_physical_sparse_core,
        embedding_table,
        # slot variables
        accumulator,
        # activations grad
        activations_grad,
    ]
  else:
    call_target = "SparseDenseMatmulGradOpWithOptimizerUpdate"
    operands += [
        activations_grad,
        embedding_table,
        # slot variables
        accumulator,
    ]
  operands += [
      # hyperparameters
      learning_rate,
  ]

  op = jax.ffi.ffi_lowering(
      call_target,
      result_types=[
          ir.TupleType.get_tuple([embedding_table.type, accumulator.type])
      ],
      backend_config=backend_config,
      called_computations=[optimizer_update_computation_name],
      skip_ffi_layout_processing=True,
      api_version=1,
  )(ctx, *operands)

  assert isinstance(op[0], ir.Value)
  table_tuple_op = hlo.GetTupleElementOp(op[0], 0)
  table_tuple_op = utils.annotate_sparse_compute_type(table_tuple_op)
  accumulator_tuple_op = hlo.GetTupleElementOp(op[0], 1)
  accumulator_tuple_op = utils.annotate_sparse_compute_type(
      accumulator_tuple_op
  )

  return (
      table_tuple_op.results,
      accumulator_tuple_op.results,
  )


mlir.register_lowering(
    tpu_sparse_dense_matmul_grad_with_adagrad_primitive,
    _tpu_sparse_dense_matmul_grad_with_adagrad_lowering,
)
