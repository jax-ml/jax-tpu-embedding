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


def _tpu_sparse_dense_matmul_grad_with_adagrad_abstract_eval(
    lhs_row_pointers: core.ShapedArray,
    lhs_local_embedding_ids: core.ShapedArray,
    lhs_local_sample_ids: core.ShapedArray,
    lhs_gains: core.ShapedArray,
    num_minibatches_per_physical_sparse_core: core.ShapedArray,
    embedding_table: core.ShapedArray,
    *rest: core.ShapedArray,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "adagrad_optimizer_update",
    sharding_strategy: int = 1,
    # NOMUTANTS -- unused param for abstract eval.
    enable_minibatching: bool = False,
    min_value: float | None = None,
    max_value: float | None = None,
) -> Tuple[core.ShapedArray, ...]:
  """Abstract eval for sparse_dense_matmul_adagrad."""
  del enable_minibatching

  if len(rest) == 4 and embedding_table.dtype == np.int16:
    row_scale, accumulator, activations_grad, learning_rate = rest
  else:
    row_scale = None
    accumulator, activations_grad, learning_rate = rest[:3]

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

  if embedding_table.shape != accumulator.shape:
    raise ValueError(
        "embedding_table and accumulator must have equal shapes, got"
        f" {embedding_table.shape} and {accumulator.shape}"
    )

  if row_scale is not None:
    utils.ensure_dtype(embedding_table, np.int16, "embedding_table")
    utils.ensure_dtype(row_scale, np.float32, "row_scale")
    utils.ensure_dim(row_scale, 1, "row_scale")
    if row_scale.shape[0] != embedding_table.shape[0]:
      raise ValueError(
          "row_scale and embedding_table must have equal row counts, got"
          f" {row_scale.shape} and {embedding_table.shape}"
      )
    return embedding_table, row_scale, accumulator
  elif embedding_table.dtype == np.int16:
    raise ValueError(
        "row_scale must be provided when embedding_table has dtype int16"
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
    *rest: ir.BlockArgument,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "adgrad_optimizer_update",
    sharding_strategy: int = 1,
    enable_minibatching: bool = False,
    min_value: float | None = None,
    max_value: float | None = None,
) -> Tuple[Sequence[ir.Value], ...]:
  """Lowering for sparse_dense_matmul_grad_with_adagrad."""
  if len(rest) == 4:
    row_scale, accumulator, activations_grad, learning_rate = rest
  else:
    row_scale = None
    accumulator, activations_grad, learning_rate = rest[:3]

  sdmm_sgd_config: dict[str, object] = {
      "max_ids_per_partition": max_ids_per_partition,
      "max_unique_ids_per_partition": max_unique_ids_per_partition,
      "pad_value": constants.PADDING_VALUE,
      "sharding_strategy": sharding_strategy,
      "num_slot_variables": 2 if row_scale is not None else 1,
      "num_hyperparameters": 1,
  }
  if row_scale is not None:
    sdmm_sgd_config["storage_format"] = {
        "element_type": "S16",
        "scale_mode": "SCALE_MODE_ROW",
        "scale_policy": "SCALE_UPDATE_DYNAMIC_ABSMAX",
        "min_rowmax": 1e-12,
        "scale_slot_index": 2,
    }
  backend_config = json.dumps({
      "sparse_dense_matmul_config": sdmm_sgd_config,
      "device_type": "DEVICE_TYPE_SPARSECORE",
  })

  optimizer_update_computation_name = computation_name

  row_type = utils.get_row_type(embedding_table)

  optimizer_update = func_dialect.FuncOp(
      computation_name,
      (
          [
              row_type,
              row_type,
              row_type,
              row_type,
          ],
          [
              ir.TupleType.get_tuple([
                  row_type,
                  row_type,
              ]),
          ],
      ),
      ip=ctx.module_context.ip,
      visibility="private",
  )

  entry_block = optimizer_update.add_entry_block()
  with ir.InsertionPoint(entry_block):
    # new_accumulator = accumulator + grad * grad
    grad_squared = hlo.multiply(
        entry_block.arguments[0],
        entry_block.arguments[0],
    )
    new_accumulator = hlo.add(
        entry_block.arguments[2],
        grad_squared,
    )
    # updated_embedding_table = (learning_rate * grad) / sqrt(new_accumulator)
    updated_embedding_table = hlo.subtract(
        entry_block.arguments[1],
        hlo.divide(
            hlo.multiply(
                entry_block.arguments[3],
                entry_block.arguments[0],
            ),
            hlo.sqrt(new_accumulator),
        ),
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

  slot_vars = [accumulator] + ([row_scale] if row_scale is not None else [])
  # b/436897459 - Unify argument order.
  if enable_minibatching:
    call_target = "SparseDenseMatmulGradOptimizerUpdateWithMinibatchingOp"
    operands += (
        [
            num_minibatches_per_physical_sparse_core,
            embedding_table,
        ]
        + slot_vars
        + [
            # activations grad
            activations_grad,
        ]
    )
  else:
    call_target = "SparseDenseMatmulGradOpWithOptimizerUpdate"
    operands += [
        activations_grad,
        embedding_table,
    ] + slot_vars
  operands += [
      # hyperparameters
      learning_rate,
  ]

  tuple_types = [embedding_table.type, accumulator.type] + (
      [row_scale.type] if row_scale is not None else []
  )
  op = jax.ffi.ffi_lowering(
      call_target,
      result_types=[ir.TupleType.get_tuple(tuple_types)],
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

  if row_scale is not None:
    row_scale_tuple_op = hlo.GetTupleElementOp(op[0], 2)
    row_scale_tuple_op = utils.annotate_sparse_compute_type(row_scale_tuple_op)
    return (
        table_tuple_op.results,
        row_scale_tuple_op.results,
        accumulator_tuple_op.results,
    )

  return (
      table_tuple_op.results,
      accumulator_tuple_op.results,
  )


mlir.register_lowering(
    tpu_sparse_dense_matmul_grad_with_adagrad_primitive,
    _tpu_sparse_dense_matmul_grad_with_adagrad_lowering,
)
