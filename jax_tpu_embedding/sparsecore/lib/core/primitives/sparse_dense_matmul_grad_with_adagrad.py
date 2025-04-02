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
from typing import Tuple

from jax import core
from jax._src import dispatch
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func as func_dialect
from jax._src.lib.mlir.dialects import hlo
import jax.extend as jex
from jax.interpreters import mlir
from jax_tpu_embedding.sparsecore.lib.core import constants
from jax_tpu_embedding.sparsecore.lib.core.primitives import utils
import numpy as np

tpu_sparse_dense_matmul_grad_with_adagrad_primitive = jex.core.Primitive(
    "sparse_dense_matmul_grad_with_adagrad_primitive",
)

tpu_sparse_dense_matmul_grad_with_adagrad_primitive.multiple_results = True


tpu_sparse_dense_matmul_grad_with_adagrad_primitive.def_impl(
    functools.partial(
        dispatch.apply_primitive,
        tpu_sparse_dense_matmul_grad_with_adagrad_primitive,
    )
)


def _annotate_sparse_compute_type(op: ir.OpView):
  op.attributes["mhlo.frontend_attributes"] = ir.DictAttr.get(
      {"_xla_compute_type": ir.StringAttr.get("sparse")}
  )
  return op


def _tpu_sparse_dense_matmul_grad_with_adagrad_abstract_eval(
    lhs_row_pointers: np.ndarray,
    lhs_local_embedding_ids: np.ndarray,
    lhs_local_sample_ids: np.ndarray,
    lhs_gains: np.ndarray,
    embedding_table: np.ndarray,
    accumulator: np.ndarray,
    activations_grad: np.ndarray,
    learning_rate: np.float32,
    *_,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "adagrad_optimizer_update",
    sharding_strategy: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
  """Abstract eval for sparse_dense_matmul_adagrad."""
  utils.validate_abstract_eval_params(
      lhs_row_pointers,
      lhs_local_embedding_ids,
      lhs_local_sample_ids,
      lhs_gains,
      embedding_table,
      activations_grad,
      max_ids_per_partition,
      max_unique_ids_per_partition,
      computation_name,
      sharding_strategy,
  )

  utils.ensure_dtype(accumulator, np.float32, "accumulator")
  utils.ensure_dtype(learning_rate, np.float32, "learning_rate")

  if embedding_table.shape != accumulator.shape:
    raise ValueError(
        "embedding_table and accumulator must have equal shapes, got"
        f" {embedding_table.shape} and {accumulator.shape}"
    )

  return embedding_table, accumulator


tpu_sparse_dense_matmul_grad_with_adagrad_primitive.def_abstract_eval(
    _tpu_sparse_dense_matmul_grad_with_adagrad_abstract_eval
)


def _tpu_sparse_dense_matmul_grad_with_adagrad_lowering(
    ctx: mlir.LoweringRuleContext,
    lhs_row_pointers: np.ndarray,
    lhs_local_embedding_ids: np.ndarray,
    lhs_local_sample_ids: np.ndarray,
    lhs_gains: np.ndarray,
    embedding_table: np.ndarray,
    accumulator: np.ndarray,
    activations_grad: np.ndarray,
    learning_rate: np.ndarray,
    *,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "adgrad_optimizer_update",
    sharding_strategy: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
  """Lowering for sparse_dense_matmul_grad_with_adagrad."""
  sdmm_sgd_config = {
      "max_ids_per_partition": max_ids_per_partition,
      "max_unique_ids_per_partition": max_unique_ids_per_partition,
      "pad_value": constants.PADDING_VALUE,
      "sharding_strategy": sharding_strategy,
  }
  backend_config = json.dumps({
      "sparse_dense_matmul_config": sdmm_sgd_config,
      "device_type": "DEVICE_TYPE_SPARSECORE",
  })

  optimizer_update_computation_name = computation_name

  embedding_table_dim_size = embedding_table.type.get_dim_size(1)  # pylint: disable=attribute-error
  optimizer_update = func_dialect.FuncOp(
      computation_name,
      (
          [
              ir.RankedTensorType.get(
                  [1, embedding_table_dim_size],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(
                  [1, embedding_table_dim_size],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(
                  [1, embedding_table_dim_size],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(
                  [1, embedding_table_dim_size],
                  ir.F32Type.get(),
              ),
          ],
          [
              ir.TupleType.get_tuple([
                  ir.RankedTensorType.get(
                      [1, embedding_table_dim_size],
                      ir.F32Type.get(),
                  ),
                  ir.RankedTensorType.get(
                      [1, embedding_table_dim_size],
                      ir.F32Type.get(),
                  ),
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
    updated_embedding_tables = hlo.tuple(
        [updated_embedding_table, new_accumulator]
    )
    func_dialect.ReturnOp([updated_embedding_tables])

  table_tuple_op = hlo.TupleOp([embedding_table, accumulator])
  table_tuple_op = _annotate_sparse_compute_type(table_tuple_op)
  hyperparams_tuple_op = hlo.TupleOp([learning_rate])
  hyperparams_tuple_op = _annotate_sparse_compute_type(hyperparams_tuple_op)

  op = mlir.custom_call(
      "SparseDenseMatmulGradOpWithOptimizerUpdate",
      result_types=[
          ir.TupleType.get_tuple([embedding_table.type, accumulator.type])  # pylint: disable=attribute-error
      ],
      operands=[
          lhs_row_pointers,
          lhs_local_embedding_ids,
          lhs_local_sample_ids,
          lhs_gains,
          activations_grad,
          table_tuple_op.result,
          hyperparams_tuple_op.result,
      ],
      backend_config=backend_config,
      called_computations=[optimizer_update_computation_name],
  )

  table_tuple_op = hlo.GetTupleElementOp(op, 0)
  table_tuple_op = _annotate_sparse_compute_type(table_tuple_op)
  accumulator_tuple_op = hlo.GetTupleElementOp(op, 1)
  accumulator_tuple_op = _annotate_sparse_compute_type(accumulator_tuple_op)

  return (
      table_tuple_op.results,
      accumulator_tuple_op.results,
  )


mlir.register_lowering(
    tpu_sparse_dense_matmul_grad_with_adagrad_primitive,
    _tpu_sparse_dense_matmul_grad_with_adagrad_lowering,
)
