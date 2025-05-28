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
"""SGD optimizer for sparse dense matmul backward pass with mini batching.

This implements the Jax primitive for the SGD optimizer for the sparse dense
matmul backward pass, as a custom call to the
SparseDenseMatmulGradOptimizerUpdateWithMinibatchingOp op.
This op takes the preprocessed
input tensors, embedding table, accumulator, the grad and the learning rate as
inputs and returns the updated embedding table.
"""

import functools
import json

from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func as func_dialect
from jax._src.lib.mlir.dialects import hlo
import jax.extend as jex
from jax.interpreters import mlir
from jax.interpreters import xla
from jax_tpu_embedding.sparsecore.lib.core import constants
from jax_tpu_embedding.sparsecore.lib.core.primitives import utils
import numpy as np

tpu_sparse_dense_matmul_grad_with_sgd_with_mini_batching_primitive = (
    jex.core.Primitive("sparse_dense_matmul_grad_with_sgd_with_mini_batching")
)

tpu_sparse_dense_matmul_grad_with_sgd_with_mini_batching_primitive.def_impl(
    functools.partial(
        xla.apply_primitive,
        tpu_sparse_dense_matmul_grad_with_sgd_with_mini_batching_primitive,
    )
)


def _annotate_sparse_compute_type(op: ir.OpView):
  op.attributes["mhlo.frontend_attributes"] = ir.DictAttr.get(
      {"_xla_compute_type": ir.StringAttr.get("sparse")}
  )
  return op


def _tpu_sparse_dense_matmul_grad_with_sgd_with_mini_batching_abstract_eval(
    # Operands, passed in as ShapedArray without actual values.
    lhs_row_pointers: np.ndarray,
    lhs_local_embedding_ids: np.ndarray,
    lhs_local_sample_ids: np.ndarray,
    lhs_gains: np.ndarray,
    num_minibatches_per_physical_sparse_core: np.int32,
    embedding_table: np.ndarray,
    activations_grad: np.ndarray,
    learning_rate: np.float32,
    *_,
    # Compile time constants, which cannot be changed from run to run.
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "sgd_optimizer_update",
    sharding_strategy: int = 1,
) -> np.ndarray:
  """Abstract eval for sparse_dense_matmul_sgd."""
  del num_minibatches_per_physical_sparse_core
  utils.ensure_dtype(learning_rate, np.float32, "learning_rate")
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

  return embedding_table


tpu_sparse_dense_matmul_grad_with_sgd_with_mini_batching_primitive.def_abstract_eval(
    _tpu_sparse_dense_matmul_grad_with_sgd_with_mini_batching_abstract_eval
)


def _tpu_sparse_dense_matmul_grad_with_sgd_with_mini_batching_lowering(
    ctx: mlir.LoweringRuleContext,
    lhs_row_pointers: np.ndarray,
    lhs_local_embedding_ids: np.ndarray,
    lhs_local_sample_ids: np.ndarray,
    lhs_gains: np.ndarray,
    num_minibatches_per_physical_sparse_core: np.int32,
    embedding_table: np.ndarray,
    activations_grad: np.ndarray,
    learning_rate: np.ndarray,
    *,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "sgd_optimizer_update",
    sharding_strategy: int = 1,
) -> np.ndarray:
  """Lowering for sdmm_sgd."""
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

  # Define the optimizer update function mlir.
  # The expected signature is:
  #   func @sgd_optimizer_update(%arg0: tensor<1xNxf32>,
  #                             %arg1: tuple<tensor<1xNxf32>>,
  #                             %arg2: tuple<tensor<1xNxf32>>)
  #   -> tuple<tensor<1xNxf32>>
  # where N is the embedding dimension size.
  # The input arguments are:
  #   %arg0: the gradient vector.
  #   %arg1: the embedding tables before the update.
  #   %arg2: the hyperparameters for the optimizer.
  # The output is a tuple containing the updated embedding tables.

  # pylint: disable=attribute-error
  optimizer_update = func_dialect.FuncOp(
      optimizer_update_computation_name,
      (
          [
              ir.RankedTensorType.get(
                  [1, embedding_table.type.get_dim_size(1)],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(
                  [1, embedding_table.type.get_dim_size(1)],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(
                  [1, embedding_table.type.get_dim_size(1)],
                  ir.F32Type.get(),
              ),
          ],
          [
              ir.TupleType.get_tuple([
                  ir.RankedTensorType.get(
                      [1, embedding_table.type.get_dim_size(1)],
                      ir.F32Type.get(),
                  )
              ]),
          ],
      ),
      ip=ctx.module_context.ip,
      visibility="private",
  )
  # pylint: enable=attribute-error

  entry_block = optimizer_update.add_entry_block()
  with ir.InsertionPoint(entry_block):
    # lr * grad
    gradient_update = hlo.multiply(
        entry_block.arguments[0],
        entry_block.arguments[2],
    )
    # updated_embedding_table = embedding_table - lr * grad
    updated_embedding_table = hlo.subtract(
        entry_block.arguments[1], gradient_update
    )
    updated_embedding_tables = hlo.tuple([updated_embedding_table])
    func_dialect.ReturnOp([updated_embedding_tables])

  table_tuple_op = hlo.TupleOp([embedding_table])
  table_tuple_op = _annotate_sparse_compute_type(table_tuple_op)
  hyperparams_tuple_op = hlo.TupleOp([learning_rate])
  hyperparams_tuple_op = _annotate_sparse_compute_type(hyperparams_tuple_op)

  op = mlir.custom_call(
      "SparseDenseMatmulGradOptimizerUpdateWithMinibatchingOp",
      result_types=[ir.TupleType.get_tuple([embedding_table.type])],  # pylint: disable=attribute-error
      operands=[
          lhs_row_pointers,
          lhs_local_embedding_ids,
          lhs_local_sample_ids,
          lhs_gains,
          num_minibatches_per_physical_sparse_core,
          table_tuple_op.result,
          activations_grad,
          hyperparams_tuple_op.result,
      ],
      backend_config=backend_config,
      called_computations=[optimizer_update_computation_name],
  )

  tuple_op = hlo.GetTupleElementOp(op, 0)
  tuple_op = _annotate_sparse_compute_type(tuple_op)
  return tuple_op.results


mlir.register_lowering(
    tpu_sparse_dense_matmul_grad_with_sgd_with_mini_batching_primitive,
    _tpu_sparse_dense_matmul_grad_with_sgd_with_mini_batching_lowering,
)
