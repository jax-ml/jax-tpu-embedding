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
"""LaProp optimizer for sparse dense matmul backward pass.

This implements the Jax primitive for the LaProp optimizer for the sparse dense
matmul backward pass, as a custom call to the
SparseDenseMatmulGradOpWithOptimizerUpdate op. This op takes the preprocessed
input tensors, embedding table, LaProp hyperparameters (b1, b2, eps), LaProp
states (mu, nu), and the grad as inputs and returns the updated embedding table
and mu, nu values.
"""

import functools
import json
from typing import Tuple

from jax._src import dispatch
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func as func_dialect
from jax._src.lib.mlir.dialects import hlo
import jax.extend as jex
from jax.interpreters import mlir
from jax_tpu_embedding.sparsecore.lib.core import constants
from jax_tpu_embedding.sparsecore.lib.core.primitives import utils
import numpy as np

tpu_sparse_dense_matmul_grad_with_laprop_primitive = jex.core.Primitive(
    "sparse_dense_matmul_grad_with_laprop_primitive",
)

tpu_sparse_dense_matmul_grad_with_laprop_primitive.multiple_results = True


tpu_sparse_dense_matmul_grad_with_laprop_primitive.def_impl(
    functools.partial(
        dispatch.apply_primitive,
        tpu_sparse_dense_matmul_grad_with_laprop_primitive,
    )
)


def _annotate_sparse_compute_type(op: ir.OpView):
  op.attributes["mhlo.frontend_attributes"] = ir.DictAttr.get(
      {"_xla_compute_type": ir.StringAttr.get("sparse")}
  )
  return op


def _tpu_sparse_dense_matmul_grad_with_laprop_abstract_eval(
    lhs_row_pointers: np.ndarray,
    lhs_local_embedding_ids: np.ndarray,
    lhs_local_sample_ids: np.ndarray,
    lhs_gains: np.ndarray,
    embedding_table: np.ndarray,
    mu: np.ndarray,
    nu: np.ndarray,
    activations_grad: np.ndarray,
    b1: float,
    b2: float,
    eps: float,
    *_,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "laprop_optimizer_update",
    sharding_strategy: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Abstract eval for sparse_dense_matmul_laprop."""

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

  utils.ensure_dtype(b1, np.float32, "b1")
  utils.ensure_dtype(b2, np.float32, "b2")
  utils.ensure_dtype(eps, np.float32, "eps")
  utils.ensure_dtype(mu, np.float32, "mu")
  utils.ensure_dtype(nu, np.float32, "nu")

  if embedding_table.shape != mu.shape:
    raise ValueError(
        "embedding_table and mu must have equal shapes, got"
        f" {embedding_table.shape} and {mu.shape}"
    )
  elif embedding_table.shape != nu.shape:
    raise ValueError(
        "embedding_table and nu must have equal shapes, got"
        f" {embedding_table.shape} and {nu.shape}"
    )

  return embedding_table, mu, nu


tpu_sparse_dense_matmul_grad_with_laprop_primitive.def_abstract_eval(
    _tpu_sparse_dense_matmul_grad_with_laprop_abstract_eval
)


def _tpu_sparse_dense_matmul_grad_with_laprop_lowering(
    ctx: mlir.LoweringRuleContext,
    lhs_row_pointers: np.ndarray,
    lhs_local_embedding_ids: np.ndarray,
    lhs_local_sample_ids: np.ndarray,
    lhs_gains: np.ndarray,
    embedding_table: np.ndarray,
    mu: np.ndarray,
    nu: np.ndarray,
    activations_grad: np.ndarray,
    b1: float,
    b2: float,
    eps: float,
    *_,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "laprop_optimizer_update",
    sharding_strategy: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Lowering for sparse_dense_matmul_grad_with_laprop."""

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
  #   func @laprop_optimizer_update(
  #                             %arg0: tensor<1xNxf32>,
  #                             %arg1: tuple<tensor<1xNxf32>>,
  #                             %arg2: tuple<tensor<1xNxf32>>,
  #                             %arg3: tuple<tensor<1xNxf32>>,
  #                             %arg4: tuple<tensor<1xNxf32>>,
  #                             %arg5: tuple<tensor<1xNxf32>>,
  #                             %arg6: tuple<tensor<1xNxf32>>,
  # )
  #   -> tuple<tensor<1xNxf32>, tensor<1xNxf32>, tensor<1xNxf32>>
  # where N is the embedding dimension size.
  # The input arguments are:
  #   %arg0: the gradient vector.
  #   %arg1: the embedding tables before the update.
  #   %arg2: the optimizer states (mu).
  #   %arg3: the optimizer states (nu).
  #   %arg4: the hyperparameters (b1).
  #   %arg5: the hyperparameters (b2).
  #   %arg6: the hyperparameters (eps).
  # The output is a tuple containing the updated embedding tables and optimizer
  # states.

  embedding_table_dim_size = embedding_table.type.get_dim_size(1)

  optimizer_update = func_dialect.FuncOp(
      optimizer_update_computation_name,
      (
          [
              ir.RankedTensorType.get(  # grad
                  [1, embedding_table.type.get_dim_size(1)],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(  # embedding_table
                  [1, embedding_table.type.get_dim_size(1)],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(  # mu
                  [1, embedding_table.type.get_dim_size(1)],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(  # nu
                  [1, embedding_table.type.get_dim_size(1)],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(  # b1
                  [1, embedding_table.type.get_dim_size(1)],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(  # b2
                  [1, embedding_table.type.get_dim_size(1)],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(  # eps
                  [1, embedding_table.type.get_dim_size(1)],
                  ir.F32Type.get(),
              ),
          ],
          [
              ir.TupleType.get_tuple([
                  ir.RankedTensorType.get(  # embedding_table
                      [1, embedding_table_dim_size],
                      ir.F32Type.get(),
                  ),
                  ir.RankedTensorType.get(  # mu
                      [1, embedding_table_dim_size],
                      ir.F32Type.get(),
                  ),
                  ir.RankedTensorType.get(  # nu
                      [1, embedding_table_dim_size],
                      ir.F32Type.get(),
                  ),
              ]),
          ],
      ),
      ip=ctx.module_context.ip,
      visibility="private",
  )

  # This is the row-wise implementation of the optimizer.
  entry_block = optimizer_update.add_entry_block()
  with ir.InsertionPoint(entry_block):
    # Get parameters.
    grad_ = entry_block.arguments[0]
    embedding_table_ = entry_block.arguments[1]
    mu_ = entry_block.arguments[2]
    nu_ = entry_block.arguments[3]
    b1_ = entry_block.arguments[4]
    b2_ = entry_block.arguments[5]
    eps_ = entry_block.arguments[6]

    # update = (b1 * grad) + b2 - eps (using dummy update rule for now).
    # TODO(b/407826659) - Implement the laprop update rule.
    gradient_update = hlo.multiply(
        grad_,
        b1_,
    )
    gradient_update = hlo.add(
        gradient_update,
        b2_,
    )
    gradient_update = hlo.subtract(
        gradient_update,
        eps_,
    )
    # updated_embedding_table = embedding_table - update
    updated_embedding_table = hlo.subtract(embedding_table_, gradient_update)
    updated_tables = hlo.tuple([updated_embedding_table, mu_, nu_])

    # return the updated embedding table, mu, nu
    func_dialect.ReturnOp([updated_tables])

  table_tuple_op = hlo.TupleOp([embedding_table, mu, nu])
  table_tuple_op = _annotate_sparse_compute_type(table_tuple_op)
  hyperparams_tuple_op = hlo.TupleOp([b1, b2, eps])
  hyperparams_tuple_op = _annotate_sparse_compute_type(hyperparams_tuple_op)

  op = mlir.custom_call(
      "SparseDenseMatmulGradOpWithOptimizerUpdate",
      result_types=[
          ir.TupleType.get_tuple([embedding_table.type, mu.type, nu.type])
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
  mu_tuple_op = hlo.GetTupleElementOp(op, 1)
  mu_tuple_op = _annotate_sparse_compute_type(mu_tuple_op)
  nu_tuple_op = hlo.GetTupleElementOp(op, 2)
  nu_tuple_op = _annotate_sparse_compute_type(nu_tuple_op)

  return (
      table_tuple_op.results,
      mu_tuple_op.results,
      nu_tuple_op.results,
  )


mlir.register_lowering(
    tpu_sparse_dense_matmul_grad_with_laprop_primitive,
    _tpu_sparse_dense_matmul_grad_with_laprop_lowering,
)
