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

import jax
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func as func_dialect
from jax._src.lib.mlir.dialects import hlo
import jax.extend as jex
from jax.interpreters import mlir
from jax.interpreters import xla
from jax_tpu_embedding.sparsecore.lib.core import constants
from jax_tpu_embedding.sparsecore.lib.core.primitives import utils
import numpy as np

tpu_sparse_dense_matmul_grad_with_laprop_primitive = jex.core.Primitive(
    "sparse_dense_matmul_grad_with_laprop_primitive",
)

tpu_sparse_dense_matmul_grad_with_laprop_primitive.multiple_results = True


tpu_sparse_dense_matmul_grad_with_laprop_primitive.def_impl(
    functools.partial(
        xla.apply_primitive,
        tpu_sparse_dense_matmul_grad_with_laprop_primitive,
    )
)


def _annotate_sparse_compute_type(op: ir.OpView):
  op.attributes["mhlo.frontend_attributes"] = ir.DictAttr.get(
      {"_xla_compute_type": ir.StringAttr.get("sparse")}
  )
  return op


def _hlo_const(x: np.ndarray) -> ir.Value:
  return hlo.constant(
      ir.DenseElementsAttr.get(x, type=mlir.dtype_to_ir_type(x.dtype))
  )


def _hlo_f32(x: float, emb_dim: int):
  return _hlo_const(
      np.array(emb_dim * [x], dtype=np.float32).reshape((1, emb_dim))
  )


def _tpu_sparse_dense_matmul_grad_with_laprop_abstract_eval(
    lhs_row_pointers: np.ndarray,
    lhs_local_embedding_ids: np.ndarray,
    lhs_local_sample_ids: np.ndarray,
    lhs_gains: np.ndarray,
    num_minibatches_per_physical_sparse_core: np.int32,
    embedding_table: np.ndarray,
    mu: np.ndarray,
    nu: np.ndarray,
    activations_grad: np.ndarray,
    learning_rate: np.float32,
    b1: np.float32,
    decay_rate: np.float32,
    eps: np.float32,
    *_,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "laprop_optimizer_update",
    sharding_strategy: int = 1,
    # NOMUTANTS -- unused param for abstract eval.
    enable_minibatching: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Abstract eval for sparse_dense_matmul_laprop."""
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
  )

  utils.ensure_dtype(learning_rate, np.float32, "learning_rate")
  utils.ensure_dtype(b1, np.float32, "b1")
  utils.ensure_dtype(decay_rate, np.float32, "decay_rate")
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
    lhs_row_pointers: mlir.ir.BlockArgument,
    lhs_local_embedding_ids: mlir.ir.BlockArgument,
    lhs_local_sample_ids: mlir.ir.BlockArgument,
    lhs_gains: mlir.ir.BlockArgument,
    num_minibatches_per_physical_sparse_core: mlir.ir.BlockArgument,
    embedding_table: mlir.ir.BlockArgument,
    mu: mlir.ir.BlockArgument,
    nu: mlir.ir.BlockArgument,
    activations_grad: mlir.ir.BlockArgument,
    learning_rate: mlir.ir.BlockArgument,
    b1: mlir.ir.BlockArgument,
    decay_rate: mlir.ir.BlockArgument,
    eps: mlir.ir.BlockArgument,
    *_,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "laprop_optimizer_update",
    sharding_strategy: int = 1,
    enable_minibatching: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Lowering for sparse_dense_matmul_grad_with_laprop."""

  sdmm_sgd_config = {
      "max_ids_per_partition": max_ids_per_partition,
      "max_unique_ids_per_partition": max_unique_ids_per_partition,
      "pad_value": constants.PADDING_VALUE,
      "sharding_strategy": sharding_strategy,
      "num_slot_variables": 2,
      "num_hyperparameters": 4,
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
  #                             %arg7: tuple<tensor<1xNxf32>>,
  # )
  #   -> tuple<tensor<1xNxf32>, tensor<1xNxf32>, tensor<1xNxf32>>
  # where N is the embedding dimension size.
  # The input arguments are:
  #   %arg0: the gradient vector.
  #   %arg1: the embedding tables before the update.
  #   %arg2: the optimizer states (mu).
  #   %arg3: the optimizer states (nu).
  #   %arg4: the hyperparameters (learning_rate).
  #   %arg5: the hyperparameters (b1).
  #   %arg6: the hyperparameters (b2).
  #   %arg7: the hyperparameters (eps).
  # The output is a tuple containing the updated embedding tables and optimizer
  # states.

  embedding_table_dim_size = embedding_table.type.maybe_downcast().get_dim_size(
      1
  )
  hlo_f32 = functools.partial(_hlo_f32, emb_dim=embedding_table_dim_size)

  optimizer_update = func_dialect.FuncOp(
      optimizer_update_computation_name,
      (
          [
              ir.RankedTensorType.get(  # grad
                  [1, embedding_table.type.maybe_downcast().get_dim_size(1)],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(  # embedding_table
                  [1, embedding_table.type.maybe_downcast().get_dim_size(1)],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(  # mu
                  [1, embedding_table.type.maybe_downcast().get_dim_size(1)],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(  # nu
                  [1, embedding_table.type.maybe_downcast().get_dim_size(1)],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(  # learning_rate
                  [1, embedding_table.type.maybe_downcast().get_dim_size(1)],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(  # b1
                  [1, embedding_table.type.maybe_downcast().get_dim_size(1)],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(  # decay_rate
                  [1, embedding_table.type.maybe_downcast().get_dim_size(1)],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(  # eps
                  [1, embedding_table.type.maybe_downcast().get_dim_size(1)],
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
    lr_ = entry_block.arguments[4]
    b1_ = entry_block.arguments[5]
    decay_rate_ = entry_block.arguments[6]
    eps_ = entry_block.arguments[7]

    # grad_square = grad*grad + eps
    grad_square = hlo.multiply(
        grad_,
        grad_,
    )
    grad_square = hlo.add(
        grad_square,
        eps_,
    )

    # nu_new = decay*nu + (1-decay)*grad_square
    nu_new = hlo.multiply(
        nu_,
        decay_rate_,
    )
    nu_new = hlo.add(
        nu_new,
        hlo.multiply(hlo.subtract(hlo_f32(1.0), decay_rate_), grad_square),
    )

    # update = grad / sqrt(nu_new)
    update = hlo.divide(grad_, hlo.sqrt(nu_new))

    # TODO(b/407826659): Add RMS clipping.

    # momentum: update = b_1*mu + ( (1 - b_1^2)^0.5 ) * update

    momentum_term_1 = hlo.multiply(b1_, mu_)
    momentum_term_2 = hlo.power(
        hlo.subtract(hlo_f32(1.0), hlo.power(b1_, hlo_f32(2))),
        hlo_f32(0.5),
    )

    update = hlo.add(momentum_term_1, hlo.multiply(momentum_term_2, update))

    mu_new = update

    # updated_table = embedding_table - learning_rate * update
    updated_embedding_table = hlo.subtract(
        embedding_table_, hlo.multiply(lr_, update)
    )

    updated_tables = hlo.tuple([updated_embedding_table, mu_new, nu_new])

    # return the updated embedding table, mu, nu
    func_dialect.ReturnOp([updated_tables])

  # b/436897459 - Unify argument order.
  operands = [
      lhs_row_pointers,
      lhs_local_embedding_ids,
      lhs_local_sample_ids,
      lhs_gains,
  ]
  if enable_minibatching:
    call_target = "SparseDenseMatmulGradOptimizerUpdateWithMinibatchingOp"
    operands += [
        num_minibatches_per_physical_sparse_core,
        embedding_table,
        # slot variables
        mu,
        nu,
        # activations grad
        activations_grad,
    ]
  else:
    call_target = "SparseDenseMatmulGradOpWithOptimizerUpdate"
    operands += [
        activations_grad,
        embedding_table,
        # slot variables
        mu,
        nu,
    ]
  operands += [
      # hyperparameters
      learning_rate,
      b1,
      decay_rate,
      eps,
  ]
  op = jax.ffi.ffi_lowering(
      call_target,
      result_types=[
          ir.TupleType.get_tuple([embedding_table.type, mu.type, nu.type])
      ],
      backend_config=backend_config,
      called_computations=[optimizer_update_computation_name],
      skip_ffi_layout_processing=True,
      api_version=1,
  )(ctx, *operands)

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
