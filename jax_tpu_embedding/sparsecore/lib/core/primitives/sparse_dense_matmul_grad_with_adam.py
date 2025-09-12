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
"""Adam optimizer for sparse dense matmul backward pass.

This implements the Jax primitive for the Adam optimizer for the sparse dense
matmul backward pass, as a custom call to the
SparseDenseMatmulGradOpWithOptimizerUpdate op. This op takes the preprocessed
input tensors, embedding table, Adam hyperparameters
(alpha_t, beta_1, beta_2, epsilon_hat), Adam states (momentum, velocity), and
the grad
as inputs and returns the updated embedding table and momentum, velocity values.
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

tpu_sparse_dense_matmul_grad_with_adam_primitive = jex.core.Primitive(
    "sparse_dense_matmul_grad_with_adam_primitive",
)

tpu_sparse_dense_matmul_grad_with_adam_primitive.multiple_results = True


tpu_sparse_dense_matmul_grad_with_adam_primitive.def_impl(
    functools.partial(
        xla.apply_primitive,
        tpu_sparse_dense_matmul_grad_with_adam_primitive,
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


def _tpu_sparse_dense_matmul_grad_with_adam_abstract_eval(
    lhs_row_pointers: np.ndarray,
    lhs_local_embedding_ids: np.ndarray,
    lhs_local_sample_ids: np.ndarray,
    lhs_gains: np.ndarray,
    num_minibatches_per_physical_sparse_core: np.int32,
    embedding_table: np.ndarray,
    velocity: np.ndarray,
    momentum: np.ndarray,
    activations_grad: np.ndarray,
    alpha_t: np.float32,
    beta_1: np.float32,
    beta_2: np.float32,
    epsilon_hat: np.float32,
    *_,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "adam_optimizer_update",
    sharding_strategy: int = 1,
    # NOMUTANTS -- unused param for abstract eval.
    enable_minibatching: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Abstract eval for sparse_dense_matmul_adam."""
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

  utils.ensure_dtype(alpha_t, np.float32, "alpha_t")
  utils.ensure_dtype(beta_1, np.float32, "beta_1")
  utils.ensure_dtype(beta_2, np.float32, "beta_2")
  utils.ensure_dtype(epsilon_hat, np.float32, "epsilon_hat")
  utils.ensure_dtype(velocity, np.float32, "momentum")
  utils.ensure_dtype(momentum, np.float32, "velocity")

  if embedding_table.shape != velocity.shape:
    raise ValueError(
        "embedding_table and velocity must have equal shapes, got"
        f" {embedding_table.shape} and {velocity.shape}"
    )
  elif embedding_table.shape != momentum.shape:
    raise ValueError(
        "embedding_table and momentum must have equal shapes, got"
        f" {embedding_table.shape} and {momentum.shape}"
    )

  return embedding_table, velocity, momentum


tpu_sparse_dense_matmul_grad_with_adam_primitive.def_abstract_eval(
    _tpu_sparse_dense_matmul_grad_with_adam_abstract_eval
)


def _tpu_sparse_dense_matmul_grad_with_adam_lowering(
    ctx: mlir.LoweringRuleContext,
    lhs_row_pointers: np.ndarray,
    lhs_local_embedding_ids: np.ndarray,
    lhs_local_sample_ids: np.ndarray,
    lhs_gains: np.ndarray,
    num_minibatches_per_physical_sparse_core: np.int32,
    embedding_table: np.ndarray,
    momentum: np.ndarray,
    velocity: np.ndarray,
    activations_grad: np.ndarray,
    alpha_t: np.ndarray,
    beta_1: np.ndarray,
    beta_2: np.ndarray,
    epsilon_hat: np.ndarray,
    *_,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "adam_optimizer_update",
    sharding_strategy: int = 1,
    enable_minibatching: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Lowering for sparse_dense_matmul_grad_with_adam."""

  sdmm_adam_config = {
      "max_ids_per_partition": max_ids_per_partition,
      "max_unique_ids_per_partition": max_unique_ids_per_partition,
      "pad_value": constants.PADDING_VALUE,
      "sharding_strategy": sharding_strategy,
      "num_slot_variables": 2,
      "num_hyperparameters": 4,
  }
  backend_config = json.dumps({
      "sparse_dense_matmul_config": sdmm_adam_config,
      "device_type": "DEVICE_TYPE_SPARSECORE",
  })

  optimizer_update_computation_name = computation_name

  # Define the optimizer update function mlir.
  # The expected signature is:
  #   func @adam_optimizer_update(
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
  #   %arg2: the optimizer states (momentum).
  #   %arg3: the optimizer states (velocity).
  #   %arg4: the hyperparameters (alpha_t).
  #   %arg5: the hyperparameters (beta_1).
  #   %arg6: the hyperparameters (beta_2).
  #   %arg7: the hyperparameters (epsilon_hat).
  # The output is a tuple containing the updated embedding tables and optimizer
  # states.

  embedding_table_dim_size = embedding_table.type.get_dim_size(1)
  hlo_f32 = functools.partial(_hlo_f32, emb_dim=embedding_table_dim_size)

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
              ir.RankedTensorType.get(  # momentum
                  [1, embedding_table.type.get_dim_size(1)],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(  # velocity
                  [1, embedding_table.type.get_dim_size(1)],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(  # alpha_t
                  [1, embedding_table.type.get_dim_size(1)],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(  # beta_1
                  [1, embedding_table.type.get_dim_size(1)],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(  # beta_2
                  [1, embedding_table.type.get_dim_size(1)],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(  # epsilon_hat
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
                  ir.RankedTensorType.get(  # momentum
                      [1, embedding_table_dim_size],
                      ir.F32Type.get(),
                  ),
                  ir.RankedTensorType.get(  # velocity
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
    momentum_ = entry_block.arguments[2]
    velocity_ = entry_block.arguments[3]
    alpha_t_ = entry_block.arguments[4]
    beta_1_ = entry_block.arguments[5]
    beta_2_ = entry_block.arguments[6]
    epsilon_hat_ = entry_block.arguments[7]

    grad_square = hlo.multiply(
        grad_,
        grad_,
    )

    # momentum: m = beta_1 * m + (1 - beta_1) * grad
    #             = m + (1 - beta_1) * (grad - m)
    momentum_new = hlo.add(
        momentum_,
        hlo.multiply(
            hlo.subtract(hlo_f32(1.0), beta_1_),
            hlo.subtract(grad_, momentum_),
        ),
    )

    # velocity: v = beta_2 * v + (1 - beta_2) * grad^2
    #             = v + (1 - beta_2) * (grad^2 - v)
    velocity_new = hlo.add(
        velocity_,
        hlo.multiply(
            hlo.subtract(hlo_f32(1.0), beta_2_),
            hlo.subtract(grad_square, velocity_),
        ),
    )

    # theta = theta - alpha_t * m / (sqrt(v) + epsilon_hat)
    update = hlo.divide(
        hlo.multiply(
            alpha_t_,
            momentum_new,
        ),
        hlo.add(
            hlo.sqrt(velocity_new),
            epsilon_hat_,
        ),
    )

    theta = hlo.subtract(embedding_table_, update)

    updated_tables = hlo.tuple([theta, momentum_new, velocity_new])

    # return the updated embedding table, mu, nu
    func_dialect.ReturnOp([updated_tables])

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
        momentum,
        velocity,
        # activations grad
        activations_grad,
    ]
  else:
    call_target = "SparseDenseMatmulGradOpWithOptimizerUpdate"
    operands += [
        activations_grad,
        embedding_table,
        # slot variables
        momentum,
        velocity,
    ]
  operands += [
      # hyperparameters
      alpha_t,
      beta_1,
      beta_2,
      epsilon_hat,
  ]
  op = jax.ffi.ffi_lowering(
      call_target,
      result_types=[
          ir.TupleType.get_tuple(
              [embedding_table.type, momentum.type, velocity.type]
          )
      ],
      backend_config=backend_config,
      called_computations=[optimizer_update_computation_name],
      skip_ffi_layout_processing=True,
      api_version=1,
  )(ctx, *operands)

  table_tuple_op = hlo.GetTupleElementOp(op, 0)
  table_tuple_op = _annotate_sparse_compute_type(table_tuple_op)
  momentum_tuple_op = hlo.GetTupleElementOp(op, 1)
  momentum_tuple_op = _annotate_sparse_compute_type(momentum_tuple_op)
  velocity_tuple_op = hlo.GetTupleElementOp(op, 2)
  velocity_tuple_op = _annotate_sparse_compute_type(velocity_tuple_op)

  return (
      table_tuple_op.results,
      momentum_tuple_op.results,
      velocity_tuple_op.results,
  )


mlir.register_lowering(
    tpu_sparse_dense_matmul_grad_with_adam_primitive,
    _tpu_sparse_dense_matmul_grad_with_adam_lowering,
)
