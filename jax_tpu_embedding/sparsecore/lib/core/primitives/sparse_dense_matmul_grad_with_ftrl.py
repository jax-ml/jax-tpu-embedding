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
"""FTRL optimizer for sparse-dense matmul backward pass.

This implements the Jax primitive for the FTRL optimizer for the sparse dense
matmul backward pass, as a custom call to the
SparseDenseMatmulGradOpWithOptimizerUpdate op. This op takes the preprocessed
input tensors, embedding table, FTRL hyperparameters (learning_rate,
learning_rate_power, l1_regularization_strength, l2_regularization_strength,
beta, multiply_linear_by_learning_rate), FTRL states
(accumulator, linear), and the grad as inputs and returns the updated
embedding table and FTRL states.
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


tpu_sparse_dense_matmul_grad_with_ftrl_primitive = jex.core.Primitive(
    "sparse_dense_matmul_grad_with_ftrl"
)

tpu_sparse_dense_matmul_grad_with_ftrl_primitive.multiple_results = True

tpu_sparse_dense_matmul_grad_with_ftrl_primitive.def_impl(
    functools.partial(
        xla.apply_primitive,
        tpu_sparse_dense_matmul_grad_with_ftrl_primitive,
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


def _tpu_sparse_dense_matmul_grad_with_ftrl_abstract_eval(
    lhs_row_pointers: np.ndarray,
    lhs_local_embedding_ids: np.ndarray,
    lhs_local_sample_ids: np.ndarray,
    lhs_gains: np.ndarray,
    num_minibatches_per_physical_sparse_core: np.int32,
    embedding_table: np.ndarray,
    accumulator: np.ndarray,
    linear: np.ndarray,
    activations_grad: np.ndarray,
    learning_rate: np.float32,
    learning_rate_power: np.float32,
    l1_regularization_strength: np.float32,
    l2_regularization_strength: np.float32,
    beta: np.float32,
    *,
    multiply_linear_by_learning_rate: bool,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "ftrl_optimizer_update",
    sharding_strategy: int = 1,
    # NOMUTANTS -- unused param for abstract eval.
    enable_minibatching: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Abstract eval for sparse_dense_matmul_ftrl."""
  del multiply_linear_by_learning_rate
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

  utils.ensure_dtype(accumulator, np.float32, "accumulator")
  utils.ensure_dtype(linear, np.float32, "linear")
  utils.ensure_dtype(learning_rate, np.float32, "learning_rate")
  utils.ensure_dtype(learning_rate_power, np.float32, "learning_rate_power")
  utils.ensure_dtype(
      l1_regularization_strength, np.float32, "l1_regularization_strength"
  )
  utils.ensure_dtype(
      l2_regularization_strength, np.float32, "l2_regularization_strength"
  )
  utils.ensure_dtype(beta, np.float32, "beta")

  if (
      embedding_table.shape != accumulator.shape
      or embedding_table.shape != linear.shape
  ):
    raise ValueError(
        "embedding_table, accumulator and linear must have "
        f"identical shapes: got {embedding_table.shape}, "
        f"{accumulator.shape}, {linear.shape}"
    )

  return embedding_table, accumulator, linear


tpu_sparse_dense_matmul_grad_with_ftrl_primitive.def_abstract_eval(
    _tpu_sparse_dense_matmul_grad_with_ftrl_abstract_eval
)


def _tpu_sparse_dense_matmul_grad_with_ftrl_lowering(
    ctx: mlir.LoweringRuleContext,
    lhs_row_pointers: mlir.ir.BlockArgument,
    lhs_local_embedding_ids: mlir.ir.BlockArgument,
    lhs_local_sample_ids: mlir.ir.BlockArgument,
    lhs_gains: mlir.ir.BlockArgument,
    num_minibatches_per_physical_sparse_core: mlir.ir.BlockArgument,
    embedding_table: mlir.ir.BlockArgument,
    accumulator: mlir.ir.BlockArgument,
    linear: mlir.ir.BlockArgument,
    activations_grad: mlir.ir.BlockArgument,
    learning_rate: mlir.ir.BlockArgument,
    learning_rate_power: mlir.ir.BlockArgument,
    l1_regularization_strength: mlir.ir.BlockArgument,
    l2_regularization_strength: mlir.ir.BlockArgument,
    beta: mlir.ir.BlockArgument,
    *,
    multiply_linear_by_learning_rate: bool = False,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "ftrl_optimizer_update",
    sharding_strategy: int = 1,
    enable_minibatching: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Lowering for sparse_dense_matmul_grad_with_ftrl."""

  sdmm_ftrl_config = {
      "max_ids_per_partition": max_ids_per_partition,
      "max_unique_ids_per_partition": max_unique_ids_per_partition,
      "pad_value": constants.PADDING_VALUE,
      "sharding_strategy": sharding_strategy,
      "num_slot_variables": 2,
      "num_hyperparameters": 5,
  }
  backend_config = json.dumps({
      "sparse_dense_matmul_config": sdmm_ftrl_config,
      "device_type": "DEVICE_TYPE_SPARSECORE",
  })

  optimizer_update_computation_name = computation_name

  emb_dim_size = embedding_table.type.maybe_downcast().get_dim_size(1)

  optimizer_update = func_dialect.FuncOp(
      optimizer_update_computation_name,
      (
          [
              ir.RankedTensorType.get(
                  [1, emb_dim_size],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(
                  [1, emb_dim_size],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(
                  [1, emb_dim_size],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(
                  [1, emb_dim_size],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(
                  [1, emb_dim_size],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(
                  [1, emb_dim_size],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(
                  [1, emb_dim_size],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(
                  [1, emb_dim_size],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(
                  [1, emb_dim_size],
                  ir.F32Type.get(),
              ),
          ],
          [
              ir.TupleType.get_tuple([
                  ir.RankedTensorType.get(
                      [1, emb_dim_size],
                      ir.F32Type.get(),
                  ),
                  ir.RankedTensorType.get(
                      [1, emb_dim_size],
                      ir.F32Type.get(),
                  ),
                  ir.RankedTensorType.get(
                      [1, emb_dim_size],
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
    (
        grad,  # g
        embedding_table_arg,  # Ē_o
        accumulator_arg,  # Ā_o
        linear_arg,  # L_o
        lr_param,  # λ
        lr_power_param,  # k
        l1_param,  # γ_1
        l2_param,  # γ_2
        beta_param,  # βZ
    ) = entry_block.arguments

    two = _hlo_f32(2.0, emb_dim_size)
    zero = _hlo_f32(0.0, emb_dim_size)

    # Accumulator
    accumulator_new = hlo.add(accumulator_arg, hlo.multiply(grad, grad))

    # Power‑law terms
    neg_lr_pow = hlo.negate(lr_power_param)
    p_old = hlo.power(accumulator_arg, neg_lr_pow)
    p_new = hlo.power(accumulator_new, neg_lr_pow)
    delta_p = hlo.subtract(p_new, p_old)

    # Linear State
    if multiply_linear_by_learning_rate:
      linear_new = hlo.subtract(
          hlo.add(linear_arg, hlo.multiply(lr_param, grad)),
          hlo.multiply(delta_p, embedding_table_arg),
      )
    else:
      linear_new = hlo.subtract(
          hlo.add(linear_arg, grad),
          hlo.multiply(hlo.divide(delta_p, lr_param), embedding_table_arg),
      )

    # # Numerator
    if multiply_linear_by_learning_rate:
      scale = lr_param
    else:
      scale = _hlo_const(np.ones((1, emb_dim_size), np.float32))

    l1_scaled = hlo.multiply(l1_param, scale)
    numerator = hlo.select(
        hlo.compare(
            l1_param,
            zero,
            comparison_direction=hlo.ComparisonDirectionAttr.get("EQ"),
            compare_type=hlo.ComparisonTypeAttr.get("FLOAT"),
        ),
        hlo.negate(linear_new),
        hlo.subtract(
            hlo.clamp(hlo.negate(l1_scaled), linear_new, l1_scaled),
            linear_new,
        ),
    )

    # Denominator
    if multiply_linear_by_learning_rate:
      denominator = hlo.add(
          hlo.add(p_new, hlo.multiply(two, hlo.multiply(lr_param, l2_param))),
          beta_param,
      )
    else:
      denominator = hlo.add(
          hlo.divide(hlo.add(p_new, beta_param), lr_param),
          hlo.multiply(two, l2_param),
      )

    # Weight update
    w_new = hlo.divide(numerator, denominator)

    func_dialect.ReturnOp([hlo.tuple([w_new, accumulator_new, linear_new])])

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
        linear,
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
        linear,
    ]
  operands += [
      # hyperparameters
      learning_rate,
      learning_rate_power,
      l1_regularization_strength,
      l2_regularization_strength,
      beta,
  ]
  custom_call_op = jax.ffi.ffi_lowering(
      call_target,
      result_types=[
          ir.TupleType.get_tuple(
              [embedding_table.type, accumulator.type, linear.type]
          )
      ],
      backend_config=backend_config,
      called_computations=[optimizer_update_computation_name],
      skip_ffi_layout_processing=True,
      api_version=1,
  )(ctx, *operands)

  updated_table_op = _annotate_sparse_compute_type(
      hlo.GetTupleElementOp(custom_call_op, 0)
  )
  updated_accumulator_op = _annotate_sparse_compute_type(
      hlo.GetTupleElementOp(custom_call_op, 1)
  )
  updated_linear_op = _annotate_sparse_compute_type(
      hlo.GetTupleElementOp(custom_call_op, 2)
  )

  return (
      updated_table_op.result,
      updated_accumulator_op.result,
      updated_linear_op.result,
  )


mlir.register_lowering(
    tpu_sparse_dense_matmul_grad_with_ftrl_primitive,
    _tpu_sparse_dense_matmul_grad_with_ftrl_lowering,
)
