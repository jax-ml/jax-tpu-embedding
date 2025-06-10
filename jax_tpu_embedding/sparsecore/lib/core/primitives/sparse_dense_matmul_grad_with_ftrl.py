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
    embedding_table: np.ndarray,
    accumulator: np.ndarray,
    linear: np.ndarray,
    activations_grad: np.ndarray,
    learning_rate: np.float32,
    learning_rate_power: np.float32,
    l1_regularization_strength: np.float32,
    l2_regularization_strength: np.float32,
    beta: np.float32,
    multiply_linear_by_learning_rate: np.bool = np.bool(False),
    *_,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "ftrl_optimizer_update",
    sharding_strategy: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Abstract eval for sparse_dense_matmul_ftrl."""

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
  utils.ensure_dtype(linear, np.float32, "linear")
  utils.ensure_dtype(learning_rate, np.float32, "learning_rate_")
  utils.ensure_dtype(learning_rate_power, np.float32, "learning_rate_power_")
  utils.ensure_dtype(
      l1_regularization_strength, np.float32, "l1_regularization_strength_"
  )
  utils.ensure_dtype(
      l2_regularization_strength, np.float32, "l2_regularization_strength_"
  )
  utils.ensure_dtype(beta, np.float32, "beta_")
  utils.ensure_dtype(
      multiply_linear_by_learning_rate,
      np.bool,
      "multiply_linear_by_learning_rate_",
  )

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
    lhs_row_pointers: np.ndarray,
    lhs_local_embedding_ids: np.ndarray,
    lhs_local_sample_ids: np.ndarray,
    lhs_gains: np.ndarray,
    embedding_table: np.ndarray,
    accumulator: np.ndarray,
    linear: np.ndarray,
    activations_grad: np.ndarray,
    learning_rate_: np.ndarray,
    learning_rate_power_: np.ndarray,
    l1_regularization_strength_: np.ndarray,
    l2_regularization_strength_: np.ndarray,
    beta_: np.ndarray,
    multiply_linear_by_learning_rate_: np.ndarray,
    *,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "ftrl_optimizer_update",
    sharding_strategy: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Lowering for sparse_dense_matmul_grad_with_ftrl."""

  sdmm_ftrl_config = {
      "max_ids_per_partition": max_ids_per_partition,
      "max_unique_ids_per_partition": max_unique_ids_per_partition,
      "pad_value": constants.PADDING_VALUE,
      "sharding_strategy": sharding_strategy,
  }
  backend_config = json.dumps({
      "sparse_dense_matmul_config": sdmm_ftrl_config,
      "device_type": "DEVICE_TYPE_SPARSECORE",
  })

  optimizer_update_computation_name = computation_name

  emb_dim_size = embedding_table.type.get_dim_size(1)  # pylint: disable=attribute-error

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
              ir.RankedTensorType.get(
                  [1, emb_dim_size], ir.IntegerType.get_signless(1)
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
        grad_,  # g
        embedding_table_,  # Ē_o
        accumulator_,  # Ā_o
        linear_,  # L_o
        lr_param_,  # λ
        lr_power_param_,  # k
        l1_param_,  # γ_1
        l2_param_,  # γ_2
        beta_param_,  # βZ
        mul_by_lr_flag_,
    ) = entry_block.arguments

    two_ = _hlo_f32(2.0, emb_dim_size)
    zero_ = _hlo_f32(0.0, emb_dim_size)

    # Accumulator
    accumulator_new_ = hlo.add(accumulator_, hlo.multiply(grad_, grad_))

    # Power‑law terms
    neg_lr_pow_ = hlo.negate(lr_power_param_)
    p_old_ = hlo.power(accumulator_, neg_lr_pow_)
    p_new_ = hlo.power(accumulator_new_, neg_lr_pow_)
    delta_p_ = hlo.subtract(p_new_, p_old_)

    # Linear State
    linear_new_ = hlo.select(
        mul_by_lr_flag_,
        # multiply_linear_by_learning_rate = True
        hlo.subtract(
            hlo.add(linear_, hlo.multiply(lr_param_, grad_)),
            hlo.multiply(delta_p_, embedding_table_),
        ),
        # multiply_linear_by_learning_rate = False
        hlo.subtract(
            hlo.add(linear_, grad_),
            hlo.multiply(hlo.divide(delta_p_, lr_param_), embedding_table_),
        ),
    )

    # Numerator
    scale_ = hlo.select(
        mul_by_lr_flag_,
        lr_param_,
        _hlo_const(np.ones((1, emb_dim_size), np.float32)),
    )
    l1_scaled = hlo.multiply(l1_param_, scale_)
    numerator_ = hlo.select(
        hlo.compare(
            l1_param_,
            zero_,
            comparison_direction=hlo.ComparisonDirectionAttr.get("EQ"),
            compare_type=hlo.ComparisonTypeAttr.get("FLOAT"),
        ),
        hlo.negate(linear_new_),
        hlo.subtract(
            hlo.clamp(hlo.negate(l1_scaled), linear_new_, l1_scaled),
            linear_new_,
        ),
    )

    # Denominator
    denominator_ = hlo.select(
        mul_by_lr_flag_,
        # multiply_linear_by_learning_rate = True
        hlo.add(
            hlo.add(
                p_new_, hlo.multiply(two_, hlo.multiply(lr_param_, l2_param_))
            ),
            beta_param_,
        ),
        # multiply_linear_by_learning_rate = False
        hlo.add(
            hlo.divide(hlo.add(p_new_, beta_param_), lr_param_),
            hlo.multiply(two_, l2_param_),
        ),
    )

    # Weight update
    w_new = hlo.divide(numerator_, denominator_)

    func_dialect.ReturnOp([hlo.tuple([w_new, accumulator_new_, linear_new_])])

  table_tuple_op = _annotate_sparse_compute_type(
      hlo.TupleOp([embedding_table, accumulator, linear])
  )

  hyperparams_tuple_op = _annotate_sparse_compute_type(
      hlo.TupleOp([
          learning_rate_,
          learning_rate_power_,
          l1_regularization_strength_,
          l2_regularization_strength_,
          beta_,
          multiply_linear_by_learning_rate_,
      ])
  )

  custom_call_op = mlir.custom_call(
      "SparseDenseMatmulGradOpWithOptimizerUpdate",
      result_types=[
          ir.TupleType.get_tuple(
              [embedding_table.type, accumulator.type, linear.type]
          )
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
