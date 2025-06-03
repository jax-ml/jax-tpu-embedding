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
beta, clip_weight_min, clip_weight_max), FTRL states (accumulator, linear),
and the grad as inputs and returns the updated embedding table and FTRL states.
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


tpu_sparse_dense_matmul_grad_with_ftrl_primitive = jex.core.Primitive(
    "sparse_dense_matmul_grad_with_ftrl"
)

tpu_sparse_dense_matmul_grad_with_ftrl_primitive.multiple_results = True

tpu_sparse_dense_matmul_grad_with_ftrl_primitive.def_impl(
    functools.partial(
        dispatch.apply_primitive,
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
    clip_weight_min: np.float32,
    clip_weight_max: np.float32,
    weight_decay_factor: np.float32,
    multiply_weight_decay_factor_by_learning_rate: np.bool = np.bool(False),
    multiply_linear_by_learning_rate: np.bool = np.bool(False),
    allow_zero_accumulator: np.bool = np.bool(False),
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
  utils.ensure_dtype(clip_weight_min, np.float32, "clip_weight_min_")
  utils.ensure_dtype(clip_weight_max, np.float32, "clip_weight_max_")
  utils.ensure_dtype(weight_decay_factor, np.float32, "weight_decay_factor_")
  utils.ensure_dtype(
      multiply_weight_decay_factor_by_learning_rate,
      np.bool,
      "multiply_weight_decay_factor_by_learning_rate_",
  )
  utils.ensure_dtype(
      multiply_linear_by_learning_rate,
      np.bool,
      "multiply_linear_by_learning_rate_",
  )
  utils.ensure_dtype(allow_zero_accumulator, np.bool, "allow_zero_accumulator_")

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
    clip_weight_min_: np.ndarray,
    clip_weight_max_: np.ndarray,
    weight_decay_factor_: np.ndarray,
    multiply_weight_decay_factor_by_learning_rate_: np.ndarray,
    multiply_linear_by_learning_rate_: np.ndarray,
    allow_zero_accumulator_: np.ndarray,
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
              ir.RankedTensorType.get(
                  [1, emb_dim_size], ir.IntegerType.get_signless(1)
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
        clip_weight_min_param_,
        clip_weight_max_param_,
        weight_decay_factor_param_,
        weight_decay_factor_by_lr_flag_,
        mul_by_lr_flag_,
        allow_zero_accumulator_flag_,
    ) = entry_block.arguments

    epsilon_ = _hlo_f32(1.0e-30, emb_dim_size)
    two_ = _hlo_f32(2.0, emb_dim_size)
    zero_ = _hlo_f32(0.0, emb_dim_size)

    accumulator_safe_ = hlo.select(
        allow_zero_accumulator_flag_,
        accumulator_,
        hlo.add(accumulator_, epsilon_),
    )

    # Weight decay
    wd_scale_ = hlo.select(
        weight_decay_factor_by_lr_flag_,
        hlo.multiply(weight_decay_factor_param_, lr_param_),
        weight_decay_factor_param_,
    )
    wd_term_ = hlo.multiply(wd_scale_, embedding_table_)
    grad_with_wd_ = hlo.add(grad_, wd_term_)

    # Accumulator update: Ā_n = Ā_o + g ^ 2
    grad_square_ = hlo.multiply(grad_with_wd_, grad_with_wd_)
    accumulator_new_ = hlo.add(accumulator_safe_, grad_square_)

    # Power terms: P_α = Ā_α ^ (-k)
    # neg_lr_power_param_ is -k
    neg_lr_power_param_ = hlo.negate(lr_power_param_)
    # p_o_ = accumulator_ ^ (-k)
    p_o_ = hlo.power(accumulator_safe_, neg_lr_power_param_)
    # p_n_ = accumulator_new_ ^ (-k)
    p_n_ = hlo.power(accumulator_new_, neg_lr_power_param_)
    # p_n_plus_beta_ = P_n + β
    p_n_plus_beta_ = hlo.add(p_n_, beta_param_)
    # delta_p_ = P_n - P_o
    delta_p_ = hlo.subtract(p_n_, p_o_)

    # Linear term update: L_n = L_o + g - (1/λ) * ΔP * Ē_o
    # candidate_no_mlr = L_o + g - (1/λ) * ΔP * Ē_o
    # candidate_mlr = L_o + λ * g - ΔP * Ē_o
    term_to_sub_no_mlr_ = hlo.multiply(
        hlo.divide(delta_p_, lr_param_), embedding_table_
    )
    cand_no_mlr_ = hlo.subtract(
        hlo.add(linear_, grad_with_wd_), term_to_sub_no_mlr_
    )
    cand_mlr_ = hlo.subtract(
        hlo.add(linear_, hlo.multiply(lr_param_, grad_with_wd_)),
        hlo.multiply(delta_p_, embedding_table_),
    )
    linear_scaled_new_ = hlo.select(mul_by_lr_flag_, cand_mlr_, cand_no_mlr_)

    # For weight-update we need unscaled L_n
    linear_unscaled_new_ = hlo.select(
        mul_by_lr_flag_,
        hlo.divide(linear_scaled_new_, lr_param_),
        linear_scaled_new_,
    )

    l1_effective_ = hlo.select(
        mul_by_lr_flag_, hlo.multiply(l1_param_, lr_param_), l1_param_
    )

    sign_l_ = hlo.sign(linear_scaled_new_)
    numerator_ = hlo.subtract(
        hlo.multiply(sign_l_, l1_effective_), linear_scaled_new_
    )
    denom_ = hlo.add(
        hlo.divide(p_n_plus_beta_, lr_param_), hlo.multiply(two_, l2_param_)
    )

    abs_l_ = hlo.abs(linear_scaled_new_)
    cond_update_ = hlo.compare(
        abs_l_,
        l1_effective_,
        comparison_direction=hlo.ComparisonDirectionAttr.get("GT"),
        compare_type=hlo.ComparisonTypeAttr.get("FLOAT"),
    )
    w_unclipped_ = hlo.select(
        cond_update_, hlo.divide(numerator_, denom_), zero_
    )
    w_new_ = hlo.clamp(
        clip_weight_min_param_, w_unclipped_, clip_weight_max_param_
    )

    out_tuple = hlo.tuple([w_new_, accumulator_new_, linear_unscaled_new_])
    func_dialect.ReturnOp([out_tuple])

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
          clip_weight_min_,
          clip_weight_max_,
          weight_decay_factor_,
          multiply_weight_decay_factor_by_learning_rate_,
          multiply_linear_by_learning_rate_,
          allow_zero_accumulator_,
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
