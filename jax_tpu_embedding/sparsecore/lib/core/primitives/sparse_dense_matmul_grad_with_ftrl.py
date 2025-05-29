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
    "sparse_dense_matmul_grad_with_ftrl")

tpu_sparse_dense_matmul_grad_with_ftrl_primitive.multiple_results = True

tpu_sparse_dense_matmul_grad_with_ftrl_primitive.def_impl(
    functools.partial(dispatch.apply_primitive,
                      tpu_sparse_dense_matmul_grad_with_ftrl_primitive))


def _annotate_sparse_compute_type(op: ir.OpView):
  op.attributes["mhlo.frontend_attributes"] = ir.DictAttr.get(
      {"_xla_compute_type": ir.StringAttr.get("sparse")}
  )
  return op


def _hlo_const(arr: np.ndarray) -> ir.Value:
  """Return an HLO constant from a NumPy array (any rank)."""
  return hlo.constant(
      ir.DenseElementsAttr.get(arr, type=mlir.dtype_to_ir_type(arr.dtype))
  )


def _hlo_f32(x: float, emb_dim: int) -> ir.Value:
  """Return a <1 x emb_dim> f32 constant filled with x."""
  return _hlo_const(
      np.full((1, emb_dim), x, dtype=np.float32)
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
    learning_rate_: np.float32,
    learning_rate_power_: np.float32,
    l1_regularization_strength_: np.float32,
    l2_regularization_strength_: np.float32,
    beta_: np.float32,
    clip_weight_min_: np.float32,
    clip_weight_max_: np.float32,
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
  utils.ensure_dtype(learning_rate_, np.float32, "learning_rate_")
  utils.ensure_dtype(learning_rate_power_, np.float32, "learning_rate_power_")
  utils.ensure_dtype(
      l1_regularization_strength_, np.float32, "l1_regularization_strength_"
  )
  utils.ensure_dtype(
      l2_regularization_strength_, np.float32, "l2_regularization_strength_"
  )
  utils.ensure_dtype(beta_, np.float32, "beta_")
  utils.ensure_dtype(clip_weight_min_, np.float32, "clip_weight_min_")
  utils.ensure_dtype(clip_weight_max_, np.float32, "clip_weight_max_")

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
    lhs_row_pointers: ir.Value,
    lhs_local_embedding_ids: ir.Value,
    lhs_local_sample_ids: ir.Value,
    lhs_gains: ir.Value,
    embedding_table: ir.Value,
    accumulator: ir.Value,
    linear: ir.Value,
    activations_grad: ir.Value,
    learning_rate_: ir.Value,
    learning_rate_power_: ir.Value,
    l1_regularization_strength_: ir.Value,
    l2_regularization_strength_: ir.Value,
    beta_: ir.Value,
    clip_weight_min_: ir.Value,
    clip_weight_max_: ir.Value,
    *,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "ftrl_optimizer_update",
    sharding_strategy: int = 1,
) -> Tuple[ir.Value, ir.Value, ir.Value]:
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

  embedding_table_ranked_type = ir.RankedTensorType(embedding_table.type)
  emb_dim = embedding_table_ranked_type.shape[1]

  f32 = ir.F32Type.get()
  hlo_f32_fn = functools.partial(_hlo_f32, emb_dim=emb_dim)

  optimizer_update = func_dialect.FuncOp(
      optimizer_update_computation_name,
      (
          [
              ir.RankedTensorType.get([1, emb_dim], f32),  # grad_
              ir.RankedTensorType.get(
                  [1, emb_dim], f32
              ),  # embedding_table_ (Ē_o)
              ir.RankedTensorType.get([1, emb_dim], f32),  # accumulator_ (Ā_o)
              ir.RankedTensorType.get([1, emb_dim], f32),  # linear_ (L_o)
              ir.RankedTensorType.get([1, emb_dim], f32),  # lr_param_ (λ)
              ir.RankedTensorType.get([1, emb_dim], f32),  # lr_power_param_ (k)
              ir.RankedTensorType.get([1, emb_dim], f32),  # l1_param_ (γ₁)
              ir.RankedTensorType.get([1, emb_dim], f32),  # l2_param_ (γ₂)
              ir.RankedTensorType.get([1, emb_dim], f32),  # beta_param_ (β)
              ir.RankedTensorType.get(
                  [1, emb_dim], f32
              ),  # clip_weight_min_param_
              ir.RankedTensorType.get(
                  [1, emb_dim], f32
              ),  # clip_weight_max_param_
          ],
          [
              ir.TupleType.get_tuple([
                  ir.RankedTensorType.get(
                      [1, emb_dim], f32
                  ),  # updated_table_ (Ē_n)
                  ir.RankedTensorType.get(
                      [1, emb_dim], f32
                  ),  # updated_accumulator_ (Ā_n)
                  ir.RankedTensorType.get(
                      [1, emb_dim], f32
                  ),  # updated_linear_ (L_n)
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
        l1_param_,  # γ₁
        l2_param_,  # γ₂
        beta_param_,  # β
        clip_weight_min_param_,
        clip_weight_max_param_,
    ) = entry_block.arguments

    # Accumulator update: Ā_n = Ā_o + g²
    grad_square_ = hlo.multiply(grad_, grad_)
    accumulator_new_ = hlo.add(accumulator_, grad_square_)  # Ā_n

    # Power terms: P_α = Ā_α⁻ᵏ
    # neg_lr_power_param_ is -k
    neg_lr_power_param_ = hlo.negate(lr_power_param_)
    # p_o_ = accumulator_ ^ (-k)
    p_o_ = hlo.power(accumulator_, neg_lr_power_param_)
    # p_n_ = accumulator_new_ ^ (-k)
    p_n_ = hlo.power(accumulator_new_, neg_lr_power_param_)
    # p_n_plus_beta_ = P_n + β
    p_n_plus_beta_ = hlo.add(p_n_, beta_param_)

    # Linear term update: L_n = L_o + g - (1/λ) * ΔP * Ē_o
    # where ΔP = P_n - P_o
    delta_p_ = hlo.subtract(p_n_, p_o_)  # ΔP = P_n - P_o

    one_ = hlo_f32_fn(1.0)
    # one_over_lambda_ = 1.0 / λ
    one_over_lambda_ = hlo.divide(one_, lr_param_)
    # (1 / λ) · (P_n + β)
    term1_denominator_ = hlo.multiply(one_over_lambda_, p_n_plus_beta_)

    # delta_p_times_embedding_table_ = ΔP * Ē_o
    delta_p_times_embedding_table_ = hlo.multiply(delta_p_, embedding_table_)

    # term_to_subtract_linear_ = (1/λ) * ΔP * Ē_o
    term_to_subtract_linear_ = hlo.multiply(
        one_over_lambda_, delta_p_times_embedding_table_
    )

    linear_plus_grad_ = hlo.add(linear_, grad_)  # L_o + g
    linear_new_ = hlo.subtract(
        linear_plus_grad_, term_to_subtract_linear_
    )  # L_n

    # Proximal update part for Ē_n
    # Numerator for weight update: min(|L_n|, γ₁) * sign(L_n) - L_n
    # This simplifies to: (sign(L_n) * γ₁ - L_n) if |L_n| > γ₁, else 0.
    # The 'select' op later will handle the 'else 0' part.
    sign_linear_new_ = hlo.sign(linear_new_)
    # sign_linear_new_times_l1_ = sign(L_n) * γ₁
    sign_linear_new_times_l1_ = hlo.multiply(sign_linear_new_, l1_param_)
    # numerator_ = sign(L_n)γ₁ - L_n
    numerator_ = hlo.subtract(sign_linear_new_times_l1_, linear_new_)

    # Denominator for weight update: (1/λ)(P_n + β) + 2γ₂
    # p_n_plus_beta_ = P_n + β
    two_ = hlo_f32_fn(2.0)
    # two_times_l2_ = 2 * γ₂
    two_times_l2_ = hlo.multiply(two_, l2_param_)

    # denominator_ = (P_n + β)/λ + 2γ₂
    denominator_ = hlo.add(term1_denominator_, two_times_l2_)

    # Condition for update: |L_n| > γ₁
    abs_linear_new_ = hlo.abs(linear_new_)
    update_condition_ = hlo.compare(
        abs_linear_new_,
        l1_param_,
        comparison_direction=hlo.ComparisonDirectionAttr.get("GT"),
        compare_type=hlo.ComparisonTypeAttr.get("FLOAT"),
    )

    # Ē_n_unclipped = numerator / denominator (if condition met, else 0)
    weight_update_ = hlo.divide(numerator_, denominator_)

    zero_ = hlo_f32_fn(0.0)
    unclipped_updated_table_ = hlo.select(
        update_condition_, weight_update_, zero_
    )

    # Apply weight clipping
    updated_table_ = hlo.clamp(
        clip_weight_min_param_, unclipped_updated_table_, clip_weight_max_param_
    )

    updated_states_ = hlo.tuple([updated_table_, accumulator_new_, linear_new_])
    func_dialect.ReturnOp([updated_states_])

  table_tuple_op = _annotate_sparse_compute_type(
      hlo.TupleOp([embedding_table, accumulator, linear])
  )

  hyperparams_tuple_op = _annotate_sparse_compute_type(
      hlo.TupleOp([
          learning_rate_,  # This is scalar ir.Value
          learning_rate_power_,
          l1_regularization_strength_,
          l2_regularization_strength_,
          beta_,
          clip_weight_min_,
          clip_weight_max_,
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
          table_tuple_op.result,  # embedding_table, accumulator, linear
          hyperparams_tuple_op.result,  # scalar hyperparams
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
