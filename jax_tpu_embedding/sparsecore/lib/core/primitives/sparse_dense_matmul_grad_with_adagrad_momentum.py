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
"""Adagrad with momentum optimizer for sparse dense matmul backward pass.

This implements the Jax primitive for the Adagrad with momentum optimizer for
the
sparse dense matmul backward pass, as a custom call to the
SparseDenseMatmulGradOpWithOptimizerUpdate op. This op takes the preprocessed
input tensors, embedding table, accumulator, momentum buffer, the grad, the
learning rate and momentum hyperparameter as inputs and returns the updated
embedding table, accumulator, and momentum buffer.
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

tpu_sparse_dense_matmul_grad_with_adagrad_momentum_primitive = (
    jex.core.Primitive("sparse_dense_matmul_grad_with_adagrad_momentum")
)
tpu_sparse_dense_matmul_grad_with_adagrad_momentum_primitive.multiple_results = (
    True
)

tpu_sparse_dense_matmul_grad_with_adagrad_momentum_primitive.def_impl(
    functools.partial(
        xla.apply_primitive,
        tpu_sparse_dense_matmul_grad_with_adagrad_momentum_primitive,
    )
)


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
  return _hlo_const(np.full((1, emb_dim), x, dtype=np.float32))


def _tpu_sparse_dense_matmul_grad_with_adagrad_momentum_abstract_eval(
    lhs_row_pointers: np.ndarray,
    lhs_local_embedding_ids: np.ndarray,
    lhs_local_sample_ids: np.ndarray,
    lhs_gains: np.ndarray,
    num_minibatches_per_physical_sparse_core: np.int32,
    embedding_table: np.ndarray,
    accumulator: np.ndarray,
    momentum: np.ndarray,
    activations_grad: np.ndarray,
    learning_rate: np.float32,
    momentum_param: np.float32,
    beta2: np.float32,
    epsilon: np.float32,
    exponent: np.float32,
    use_nesterov: np.bool = np.bool(False),
    *_,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "adagrad_momentum_optimizer_update",
    sharding_strategy: int = 1,
    enable_minibatching: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Abstract eval for sparse_dense_matmul_adagrad_momentum."""
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
  utils.ensure_dtype(momentum, np.float32, "momentum")
  utils.ensure_dtype(learning_rate, np.float32, "learning_rate")
  utils.ensure_dtype(momentum_param, np.float32, "momentum_param")
  utils.ensure_dtype(beta2, np.float32, "beta2")
  utils.ensure_dtype(epsilon, np.float32, "epsilon")
  utils.ensure_dtype(exponent, np.float32, "exponent")
  utils.ensure_dtype(use_nesterov, np.bool, "use_nesterov")

  if (
      embedding_table.shape != accumulator.shape
      or embedding_table.shape != momentum.shape
  ):
    raise ValueError(
        "embedding_table, accumulator and momentum must have identical shapes: "
        f"got {embedding_table.shape}, {accumulator.shape}, {momentum.shape}"
    )
  return embedding_table, accumulator, momentum


tpu_sparse_dense_matmul_grad_with_adagrad_momentum_primitive.def_abstract_eval(
    _tpu_sparse_dense_matmul_grad_with_adagrad_momentum_abstract_eval
)


def _tpu_sparse_dense_matmul_grad_with_adagrad_momentum_lowering(
    ctx: mlir.LoweringRuleContext,
    lhs_row_pointers: mlir.ir.BlockArgument,
    lhs_local_embedding_ids: mlir.ir.BlockArgument,
    lhs_local_sample_ids: mlir.ir.BlockArgument,
    lhs_gains: mlir.ir.BlockArgument,
    num_minibatches_per_physical_sparse_core: mlir.ir.BlockArgument,
    embedding_table: mlir.ir.BlockArgument,
    accumulator: mlir.ir.BlockArgument,
    momentum: mlir.ir.BlockArgument,
    activations_grad: mlir.ir.BlockArgument,
    learning_rate: mlir.ir.BlockArgument,
    momentum_param: mlir.ir.BlockArgument,
    beta2: mlir.ir.BlockArgument,
    epsilon: mlir.ir.BlockArgument,
    exponent: mlir.ir.BlockArgument,
    use_nesterov: mlir.ir.BlockArgument,
    *,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "adagrad_momentum_optimizer_update",
    sharding_strategy: int = 1,
    enable_minibatching: bool = False,
) -> Tuple[ir.Value, ir.Value, ir.Value]:
  """Lowering for sparse_dense_matmul_grad_with_adagrad_momentum."""
  sdmm_config = {
      "max_ids_per_partition": max_ids_per_partition,
      "max_unique_ids_per_partition": max_unique_ids_per_partition,
      "pad_value": constants.PADDING_VALUE,
      "sharding_strategy": sharding_strategy,
      "num_slot_variables": 2,
      "num_hyperparameters": 6,
  }
  backend_config = json.dumps({
      "sparse_dense_matmul_config": sdmm_config,
      "device_type": "DEVICE_TYPE_SPARSECORE",
  })

  optimizer_update_computation_name = computation_name

  emb_dim_size = ir.ShapedType(embedding_table.type).get_dim_size(1)
  optimizer_update = func_dialect.FuncOp(
      computation_name,
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
        momentum_,  # L_o
        lr_param_,  # λ
        momentum_param_,  # k
        beta2_,  # βZ
        epsilon_param_,  # ε
        exponent_param_,
        use_nesterov_flag_,
    ) = entry_block.arguments

    one_ = _hlo_f32(1.0, emb_dim_size)
    neg_one_ = _hlo_f32(-1.0, emb_dim_size)

    # Accumulator
    grad_sq_ = hlo.multiply(grad_, grad_)
    beta2_eq_1_ = hlo.compare(
        beta2_,
        one_,
        comparison_direction=hlo.ComparisonDirectionAttr.get("EQ"),
        compare_type=hlo.ComparisonTypeAttr.get("FLOAT"),
    )
    accum_plus_ = hlo.add(accumulator_, grad_sq_)
    one_minus_beta2_ = hlo.subtract(one_, beta2_)
    scaled_accum_ = hlo.add(
        hlo.multiply(beta2_, accumulator_),
        hlo.multiply(one_minus_beta2_, grad_sq_),
    )
    accum_new_ = hlo.select(beta2_eq_1_, accum_plus_, scaled_accum_)

    # Scaled gradient
    neg_inv_exp_ = hlo.divide(neg_one_, exponent_param_)
    accum_eps_ = hlo.add(accum_new_, epsilon_param_)
    p_new_ = hlo.power(accum_eps_, neg_inv_exp_)
    scaled_grad_ = hlo.multiply(p_new_, grad_)

    # Momentum
    m_new_ = hlo.add(hlo.multiply(momentum_param_, momentum_), scaled_grad_)

    # Delta E
    nesterov_update_ = hlo.add(
        hlo.multiply(momentum_param_, m_new_), scaled_grad_
    )
    update_ = hlo.select(use_nesterov_flag_, nesterov_update_, m_new_)
    lr_update_ = hlo.multiply(lr_param_, update_)

    # Weight update
    w_new_ = hlo.subtract(embedding_table_, lr_update_)

    out_tuple = hlo.tuple([w_new_, accum_new_, m_new_])
    func_dialect.ReturnOp([out_tuple])

  operands = (
      [
          lhs_row_pointers,
          lhs_local_embedding_ids,
          lhs_local_sample_ids,
          lhs_gains,
      ]
      + (
          [num_minibatches_per_physical_sparse_core]
          if enable_minibatching
          else []
      )
      + [
          activations_grad,
          embedding_table,
          # slot variables
          accumulator,
          momentum,
          # hyperparameters
          learning_rate,
          momentum_param,
          beta2,
          epsilon,
          exponent,
          use_nesterov,
      ]
  )

  if enable_minibatching:
    call_target = "SparseDenseMatmulGradOpWithOptimizerUpdateWithMinibatching"
  else:
    call_target = "SparseDenseMatmulGradOpWithOptimizerUpdate"

  custom_call_op = jax.ffi.ffi_lowering(
      call_target,
      result_types=[
          ir.TupleType.get_tuple([
              embedding_table.type,
              accumulator.type,
              momentum.type,
          ])
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
  updated_momentum_op = _annotate_sparse_compute_type(
      hlo.GetTupleElementOp(custom_call_op, 2)
  )

  return (
      updated_table_op.results,
      updated_accumulator_op.results,
      updated_momentum_op.results,
  )


mlir.register_lowering(
    tpu_sparse_dense_matmul_grad_with_adagrad_momentum_primitive,
    _tpu_sparse_dense_matmul_grad_with_adagrad_momentum_lowering,
)
