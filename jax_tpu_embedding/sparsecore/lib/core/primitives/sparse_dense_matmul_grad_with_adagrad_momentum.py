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
    embedding_table: np.ndarray,
    accumulator: np.ndarray,
    momentum: np.ndarray,
    activations_grad: np.ndarray,
    learning_rate: np.float32,
    momentum_param: np.float32,
    beta2_param: np.float32,
    epsilon: np.float32,
    k_power: np.float32,
    *_,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "adagrad_momentum_optimizer_update",
    sharding_strategy: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Abstract eval for sparse_dense_matmul_adagrad_momentum."""
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
  utils.ensure_dtype(momentum, np.float32, "momentum")
  utils.ensure_dtype(learning_rate, np.float32, "learning_rate")
  utils.ensure_dtype(momentum_param, np.float32, "momentum_param")
  utils.ensure_dtype(beta2_param, np.float32, "beta2_param")
  utils.ensure_dtype(epsilon, np.float32, "epsilon")
  utils.ensure_dtype(k_power, np.float32, "k_power")

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
    lhs_row_pointers: ir.Value,
    lhs_local_embedding_ids: ir.Value,
    lhs_local_sample_ids: ir.Value,
    lhs_gains: ir.Value,
    embedding_table: ir.Value,
    accumulator: ir.Value,
    momentum_buffer: ir.Value,
    activations_grad: ir.Value,
    learning_rate: ir.Value,
    momentum_param: ir.Value,
    beta2_param: ir.Value,
    epsilon: ir.Value,
    k_power: ir.Value,
    *,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "adagrad_momentum_optimizer_update",
    sharding_strategy: int = 1,
) -> Tuple[ir.Value, ir.Value, ir.Value]:
  """Lowering for sparse_dense_matmul_grad_with_adagrad_momentum."""
  sdmm_config = {
      "max_ids_per_partition": max_ids_per_partition,
      "max_unique_ids_per_partition": max_unique_ids_per_partition,
      "pad_value": constants.PADDING_VALUE,
      "sharding_strategy": sharding_strategy,
  }
  backend_config = json.dumps({
      "sparse_dense_matmul_config": sdmm_config,
      "device_type": "DEVICE_TYPE_SPARSECORE",
  })

  optimizer_update_computation_name = computation_name

  embedding_table_dim_size = ir.ShapedType(embedding_table.type).get_dim_size(1)
  f32 = ir.F32Type.get()
  emb_shape = [1, embedding_table_dim_size]

  optimizer_update = func_dialect.FuncOp(
      computation_name,
      (
          [
              ir.RankedTensorType.get(emb_shape, f32),  # grad
              ir.RankedTensorType.get(emb_shape, f32),  # embedding_table
              ir.RankedTensorType.get(emb_shape, f32),  # accumulator
              ir.RankedTensorType.get(emb_shape, f32),  # momentum_buffer
              ir.RankedTensorType.get(
                  emb_shape, f32
              ),
              ir.RankedTensorType.get(
                  emb_shape, f32
              ),
              ir.RankedTensorType.get(
                  emb_shape, f32
              ),
              ir.RankedTensorType.get(
                  emb_shape, f32
              ),
              ir.RankedTensorType.get(
                  emb_shape, f32
              ),
          ],
          [
              ir.TupleType.get_tuple([
                  ir.RankedTensorType.get(emb_shape, f32),
                  ir.RankedTensorType.get(
                      emb_shape, f32
                  ),
                  ir.RankedTensorType.get(emb_shape, f32),
              ]),
          ],
      ),
      ip=ctx.module_context.ip,
      visibility="private",
  )

  entry_block = optimizer_update.add_entry_block()
  with ir.InsertionPoint(entry_block):
    grad_, table_, acc_, mom_buf_, lr_, mom_, beta2_, eps_, k_pow_ = (
        entry_block.arguments
    )

    # accumulator update: acc_new = beta2 * acc_old + (1 - beta2) * grad^2
    one_val = _hlo_f32(
        1.0, embedding_table_dim_size
    )
    one_minus_beta2 = hlo.subtract(one_val, beta2_)
    beta2_times_acc = hlo.multiply(beta2_, acc_)
    grad_sq = hlo.multiply(grad_, grad_)
    one_minus_beta2_times_grad_sq = hlo.multiply(one_minus_beta2, grad_sq)
    acc_new = hlo.add(beta2_times_acc, one_minus_beta2_times_grad_sq)

    # compute adjusted gradient: (lr * grad) / (acc_new + eps) ^ (-1/k)
    lr_times_grad = hlo.multiply(lr_, grad_)
    acc_new_plus_eps = hlo.add(acc_new, eps_)
    neg_one_val = _hlo_f32(-1.0, embedding_table_dim_size)
    neg_one_over_k = hlo.divide(neg_one_val, k_pow_)
    power_val = hlo.power(acc_new_plus_eps, neg_one_over_k)
    adjusted = hlo.multiply(
        lr_times_grad,
        power_val,
    )

    # momentum update: m_new = (mom_param * m_old) + adjusted_grad
    mom_scaled = hlo.multiply(mom_, mom_buf_)
    mom_new = hlo.add(mom_scaled, adjusted)

    # parameter update: table_new = table - m_new
    table_new = hlo.subtract(table_, mom_new)
    updated_states = hlo.tuple([table_new, acc_new, mom_new])
    func_dialect.ReturnOp([updated_states])

  # Prepare inputs for custom call.
  table_tuple_op = hlo.TupleOp([embedding_table, accumulator, momentum_buffer])
  table_tuple_op = _annotate_sparse_compute_type(table_tuple_op)

  # Hyperparams tuple should contain the original scalar ir.Values
  hyperparams_tuple_op = hlo.TupleOp(
      [learning_rate, momentum_param, beta2_param, epsilon, k_power]
  )
  hyperparams_tuple_op = _annotate_sparse_compute_type(hyperparams_tuple_op)

  custom_call = mlir.custom_call(
      "SparseDenseMatmulGradOpWithOptimizerUpdate",
      result_types=[
          ir.TupleType.get_tuple([
              embedding_table.type,
              accumulator.type,
              momentum_buffer.type,
          ])
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

  updated_table_op = hlo.GetTupleElementOp(custom_call, 0)
  updated_table_op = _annotate_sparse_compute_type(updated_table_op)
  updated_acc_op = hlo.GetTupleElementOp(custom_call, 1)
  updated_acc_op = _annotate_sparse_compute_type(updated_acc_op)
  updated_mom_op = hlo.GetTupleElementOp(custom_call, 2)
  updated_mom_op = _annotate_sparse_compute_type(updated_mom_op)

  return (
      updated_table_op.results,
      updated_acc_op.results,
      updated_mom_op.results,
  )


mlir.register_lowering(
    tpu_sparse_dense_matmul_grad_with_adagrad_momentum_primitive,
    _tpu_sparse_dense_matmul_grad_with_adagrad_momentum_lowering,
)
