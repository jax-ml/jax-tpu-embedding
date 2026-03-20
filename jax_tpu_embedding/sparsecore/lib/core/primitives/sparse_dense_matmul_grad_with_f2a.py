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
"""F2A optimizer for sparse-dense matmul backward pass.

This implements the Jax primitive for the F2A optimizer for the sparse dense
matmul backward pass, as a custom call to the
SparseDenseMatmulGradOpWithOptimizerUpdate op. This op takes the preprocessed
input tensors, embedding table, F2A hyperparameters (rho, learning_rate,
l1_regularization_strength, l2_regularization_strength, max_lr_multiplier)  ,
F2A states (accumulator, local_step), and the grad as inputs and returns the
updated embedding table and F2A states.
"""

import json
from typing import Tuple

import jax
import jax.extend as jex
from jax.extend.mlir import ir
from jax.extend.mlir.dialects import func as func_dialect
from jax.extend.mlir.dialects import stablehlo as hlo
from jax.interpreters import mlir
from jax.interpreters import xla
from jax_tpu_embedding.sparsecore.lib.core import constants
from jax_tpu_embedding.sparsecore.lib.core.primitives import utils
import numpy as np

tpu_sparse_dense_matmul_grad_with_f2a_primitive = jex.core.Primitive(
    "tpu_sparse_dense_matmul_grad_with_f2a"
)
tpu_sparse_dense_matmul_grad_with_f2a_primitive.multiple_results = True

tpu_sparse_dense_matmul_grad_with_f2a_primitive.def_impl(
    lambda *args, **kwargs: xla.apply_primitive(
        tpu_sparse_dense_matmul_grad_with_f2a_primitive, *args, **kwargs
    )
)


def _annotate_sparse_compute_type(op):
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


def _tpu_sparse_dense_matmul_grad_with_f2a_abstract_eval(
    lhs_row_pointers: np.ndarray,
    lhs_local_embedding_ids: np.ndarray,
    lhs_local_sample_ids: np.ndarray,
    lhs_gains: np.ndarray,
    num_minibatches_per_physical_sparse_core: np.int32,
    embedding_table: np.ndarray,
    accumulator: np.ndarray,
    local_step: np.ndarray,
    activations_grad: np.ndarray,
    # Hyperparameters
    learning_rate: np.float32,
    rho: np.float32,
    l1_regularization_strength: np.float32,
    l2_regularization_strength: np.float32,
    max_lr_multiplier: np.float32,
    global_step: np.float32,
    *_,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "f2a_optimizer_update",
    sharding_strategy: int = 1,
    # NOMUTANTS -- Unused param for abstract eval.
    enable_minibatching: bool = False,
    min_value: float | None = None,
    max_value: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Abstract evaluation for tpu_sparse_dense_matmul_grad_with_f2a."""
  del enable_minibatching
  utils.validate_abstract_eval_params(
      lhs_row_pointers,
      lhs_local_embedding_ids,
      lhs_local_sample_ids,
      lhs_gains,
      num_minibatches_per_physical_sparse_core,
      embedding_table,
      activations_grad,
      max_ids_per_partition=max_ids_per_partition,
      max_unique_ids_per_partition=max_unique_ids_per_partition,
      computation_name=computation_name,
      sharding_strategy=sharding_strategy,
      min_value=min_value,
      max_value=max_value,
  )

  utils.ensure_dtype(embedding_table, np.float32, "embedding_table")
  utils.ensure_dtype(accumulator, np.float32, "accumulator")
  utils.ensure_dtype(local_step, np.float32, "local_step")
  utils.ensure_dtype(learning_rate, np.float32, "learning_rate")
  utils.ensure_dtype(rho, np.float32, "rho")
  utils.ensure_dtype(
      l1_regularization_strength, np.float32, "l1_regularization_strength"
  )
  utils.ensure_dtype(
      l2_regularization_strength, np.float32, "l2_regularization_strength"
  )
  utils.ensure_dtype(max_lr_multiplier, np.float32, "max_lr_multiplier")
  utils.ensure_dtype(global_step, np.float32, "global_step")

  if accumulator.shape != embedding_table.shape:
    raise ValueError(
        f"accumulator shape {accumulator.shape} must match embedding_table"
        f" shape {embedding_table.shape}"
    )
  if local_step.shape != embedding_table.shape:
    raise ValueError(
        f"local_step shape {local_step.shape} must match embedding_table shape"
        f" {embedding_table.shape}"
    )

  return embedding_table, accumulator, local_step


tpu_sparse_dense_matmul_grad_with_f2a_primitive.def_abstract_eval(
    _tpu_sparse_dense_matmul_grad_with_f2a_abstract_eval
)


def _tpu_sparse_dense_matmul_grad_with_f2a_lowering(
    ctx: mlir.LoweringRuleContext,
    lhs_row_pointers: mlir.ir.BlockArgument,
    lhs_local_embedding_ids: mlir.ir.BlockArgument,
    lhs_local_sample_ids: mlir.ir.BlockArgument,
    lhs_gains: mlir.ir.BlockArgument,
    num_minibatches_per_physical_sparse_core: np.int32,
    embedding_table: mlir.ir.BlockArgument,
    accumulator: mlir.ir.BlockArgument,
    local_step: mlir.ir.BlockArgument,
    activations_grad: mlir.ir.BlockArgument,
    learning_rate: mlir.ir.BlockArgument,
    rho: mlir.ir.BlockArgument,
    l1_regularization_strength: mlir.ir.BlockArgument,
    l2_regularization_strength: mlir.ir.BlockArgument,
    max_lr_multiplier: mlir.ir.BlockArgument,
    global_step: mlir.ir.BlockArgument,
    *,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "f2a_optimizer_update",
    sharding_strategy: int = 1,
    enable_minibatching: bool = False,
    min_value: float | None = None,
    max_value: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Lowering for tpu_sparse_dense_matmul_grad_with_f2a."""

  sdmm_f2a_config = {
      "max_ids_per_partition": max_ids_per_partition,
      "max_unique_ids_per_partition": max_unique_ids_per_partition,
      "pad_value": constants.PADDING_VALUE,
      "sharding_strategy": sharding_strategy,
      "num_slot_variables": 2,
      "num_hyperparameters": 6,
  }
  backend_config = json.dumps({
      "sparse_dense_matmul_config": sdmm_f2a_config,
      "device_type": "DEVICE_TYPE_SPARSECORE",
  })

  embedding_table_dim_size = ir.RankedTensorType(
      embedding_table.type
  ).get_dim_size(1)

  optimizer_update_computation_name = computation_name

  optimizer_update = func_dialect.FuncOp(
      optimizer_update_computation_name,
      ir.FunctionType.get(
          [
              # Grad.
              ir.RankedTensorType.get(
                  [1, embedding_table_dim_size], ir.F32Type.get()
              ),
              # Param.
              ir.RankedTensorType.get(
                  [1, embedding_table_dim_size], ir.F32Type.get()
              ),
              # Accum.
              ir.RankedTensorType.get(
                  [1, embedding_table_dim_size], ir.F32Type.get()
              ),
              # Local step.
              ir.RankedTensorType.get(
                  [1, embedding_table_dim_size], ir.F32Type.get()
              ),
              # Learning rate.
              ir.RankedTensorType.get(
                  [1, embedding_table_dim_size], ir.F32Type.get()
              ),
              # Rho.
              ir.RankedTensorType.get(
                  [1, embedding_table_dim_size], ir.F32Type.get()
              ),
              # L1 regularization strength.
              ir.RankedTensorType.get(
                  [1, embedding_table_dim_size], ir.F32Type.get()
              ),
              # L2 regularization strength.
              ir.RankedTensorType.get(
                  [1, embedding_table_dim_size], ir.F32Type.get()
              ),
              # Max LR multiplier.
              ir.RankedTensorType.get(
                  [1, embedding_table_dim_size], ir.F32Type.get()
              ),
              # Global step.
              ir.RankedTensorType.get(
                  [1, embedding_table_dim_size], ir.F32Type.get()
              ),
          ],
          [
              ir.TupleType.get_tuple([
                  # Updated embedding table.
                  ir.RankedTensorType.get(
                      [1, embedding_table_dim_size], ir.F32Type.get()
                  ),
                  # Updated accumulator.
                  ir.RankedTensorType.get(
                      [1, embedding_table_dim_size], ir.F32Type.get()
                  ),
                  # Updated local step.
                  ir.RankedTensorType.get(
                      [1, embedding_table_dim_size], ir.F32Type.get()
                  ),
              ]),
          ],
      ),
      ip=ctx.module_context.ip,
      visibility="private",
  )

  entry_block = optimizer_update.add_entry_block()
  with ir.InsertionPoint(entry_block):
    grad = entry_block.arguments[0]
    param = entry_block.arguments[1]
    acc = entry_block.arguments[2]
    l_step = entry_block.arguments[3]
    lr = entry_block.arguments[4]
    rho_val = entry_block.arguments[5]
    l1_val = entry_block.arguments[6]
    l2_val = entry_block.arguments[7]
    max_lr_multiplier_val = entry_block.arguments[8]
    g_step = entry_block.arguments[9]

    one_broadcasted = _hlo_f32(1.0, embedding_table_dim_size)

    # fa_multiplier = (global_step / local_step) ^ rho
    new_local_step = hlo.add(l_step, one_broadcasted)
    safe_global_step = hlo.maximum(g_step, one_broadcasted)
    frequency_ratio = hlo.divide(safe_global_step, new_local_step)
    fa_multiplier = hlo.power(frequency_ratio, rho_val)
    fa_multiplier = hlo.minimum(fa_multiplier, max_lr_multiplier_val)

    # new_accumulator = accumulator + grad^2
    grad_sq = hlo.multiply(grad, grad)
    new_accumulator = hlo.add(acc, grad_sq)

    denominator = hlo.sqrt(new_accumulator)

    # Regularization
    sign_param = hlo.sign(param)
    l1_decay = hlo.multiply(l1_val, sign_param)
    l2_shrinkage = hlo.multiply(l2_val, param)

    # update = fa_multiplier * grad / adagrad_denom + l1_decay + l2_shrinkage
    grad_scaled = hlo.divide(hlo.multiply(grad, fa_multiplier), denominator)
    update_sum = hlo.add(grad_scaled, hlo.add(l1_decay, l2_shrinkage))

    # total_update = lr * update_sum
    update_total = hlo.multiply(lr, update_sum)

    new_param = hlo.subtract(param, update_total)
    new_param_clipped = utils.maybe_clip_params(new_param, min_value, max_value)

    updated_variables = hlo.tuple(
        [new_param_clipped, new_accumulator, new_local_step]
    )
    func_dialect.ReturnOp([updated_variables])

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
        accumulator,
        local_step,
        activations_grad,
    ]
  else:
    call_target = "SparseDenseMatmulGradOpWithOptimizerUpdate"
    operands += [
        activations_grad,
        embedding_table,
        accumulator,
        local_step,
    ]
  operands += [
      # hyperparameters
      learning_rate,
      rho,
      l1_regularization_strength,
      l2_regularization_strength,
      max_lr_multiplier,
      global_step,
  ]

  sparse_core_custom_call_op = jax.ffi.ffi_lowering(
      call_target,
      result_types=[
          ir.TupleType.get_tuple(
              [embedding_table.type, accumulator.type, local_step.type]
          )
      ],
      backend_config=backend_config,
      called_computations=[optimizer_update_computation_name],
      skip_ffi_layout_processing=True,
      api_version=1,
  )(ctx, *operands)

  table_tuple_op = hlo.GetTupleElementOp(sparse_core_custom_call_op, 0)
  table_tuple_op = _annotate_sparse_compute_type(table_tuple_op)
  accumulator_tuple_op = hlo.GetTupleElementOp(sparse_core_custom_call_op, 1)
  accumulator_tuple_op = _annotate_sparse_compute_type(accumulator_tuple_op)
  local_step_tuple_op = hlo.GetTupleElementOp(sparse_core_custom_call_op, 2)
  local_step_tuple_op = _annotate_sparse_compute_type(local_step_tuple_op)

  return (  # pytype: disable=bad-return-type
      table_tuple_op.results,
      accumulator_tuple_op.results,
      local_step_tuple_op.results,
  )


mlir.register_lowering(
    tpu_sparse_dense_matmul_grad_with_f2a_primitive,
    _tpu_sparse_dense_matmul_grad_with_f2a_lowering,
)
