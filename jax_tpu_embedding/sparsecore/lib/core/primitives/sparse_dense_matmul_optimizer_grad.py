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
"""Primitive for sparse dense matmul grad with optimizer.

This module mainly exposes a single Jax primitive that can be used to apply
optimizer updates to the embedding tables.

The optimizer update is defined by the optimizer_generator. The optimizer
generator is a function that generates the MLIR code for the optimizer update
computation. Take a look in optimizer.py for examples.

Depending on the optimizer, different number of embedding variables may be
required. For example, for Adagrad, the optimizer update computation requires
both the embedding table and the accumulator.

These variables are passed in as an 3D array of shape [num_tables, vocab_size,
emb_size].
The order in which the variables are stacked _must_ be identical to the order
that the XLA compiler expects. For example, for Adagrad, the embedding table
must be at index 0 and the accumulator must be at index 1.

The hyperparameters are passed in as a 1D array of shape [num_hyperparameters].
The order of the hyperparameters _must_ be identical to the order that the XLA
compiler expects. For example, for SGD and Adagrad, the learning rate must be at
index 0.
"""

import functools
import json
from typing import Sequence, Tuple

import jax
from jax import core
import jax.extend as jex
from jax.extend.mlir import ir
from jax.extend.mlir.dialects import stablehlo as hlo
from jax.interpreters import mlir
from jax.interpreters import xla
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core import constants
from jax_tpu_embedding.sparsecore.lib.core.primitives import utils
import numpy as np


tpu_sparse_dense_matmul_optimizer_grad_primitive = jex.core.Primitive(
    "sparse_dense_matmul_optimizer_grad_primitive",
)

tpu_sparse_dense_matmul_optimizer_grad_primitive.multiple_results = True


tpu_sparse_dense_matmul_optimizer_grad_primitive.def_impl(
    functools.partial(
        xla.apply_primitive,
        tpu_sparse_dense_matmul_optimizer_grad_primitive,
    )
)


def _tpu_sparse_dense_matmul_optimizer_grad_abstract_eval(
    lhs_row_pointers: core.ShapedArray,
    lhs_local_embedding_ids: core.ShapedArray,
    lhs_local_sample_ids: core.ShapedArray,
    lhs_gains: core.ShapedArray,
    num_minibatches_per_physical_sparse_core: core.ShapedArray,
    activations_grad: core.ShapedArray,
    *hyperparams_and_embedding_vars: core.ShapedArray,
    num_hyperparameters: int,
    stablehlo: str | bytes | ir.Module,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "sparse_dense_matmul_optimizer_grad",
    sharding_strategy: int = 1,
    enable_minibatching: bool = False,
) -> Tuple[core.ShapedArray, ...]:
  """Abstract eval for sparse_dense_matmul_optimizer_grad."""
  del enable_minibatching
  hyperparameters = hyperparams_and_embedding_vars[:num_hyperparameters]
  embedding_variables = hyperparams_and_embedding_vars[num_hyperparameters:]

  if not embedding_variables:
    raise ValueError("At least one embedding variable must be passed.")

  # Squeeze trailing dimensions of size 1 (e.g. [N, 1] -> [N]) to support 1D
  # embedding variables.
  activations_grad = utils.maybe_squeeze_abstract_eval(activations_grad, 1)
  embedding_variables = tuple(
      utils.maybe_squeeze_abstract_eval(var, 1) for var in embedding_variables
  )

  utils.validate_abstract_eval_params(
      lhs_row_pointers=lhs_row_pointers,
      lhs_local_embedding_ids=lhs_local_embedding_ids,
      lhs_local_sample_ids=lhs_local_sample_ids,
      lhs_gains=lhs_gains,
      num_minibatches_per_physical_sparse_core=num_minibatches_per_physical_sparse_core,
      embedding_table=embedding_variables[0],
      activations_grad=activations_grad,
      max_ids_per_partition=max_ids_per_partition,
      max_unique_ids_per_partition=max_unique_ids_per_partition,
      computation_name=computation_name,
      sharding_strategy=sharding_strategy,
  )

  for param in hyperparameters:
    if param.dtype != np.float32:
      raise ValueError(f"hyperparameters must be float32, got {param.dtype}")
    if len(param.shape) != 0 and param.shape != (1,):
      raise ValueError(
          f"hyperparameters must be scalars or 1D of size 1, got {param.shape}"
      )

  for var in embedding_variables:
    if len(var.shape) not in (1, 2):
      raise ValueError(
          f"embedding variables must have rank 1 or 2, got {var.shape}"
      )
  if not isinstance(stablehlo, (str, bytes, ir.Module)):
    raise ValueError(
        "stablehlo must be a string, bytes, or ir.Module, got"
        f" {type(stablehlo)}"
    )

  return tuple(
      core.ShapedArray(
          var.shape,
          dtype=jnp.float32,
      )
      for var in embedding_variables
  )


tpu_sparse_dense_matmul_optimizer_grad_primitive.def_abstract_eval(
    _tpu_sparse_dense_matmul_optimizer_grad_abstract_eval
)


def _tpu_sparse_dense_matmul_optimizer_grad_lowering(
    ctx: mlir.LoweringRuleContext,
    lhs_row_pointers: ir.BlockArgument,
    lhs_local_embedding_ids: ir.BlockArgument,
    lhs_local_sample_ids: ir.BlockArgument,
    lhs_gains: ir.BlockArgument,
    num_minibatches_per_physical_sparse_core: ir.BlockArgument,
    activations_grad: ir.BlockArgument,
    *hyperparams_and_embedding_vars: ir.BlockArgument,
    num_hyperparameters: int,
    stablehlo: str | bytes | ir.Module,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "sparse_dense_matmul_optimizer_grad",
    sharding_strategy: int = 1,
    enable_minibatching: bool = False,
) -> Tuple[Sequence[ir.Value], ...]:
  """Lowering for sparse_dense_matmul_optimizer_grad."""
  hyperparameters = hyperparams_and_embedding_vars[:num_hyperparameters]
  embedding_variables = hyperparams_and_embedding_vars[num_hyperparameters:]

  num_slot_variables = len(embedding_variables) - 1
  sdmm_sgd_config = {
      "max_ids_per_partition": max_ids_per_partition,
      "max_unique_ids_per_partition": max_unique_ids_per_partition,
      "pad_value": constants.PADDING_VALUE,
      "sharding_strategy": sharding_strategy,
      "num_slot_variables": num_slot_variables,
      "num_hyperparameters": num_hyperparameters,
  }
  backend_config = json.dumps({
      "sparse_dense_matmul_config": sdmm_sgd_config,
      "device_type": "DEVICE_TYPE_SPARSECORE",
  })

  tables = list(embedding_variables)
  if isinstance(stablehlo, (str, bytes)):
    submodule = ir.Module.parse(stablehlo, context=ctx.module_context.context)
  elif isinstance(stablehlo, ir.Module):
    submodule = stablehlo
  else:
    raise ValueError(f"Unsupported stablehlo type: {type(stablehlo)}")

  optimizer_update_computation_name = mlir.merge_mlir_modules(
      ctx.module_context.module,
      computation_name,
      submodule,
      dst_symtab=ctx.module_context.symbol_table,
  )

  hyperparams = []
  f32type = mlir.aval_to_ir_type(
      ctx.module_context, core.ShapedArray((), np.float32)
  )
  for param in hyperparameters:
    if ir.RankedTensorType(param.type).rank == 0:
      hyperparams.append(param)
    else:
      reshaped = hlo.reshape(f32type, param)
      hyperparams.append(reshaped)

  activations_grad_sq = utils.maybe_squeeze_ir(activations_grad, 1)
  tables_sq = [utils.maybe_squeeze_ir(table, 1) for table in tables]

  if enable_minibatching:
    call_target = "SparseDenseMatmulGradOptimizerUpdateWithMinibatchingOp"
    operands = (
        [
            lhs_row_pointers,
            lhs_local_embedding_ids,
            lhs_local_sample_ids,
            lhs_gains,
            num_minibatches_per_physical_sparse_core,
        ]
        + tables_sq
        + [activations_grad_sq]
        + hyperparams
    )
  else:
    call_target = "SparseDenseMatmulGradOpWithOptimizerUpdate"
    operands = (
        [
            lhs_row_pointers,
            lhs_local_embedding_ids,
            lhs_local_sample_ids,
            lhs_gains,
            activations_grad_sq,
        ]
        + tables_sq
        + hyperparams
    )

  op = jax.ffi.ffi_lowering(
      call_target,
      result_types=[
          ir.TupleType.get_tuple([tables[0].type for _ in range(len(tables))])
      ],
      backend_config=backend_config,
      called_computations=[optimizer_update_computation_name],
      skip_ffi_layout_processing=True,
      api_version=1,
  )(ctx, *operands)

  result = []
  assert isinstance(op[0], ir.Value)
  for i in range(len(tables)):
    tuple_op = hlo.GetTupleElementOp(op[0], i)
    tuple_op.attributes["mhlo.frontend_attributes"] = ir.DictAttr.get(
        {"_xla_compute_type": ir.StringAttr.get("sparse")}
    )
    result.append(tuple_op.results)
  outputs: Tuple[Sequence[ir.Value], ...] = tuple(result)
  return outputs


mlir.register_lowering(
    tpu_sparse_dense_matmul_optimizer_grad_primitive,
    _tpu_sparse_dense_matmul_optimizer_grad_lowering,
)
