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
from jax import numpy as jnp
import jax.extend as jex
from jax.extend import source_info_util
from jax.extend.mlir import ir
from jax.extend.mlir.dialects import func as func_dialect
from jax.extend.mlir.dialects import stablehlo as hlo
from jax.interpreters import mlir
from jax.interpreters import xla
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
    jaxpr: jex.core.ClosedJaxpr,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "sparse_dense_matmul_optimizer_grad",
    sharding_strategy: int = 1,
) -> Tuple[core.ShapedArray, ...]:
  """Abstract eval for sparse_dense_matmul_adagrad."""
  hyperparameters = hyperparams_and_embedding_vars[:num_hyperparameters]
  embedding_variables = hyperparams_and_embedding_vars[num_hyperparameters:]

  if not embedding_variables:
    raise ValueError("At least one embedding variable must be passed.")

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
    if len(var.shape) != 2:
      raise ValueError(f"embedding variables must have rank 2, got {var.shape}")
  if not isinstance(jaxpr, jex.core.ClosedJaxpr):
    raise ValueError("jaxpr must be a ClosedJaxpr")

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
    jaxpr: jex.core.ClosedJaxpr,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "sparse_dense_matmul_optimizer_grad",
    sharding_strategy: int = 1,
) -> Tuple[Sequence[ir.Value], ...]:
  """Lowering for sparse_dense_matmul_optimizer_grad."""
  del num_minibatches_per_physical_sparse_core
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

  optimizer_update_computation_name = computation_name

  tables = list(embedding_variables)
  table_shape = (
      ir.RankedTensorType(tables[0].type).get_dim_size(0),
      ir.RankedTensorType(tables[0].type).get_dim_size(1),
  )
  dim_size = table_shape[1]
  row_tensor_type = ir.RankedTensorType.get([1, dim_size], ir.F32Type.get())

  wrapper_input_types = [row_tensor_type]  # grad
  for _ in tables:
    wrapper_input_types.append(row_tensor_type)
  for _ in range(num_hyperparameters):
    wrapper_input_types.append(row_tensor_type)

  const_types = [mlir.aval_to_ir_type(v.aval) for v in jaxpr.constvars]
  wrapper_input_types.extend(const_types)

  output_types = [row_tensor_type for _ in range(num_slot_variables + 1)]

  wrapper_func = func_dialect.FuncOp(
      optimizer_update_computation_name,
      (
          wrapper_input_types,
          [ir.TupleType.get_tuple(output_types)],
      ),
      ip=ctx.module_context.ip,
      visibility="private",
  )

  entry_block = wrapper_func.add_entry_block()
  with ir.InsertionPoint(entry_block):
    wa = list(entry_block.arguments)

    in_args = wa[: 1 + len(tables) + num_hyperparameters]
    consts_wa = wa[1 + len(tables) + num_hyperparameters :]

    name_stack = source_info_util.NameStack()
    tokens_in = mlir.TokenSet()

    out_vals, _ = mlir.jaxpr_subcomp(
        ctx.module_context,
        jaxpr.jaxpr,
        name_stack,
        tokens_in,
        consts_wa,
        *in_args,
        dim_var_values=[],
        const_lowering={},
        outer_traceback=None,
    )

    flat_out_vals = []
    for v in out_vals:
      if isinstance(v, (list, tuple)):
        flat_out_vals.extend(v)
      else:
        flat_out_vals.append(v)

    result_tuple = hlo.tuple(flat_out_vals)
    func_dialect.ReturnOp([result_tuple])

  hyperparams = []
  f32type = mlir.aval_to_ir_type(core.ShapedArray((), np.float32))
  for param in hyperparameters:
    if ir.RankedTensorType(param.type).rank == 0:
      hyperparams.append(param)
    else:
      reshaped = hlo.reshape(f32type, param)
      hyperparams.append(reshaped)

  operands = (
      [
          lhs_row_pointers,
          lhs_local_embedding_ids,
          lhs_local_sample_ids,
          lhs_gains,
          activations_grad,
      ]
      + tables
      + hyperparams
  )
  op = jax.ffi.ffi_lowering(
      "SparseDenseMatmulGradOpWithOptimizerUpdate",
      result_types=[
          ir.TupleType.get_tuple([tables[0].type for _ in range(len(tables))])
      ],
      backend_config=backend_config,
      called_computations=[optimizer_update_computation_name],
      skip_ffi_layout_processing=True,
      api_version=1,
  )(ctx, *operands)

  result = []
  for i in range(len(tables)):
    tuple_op = hlo.GetTupleElementOp(op, i)
    tuple_op.attributes["mhlo.frontend_attributes"] = ir.DictAttr.get(
        {"_xla_compute_type": ir.StringAttr.get("sparse")}
    )
    result.append(tuple_op.results)
  return tuple(result)


mlir.register_lowering(
    tpu_sparse_dense_matmul_optimizer_grad_primitive,
    _tpu_sparse_dense_matmul_optimizer_grad_lowering,
)
