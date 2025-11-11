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
from typing import Callable, Tuple

import jax
from jax import core
from jax import numpy as jnp
import jax.extend as jex
from jax.extend.mlir import ir
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
    lhs_row_pointers: np.ndarray,
    lhs_local_embedding_ids: np.ndarray,
    lhs_local_sample_ids: np.ndarray,
    lhs_gains: np.ndarray,
    num_minibatches_per_physical_sparse_core: np.int32,
    embedding_variables: np.ndarray,
    activations_grad: np.ndarray,
    hyperparameters: np.ndarray,
    *_,
    optimizer_generator: Callable[[mlir.LoweringRuleContext, str, int], None],
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "sparse_dense_matmul_optimizer_grad",
    sharding_strategy: int = 1,
):
  """Abstract eval for sparse_dense_matmul_adagrad."""
  utils.validate_abstract_eval_params(
      lhs_row_pointers=lhs_row_pointers,
      lhs_local_embedding_ids=lhs_local_embedding_ids,
      lhs_local_sample_ids=lhs_local_sample_ids,
      lhs_gains=lhs_gains,
      num_minibatches_per_physical_sparse_core=num_minibatches_per_physical_sparse_core,
      embedding_table=core.ShapedArray(
          embedding_variables.shape[1:], embedding_variables.dtype
      ),
      activations_grad=activations_grad,
      max_ids_per_partition=max_ids_per_partition,
      max_unique_ids_per_partition=max_unique_ids_per_partition,
      computation_name=computation_name,
      sharding_strategy=sharding_strategy,
  )

  if hyperparameters.dtype != np.float32 or len(hyperparameters.shape) != 1:
    raise ValueError(
        "hyperparameters must be 1 dimensional with dtype float32, got"
        f" {hyperparameters.dtype} and shape {hyperparameters.shape}"
    )
  if len(embedding_variables.shape) != 3:
    raise ValueError(
        f"embedding_table must have rank 3, got {embedding_variables.shape}"
    )
  if not callable(optimizer_generator):
    raise ValueError("optimizer_generator must be callable")

  num_tables = embedding_variables.shape[0]
  return tuple(
      core.ShapedArray(
          (embedding_variables.shape[1], embedding_variables.shape[2]),
          dtype=jnp.float32,
      )
      for _ in range(num_tables)
  )


tpu_sparse_dense_matmul_optimizer_grad_primitive.def_abstract_eval(
    _tpu_sparse_dense_matmul_optimizer_grad_abstract_eval
)


def _tpu_sparse_dense_matmul_optimizer_grad_lowering(
    ctx: mlir.LoweringRuleContext,
    lhs_row_pointers: mlir.ir.BlockArgument,
    lhs_local_embedding_ids: mlir.ir.BlockArgument,
    lhs_local_sample_ids: mlir.ir.BlockArgument,
    lhs_gains: mlir.ir.BlockArgument,
    num_minibatches_per_physical_sparse_core: mlir.ir.BlockArgument,
    embedding_variables: mlir.ir.BlockArgument,
    activations_grad: mlir.ir.BlockArgument,
    hyperparameters: mlir.ir.BlockArgument,
    *,
    optimizer_generator: Callable[[mlir.LoweringRuleContext, str, int], None],
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "sparse_dense_matmul_optimizer_grad",
    sharding_strategy: int = 1,
) -> Tuple[np.ndarray, ...]:
  """Lowering for sparse_dense_matmul_optimizer_grad."""
  del num_minibatches_per_physical_sparse_core
  num_slot_variables = (
      ir.RankedTensorType(embedding_variables.type).get_dim_size(0) - 1
  )
  num_hyperparameters = ir.RankedTensorType(hyperparameters.type).get_dim_size(
      0
  )
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

  # Because we cannot take in a tuple or list of Nd arrays, we need to slice
  # the embedding tables into individual tables. The order of the user input
  # must be kept intact.
  tables = []
  table_shape = (
      ir.RankedTensorType(embedding_variables.type).get_dim_size(1),
      ir.RankedTensorType(embedding_variables.type).get_dim_size(2),
  )
  for i in range(num_slot_variables + 1):
    sliced = hlo.slice(
        embedding_variables,
        mlir.dense_int_array([i, 0, 0]),
        mlir.dense_int_array([i + 1, table_shape[0], table_shape[1]]),
        mlir.dense_int_array([1, 1, 1]),
    )
    sliced = hlo.reshape(
        ir.RankedTensorType.get(
            [table_shape[0], table_shape[1]],
            ir.F32Type.get(),
        ),
        sliced,
    )
    tables.append(sliced)
  optimizer_generator(
      ctx,
      optimizer_update_computation_name,
      ir.RankedTensorType(tables[0].type).get_dim_size(1),
  )
  hyperparams = []
  f32type = mlir.aval_to_ir_type(core.ShapedArray((), np.float32))
  for i in range(num_hyperparameters):
    sliced_param = hlo.slice(
        hyperparameters,
        mlir.dense_int_array([i]),
        mlir.dense_int_array([i + 1]),
        mlir.dense_int_array([1]),
    )
    sliced_param = hlo.reshape(
        f32type,
        sliced_param,
    )
    hyperparams.append(sliced_param)

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
          ir.TupleType.get_tuple([tables[0].type for _ in range(len(tables))])  # pylint: disable=attribute-error
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
