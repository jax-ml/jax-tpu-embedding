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

Depending on the optimizer, a different number of embedding variables may be
required. For example, for Adagrad, the optimizer update computation requires
both the embedding table and the accumulator.

These variables are passed in individually as positional operands:
  - the embedding table (2D: [vocab_size, emb_dim]) first,
  - followed by any number of slot variables, each of which may be
    2D ([vocab_size, emb_dim]) **or** 1D ([vocab_size]).

The order in which the variables are provided _must_ be identical to the order
that the XLA compiler expects. For example, for Adagrad, the embedding table
must be at index 0 and the accumulator must be at index 1.

The hyperparameters are passed as trailing scalar operands (0D tensors) after
the activation gradients argument. The order of the hyperparameters _must_ be
identical to the order that the XLA compiler expects. For example, for SGD and
Adagrad, the learning rate must be at index 0.
"""

import functools
import json
from typing import Callable, Tuple

import jax
from jax import core
from jax import numpy as jnp
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
import jax.extend as jex
from jax.interpreters import mlir
from jax.interpreters import xla
from jax_tpu_embedding.sparsecore.lib.core import constants
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
    *args,
    optimizer_generator: Callable[[mlir.LoweringRuleContext, str, int], None],
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "sparse_dense_matmul_optimizer_grad",
    sharding_strategy: int = 1,
):
  """Abstract eval for sparse_dense_matmul_adagrad."""
  if lhs_row_pointers.dtype != np.int32:
    raise ValueError(
        f"lhs_row_pointers must have type int32, got {lhs_row_pointers.dtype}"
    )
  if lhs_local_sample_ids.dtype != np.int32:
    raise ValueError(
        "lhs_local_sample_ids must have type int32, got"
        f" {lhs_local_sample_ids.dtype}"
    )
  if lhs_local_embedding_ids.dtype != np.int32:
    raise ValueError(
        "lhs_local_embedding_ids must have type uint32, got"
        f" {lhs_local_embedding_ids.dtype}"
    )
  if lhs_gains.dtype != np.float32:
    raise ValueError(f"lhs_gains must have type float32, got {lhs_gains.dtype}")
  # Args Layout:
  # num_minibatches, tables, activations_grad, *hparams
  arg_list = list(args)
  # Strip optional num_minibatches if present (scalar of any dtype).
  if arg_list and not arg_list[0].shape:
    arg_list = arg_list[1:]

  # Split trailing scalar hyperparameters.
  split = len(arg_list)
  while split > 0 and not arg_list[split - 1].shape:
    split -= 1
  non_hparams = arg_list[:split]

  if not non_hparams:
    raise ValueError("Missing activations_grad and table operands.")
  activations_grad = non_hparams[-1]
  tables = non_hparams[:-1]

  if not tables:
    raise ValueError("At least one table (the embedding variable) is required.")
  if activations_grad.dtype != np.float32 or len(activations_grad.shape) != 2:
    raise ValueError(
        "activations_grad must be rank-2 with dtype float32, got"
        f" dtype {activations_grad.dtype} and shape {activations_grad.shape}"
    )
  # Validate tables: embedding table first (2D), slots may be 1D or 2D.
  if tables[0].dtype != np.float32 or len(tables[0].shape) != 2:
    raise ValueError(
        "The first table must be the embedding table (rank-2, dtype float32),"
        f" got dtype {tables[0].dtype} and shape {tables[0].shape}"
    )
  for t in tables:
    if t.dtype != np.float32:
      raise ValueError("All tables must have dtype float32.")
    if len(t.shape) not in (1, 2):
      raise ValueError(
          "Slot variables must be rank-1 or rank-2; got shape {t.shape}."
      )
  if len(lhs_row_pointers.shape) != 1:
    raise ValueError(
        f"lhs_row_pointers must have rank 1, got {lhs_row_pointers.shape}"
    )
  if (
      lhs_local_sample_ids.shape != lhs_local_embedding_ids.shape
      or lhs_gains.shape != lhs_local_embedding_ids.shape
      or len(lhs_local_sample_ids.shape) != 1
  ):
    raise ValueError(
        "LHS sample IDs, embedding IDs, and gains must all have "
        f"equal rank 1 shapes, got shapes {lhs_local_sample_ids.shape}, "
        f"{lhs_local_embedding_ids.shape} and {lhs_gains.shape}"
    )
  if not callable(optimizer_generator):
    raise ValueError("optimizer_generator must be callable")

  if sharding_strategy != 1:
    raise ValueError(
        f"sharding_strategy must be MOD (1), got {sharding_strategy}"
    )

  if max_ids_per_partition <= 0:
    raise ValueError(
        f"max_ids_per_partition must be positive, got {max_ids_per_partition}"
    )

  if max_unique_ids_per_partition <= 0:
    raise ValueError(
        "max_unique_ids_per_partition must be positive, got"
        f" {max_unique_ids_per_partition}"
    )
  if not computation_name:
    raise ValueError("computation_name must be non-empty")

  return tuple(
      core.ShapedArray(t.shape, jnp.float32) for t in tables
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
    *args: mlir.ir.Value,
    optimizer_generator: Callable[[mlir.LoweringRuleContext, str, int], None],
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "sparse_dense_matmul_optimizer_grad",
    sharding_strategy: int = 1,
) -> Tuple[np.ndarray, ...]:
  """Lowering for sparse_dense_matmul_optimizer_grad."""
  args = list(args)
  if args and ir.RankedTensorType(args[0].type).rank == 0:
    args = args[1:]

  # Split trailing scalar hyperparameters.
  split = len(args)
  while split > 0 and ir.RankedTensorType(args[split - 1].type).rank == 0:
    split -= 1
  non_hparams = args[:split]
  hyperparams = args[split:]
  if not non_hparams:
    raise ValueError("Missing activations_grad and table operands.")
  activations_grad = non_hparams[-1]
  tables = non_hparams[:-1]

  if not tables:
    raise ValueError("At least one embedding table is required.")
  table_type = tables[0].type
  emb_rank = ir.RankedTensorType(table_type).rank
  if emb_rank != 2:
    raise ValueError("First table must be rank-2 embedding variable.")
  num_slot_variables = len(tables) - 1
  num_hyperparameters = len(hyperparams)
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

  optimizer_generator(
      ctx,
      optimizer_update_computation_name,
      tables[0].type.maybe_downcast().get_dim_size(1),
  )

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
          ir.TupleType.get_tuple([t.type for t in tables])  # pylint: disable=attribute-error
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
