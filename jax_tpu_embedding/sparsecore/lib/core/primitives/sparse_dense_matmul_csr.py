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
"""Primitive tpu_sparse_dense_matmul_csr."""

import functools
import json
from typing import Any, Protocol, Sequence, runtime_checkable

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


@runtime_checkable
class QuantizationConfigProtocol(Protocol):
  """Protocol for SparseCore embedding quantization configurations."""

  def validate_table(
      self, embedding_dim: int, table_name: str | None = None
  ) -> None:
    ...

  def to_backend_config(self) -> dict[str, Any]:
    ...


# Define the sparse dense matmul primitive.
tpu_sparse_dense_matmul_csr_primitive = jex.core.Primitive(
    "sparse_dense_matmul_csr"
)


# Define the impl function for the sparse dense matmul primitive.
tpu_sparse_dense_matmul_csr_primitive.def_impl(
    functools.partial(
        xla.apply_primitive, tpu_sparse_dense_matmul_csr_primitive
    )
)


# Define the abstract eval function for the sparse dense matmul primitive.
def _tpu_sparse_dense_matmul_csr_abstract_eval(
    lhs_row_pointers: core.ShapedArray,
    lhs_local_embedding_ids: core.ShapedArray,
    lhs_local_sample_ids: core.ShapedArray,
    lhs_gains: core.ShapedArray,
    num_minibatches_per_physical_sparse_core: core.ShapedArray,
    embedding_table: core.ShapedArray,
    *,
    device_batch_size: int,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    sharding_strategy: int = 1,
    quantization_config: QuantizationConfigProtocol | None = None,
    # NOMUTANTS -- unused param for abstract eval.
    enable_minibatching: bool = False,
    table_name: str | None = None,
) -> core.ShapedArray:
  """Abstract eval for sdmm_csr."""

  del enable_minibatching, table_name

  embedding_dim = utils.get_embedding_dim(embedding_table)
  utils.validate_abstract_eval_params(
      lhs_row_pointers,
      lhs_local_embedding_ids,
      lhs_local_sample_ids,
      lhs_gains,
      num_minibatches_per_physical_sparse_core,
      embedding_table,
      activations_grad=core.ShapedArray(
          (device_batch_size, embedding_dim), np.float32
      ),  # Not used in the forward pass.
      max_ids_per_partition=max_ids_per_partition,
      max_unique_ids_per_partition=max_unique_ids_per_partition,
      computation_name="fwd-pass",  # Not used in the forward pass.
      sharding_strategy=sharding_strategy,
  )

  if quantization_config is not None:
    if not isinstance(quantization_config, QuantizationConfigProtocol):
      raise ValueError(
          f"Unsupported quantization_config: {quantization_config!r}"
      )
    quantization_config.validate_table(embedding_dim)

  shape = (
      (device_batch_size, embedding_dim)
      if len(embedding_table.shape) > 1
      else (device_batch_size,)
  )
  return core.ShapedArray(shape, dtype=jnp.float32)


tpu_sparse_dense_matmul_csr_primitive.def_abstract_eval(
    _tpu_sparse_dense_matmul_csr_abstract_eval
)


# Define the mlir lowering rule for the sparse dense matmul primitive.
def _tpu_sparse_dense_matmul_csr_lowering(
    ctx: mlir.LoweringRuleContext,
    lhs_row_pointers: ir.BlockArgument,
    lhs_local_embedding_ids: ir.BlockArgument,
    lhs_local_sample_ids: ir.BlockArgument,
    lhs_gains: ir.BlockArgument,
    num_minibatches_per_physical_sparse_core: ir.BlockArgument,
    embedding_table: ir.BlockArgument,
    *,
    device_batch_size: int,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    sharding_strategy: int = 1,
    quantization_config: QuantizationConfigProtocol | None = None,
    enable_minibatching: bool = False,
    table_name: str | None = None,
) -> Sequence[ir.Value]:
  """Lowering for tpu_sparse_dense_matmul_csr."""
  constant_op = hlo.constant(ir.DenseElementsAttr.get(np.float32(0.0)))
  embedding_table_type = ir.RankedTensorType(embedding_table.type)
  if embedding_table_type.rank > 1:
    activation_init = hlo.broadcast(
        constant_op,
        mlir.dense_int_array([
            device_batch_size,
            embedding_table_type.get_dim_size(1),
        ]),
    )
  else:
    activation_init = hlo.broadcast(
        constant_op,
        mlir.dense_int_array([
            device_batch_size,
        ]),
    )

  sdmm_csr_config = {
      "max_ids_per_partition": max_ids_per_partition,
      "max_unique_ids_per_partition": max_unique_ids_per_partition,
      "sharding_strategy": sharding_strategy,
      "pad_value": constants.PADDING_VALUE,
  }
  # Add quantization params only when enabled
  if quantization_config is not None:
    sdmm_csr_config = {
        **sdmm_csr_config,
        "quantization_config": quantization_config.to_backend_config(),
    }
  if table_name is not None:
    sdmm_csr_config = {**sdmm_csr_config, "table_name": table_name}
  backend_config = json.dumps({
      "sparse_dense_matmul_config": sdmm_csr_config,
      "device_type": "DEVICE_TYPE_SPARSECORE",
  })
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
          embedding_table,
          activation_init,
      ]
  )

  if enable_minibatching:  # Buffer contains minibatches
    call_target = "SparseDenseMatmulWithMinibatchingOp"
  else:  # Buffer contains per SC buffer
    # We still have tests that use this format.
    call_target = "SparseDenseMatmulOp"

  results = jax.ffi.ffi_lowering(
      call_target,
      result_types=[
          mlir.aval_to_ir_type(ctx.module_context, aval)
          for aval in ctx.avals_out
      ],
      api_version=1,
      backend_config=backend_config,
      skip_ffi_layout_processing=True,
  )(ctx, *operands)
  return utils.to_value_sequence(results)


mlir.register_lowering(
    tpu_sparse_dense_matmul_csr_primitive, _tpu_sparse_dense_matmul_csr_lowering
)


# Define the sparse dense matmul primitive that also returns FDO statistics.
#
# It returns `[activations, max_ids, max_unique_ids]`. Each FDO statistic holds
# one HBM word per SparseCore of the device, and every element in a
# SparseCore's word holds the same value: the maximum, over the partitions the
# SparseCore sends IDs to, of the number of IDs (or of unique IDs) sent to that
# partition. With minibatching, the TPU compiler requires a single minibatch
# per step (device minibatching) and checks it at runtime.
#
# The TPU compiler infers the statistics outputs from the number of results, so
# the backend config is the same as for `tpu_sparse_dense_matmul_csr_primitive`.
# FDO statistic calculation must be enabled in the TPU compiler, otherwise
# compilation fails.
tpu_sparse_dense_matmul_csr_with_fdo_statistics_primitive = (
    jex.core.Primitive("sparse_dense_matmul_csr_with_fdo_statistics")
)
tpu_sparse_dense_matmul_csr_with_fdo_statistics_primitive.multiple_results = (
    True
)


tpu_sparse_dense_matmul_csr_with_fdo_statistics_primitive.def_impl(
    functools.partial(
        xla.apply_primitive,
        tpu_sparse_dense_matmul_csr_with_fdo_statistics_primitive,
    )
)


def _tpu_sparse_dense_matmul_csr_with_fdo_statistics_abstract_eval(
    *args: core.ShapedArray,
    fdo_statistics_size: int,
    **kwargs: Any,
) -> list[core.ShapedArray]:
  """Abstract eval for sdmm_csr with FDO statistics."""
  if fdo_statistics_size <= 0:
    raise ValueError(
        f"`fdo_statistics_size` must be positive, got {fdo_statistics_size}."
    )
  activations = _tpu_sparse_dense_matmul_csr_abstract_eval(*args, **kwargs)
  fdo_statistic = core.ShapedArray((fdo_statistics_size,), dtype=jnp.int32)
  return [activations, fdo_statistic, fdo_statistic]


tpu_sparse_dense_matmul_csr_with_fdo_statistics_primitive.def_abstract_eval(
    _tpu_sparse_dense_matmul_csr_with_fdo_statistics_abstract_eval
)


def _tpu_sparse_dense_matmul_csr_with_fdo_statistics_lowering(
    ctx: mlir.LoweringRuleContext,
    *args: ir.BlockArgument,
    fdo_statistics_size: int,
    **kwargs: Any,
) -> Sequence[ir.Value]:
  """Lowering for tpu_sparse_dense_matmul_csr with FDO statistics."""
  # The statistics outputs are already in `ctx.avals_out`.
  del fdo_statistics_size
  return _tpu_sparse_dense_matmul_csr_lowering(ctx, *args, **kwargs)


mlir.register_lowering(
    tpu_sparse_dense_matmul_csr_with_fdo_statistics_primitive,
    _tpu_sparse_dense_matmul_csr_with_fdo_statistics_lowering,
)
