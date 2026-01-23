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
"""Primitive tpu_sparse_dense_matmul_activation_unstack."""

import functools

import jax
from jax import core
from jax.experimental.layout import Layout
import jax.extend as jex
from jax.extend.mlir import ir
from jax.interpreters import mlir
from jax.interpreters import xla
import jax.numpy as jnp


tpu_sparse_dense_matmul_activation_unstack_primitive = jex.core.Primitive(
    "sparse_dense_matmul_activation_unstack",
)
tpu_sparse_dense_matmul_activation_unstack_primitive.multiple_results = True


tpu_sparse_dense_matmul_activation_unstack_primitive.def_impl(
    functools.partial(
        xla.apply_primitive,
        tpu_sparse_dense_matmul_activation_unstack_primitive,
    )
)


def _tpu_sparse_dense_matmul_activation_unstack_abstract_eval(
    stacked_activations,
    *,
    per_feature_batch_sizes: tuple[int, ...],
    per_feature_dims: tuple[int, ...],
):
  """Abstract evaluation for tpu_sparse_dense_matmul_activation_unstack."""
  del stacked_activations
  activations = []
  for batch_size, feature_dim in zip(per_feature_batch_sizes, per_feature_dims):
    activations.append(
        core.ShapedArray((batch_size, feature_dim), dtype=jnp.float32)
    )
  return tuple(activations)


tpu_sparse_dense_matmul_activation_unstack_primitive.def_abstract_eval(
    _tpu_sparse_dense_matmul_activation_unstack_abstract_eval
)


def _tpu_sparse_dense_matmul_activation_unstack_lowering(
    ctx,
    stacked_activations,
    *,
    per_feature_batch_sizes: tuple[int, ...],
    per_feature_dims: tuple[int, ...],
):
  """Lowering for tpu_sparse_dense_matmul_activation_unstack."""

  result_types = [
      ir.RankedTensorType.get(
          [batch_size, feature_dim],
          ir.F32Type.get(),
      )
      for batch_size, feature_dim in zip(
          per_feature_batch_sizes, per_feature_dims
      )
  ]

  call_target = "SparseActivationsUnstackInterleaved"

  operand_layouts = (Layout(major_to_minor=(0, 1)),)
  result_layouts = tuple(
      Layout(major_to_minor=(1, 0)) for _ in range(len(per_feature_batch_sizes))
  )

  op = jax.ffi.ffi_lowering(
      call_target,
      result_types=result_types,
      operand_layouts=operand_layouts,
      result_layouts=result_layouts,
      api_version=1,
  )(ctx, stacked_activations)

  return tuple(op)


mlir.register_lowering(
    tpu_sparse_dense_matmul_activation_unstack_primitive,
    _tpu_sparse_dense_matmul_activation_unstack_lowering,
)
