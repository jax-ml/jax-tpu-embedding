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
"""Primitive tpu_sparse_dense_matmul_gradient_stack."""

import functools

import jax
from jax import core
from jax.experimental.layout import Layout
import jax.extend as jex
from jax.extend.mlir import ir
from jax.interpreters import mlir
from jax.interpreters import xla
import jax.numpy as jnp


tpu_sparse_dense_matmul_gradient_stack_primitive = jex.core.Primitive(
    "sparse_dense_matmul_gradient_stack",
)


tpu_sparse_dense_matmul_gradient_stack_primitive.def_impl(
    functools.partial(
        xla.apply_primitive,
        tpu_sparse_dense_matmul_gradient_stack_primitive,
    )
)


def _tpu_sparse_dense_matmul_gradient_stack_abstract_eval(
    *unstacked_gradients,
    stacked_batch_size: int,
    stacked_feature_dim: int,
):
  """Abstract evaluation for tpu_sparse_dense_matmul_gradient_stack."""
  del unstacked_gradients
  return core.ShapedArray(
      (stacked_batch_size, stacked_feature_dim), dtype=jnp.float32
  )


tpu_sparse_dense_matmul_gradient_stack_primitive.def_abstract_eval(
    _tpu_sparse_dense_matmul_gradient_stack_abstract_eval
)


def _tpu_sparse_dense_matmul_gradient_stack_lowering(
    ctx,
    *stacked_activations,
    stacked_batch_size: int,
    stacked_feature_dim: int,
):
  """Lowering for tpu_sparse_dense_matmul_gradient_stack."""
  # ensure stacked_activations have greater or equal to one element.
  assert len(stacked_activations) >= 1
  operand_layouts = tuple(
      Layout(major_to_minor=(1, 0)) for _ in range(len(stacked_activations))
  )
  result_layouts = (Layout(major_to_minor=(0, 1)),)

  call_target = "SparseGradientsStackInterleaved"

  return jax.ffi.ffi_lowering(
      call_target,
      result_types=[
          ir.RankedTensorType.get(
              [stacked_batch_size, stacked_feature_dim],
              ir.F32Type.get(),
          )
      ],
      operand_layouts=operand_layouts,
      result_layouts=result_layouts,
      api_version=1,
  )(ctx, *stacked_activations)


mlir.register_lowering(
    tpu_sparse_dense_matmul_gradient_stack_primitive,
    _tpu_sparse_dense_matmul_gradient_stack_lowering,
)
