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
from typing import Sequence
import jax
from jax import core
from jax.experimental.layout import Layout
import jax.extend as jex
from jax.extend.mlir import ir
from jax.interpreters import mlir
from jax.interpreters import xla
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core.primitives import utils


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
    *unstacked_gradients: core.ShapedArray,
    stacked_batch_size: int,
    stacked_feature_dim: int,
) -> core.ShapedArray:
  """Abstract evaluation for tpu_sparse_dense_matmul_gradient_stack."""
  if not unstacked_gradients:
    raise ValueError("unstacked_gradients must be non-empty.")
  expected_dtype = unstacked_gradients[0].dtype
  if expected_dtype not in (jnp.float32, jnp.bfloat16):
    raise ValueError(
        f"Unsupported unstacked gradient dtype {expected_dtype}. Only float32"
        " and bfloat16 are supported."
    )
  for gradient in unstacked_gradients:
    if gradient.ndim != 2:
      raise ValueError(
          f"Each unstacked gradient must be rank 2, got shape {gradient.shape}."
      )
    if gradient.dtype != expected_dtype:
      raise ValueError(
          "All unstacked_gradients must have the same dtype, got"
          f" {[g.dtype for g in unstacked_gradients]}."
      )
  if stacked_batch_size <= 0 or stacked_feature_dim <= 0:
    raise ValueError(
        "stacked_batch_size and stacked_feature_dim must be positive, got"
        f" ({stacked_batch_size}, {stacked_feature_dim})."
    )
  total_rows = sum(gradient.shape[0] for gradient in unstacked_gradients)
  if total_rows != stacked_batch_size:
    raise ValueError(
        "Sum of unstacked gradient rows must equal stacked_batch_size, got"
        f" {total_rows} vs {stacked_batch_size}."
    )
  max_cols = max(gradient.shape[1] for gradient in unstacked_gradients)
  if max_cols > stacked_feature_dim:
    raise ValueError(
        "Max unstacked gradient columns cannot exceed stacked_feature_dim, got"
        f" {max_cols} vs {stacked_feature_dim}."
    )
  return core.ShapedArray(
      (stacked_batch_size, stacked_feature_dim), dtype=jnp.float32
  )


tpu_sparse_dense_matmul_gradient_stack_primitive.def_abstract_eval(
    _tpu_sparse_dense_matmul_gradient_stack_abstract_eval
)


def _tpu_sparse_dense_matmul_gradient_stack_lowering(
    ctx: mlir.LoweringRuleContext,
    *stacked_activations: ir.BlockArgument,
    stacked_batch_size: int,
    stacked_feature_dim: int,
) -> Sequence[ir.Value]:
  """Lowering for tpu_sparse_dense_matmul_gradient_stack."""
  # ensure stacked_activations have greater or equal to one element.
  assert len(stacked_activations) >= 1
  operand_layouts = tuple(
      Layout(major_to_minor=(1, 0)) for _ in range(len(stacked_activations))
  )
  result_layouts = (Layout(major_to_minor=(0, 1)),)

  call_target = "SparseGradientsStackInterleaved"

  results = jax.ffi.ffi_lowering(
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
  return utils.to_value_sequence(results)


mlir.register_lowering(
    tpu_sparse_dense_matmul_gradient_stack_primitive,
    _tpu_sparse_dense_matmul_gradient_stack_lowering,
)
