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
"""Utils for sparsecore grad primitives."""

from typing import Any, Sequence

import jax
from jax import core
from jax.extend.mlir import ir
from jax.extend.mlir.dialects import stablehlo as hlo
from jax.interpreters import mlir
import numpy as np


def maybe_clip_params(
    x: ir.Value,
    min_value: float | None = None,
    max_value: float | None = None,
) -> ir.Value:
  """Clips the embedding table to the min and max values."""
  if min_value is None and max_value is None:
    return x

  x_type = ir.RankedTensorType(x.type)
  bcast_dims = ir.DenseI64ArrayAttr.get([])

  min_bound = None
  if min_value is not None:
    min_scalar = hlo.constant(
        ir.DenseElementsAttr.get(np.array(min_value, dtype=np.float32))
    )
    min_bound = hlo.broadcast_in_dim(
        result=x_type, operand=min_scalar, broadcast_dimensions=bcast_dims
    )

  max_bound = None
  if max_value is not None:
    max_scalar = hlo.constant(
        ir.DenseElementsAttr.get(np.array(max_value, dtype=np.float32))
    )
    max_bound = hlo.broadcast_in_dim(
        result=x_type, operand=max_scalar, broadcast_dimensions=bcast_dims
    )

  if min_bound is not None and max_bound is not None:
    return hlo.clamp(min_bound, x, max_bound)
  if min_bound is not None:
    return hlo.maximum(min_bound, x)
  assert max_bound is not None
  return hlo.minimum(max_bound, x)


def ensure_dtype(check: Any, expected_type: Any, object_name: str):
  if check.dtype != expected_type:
    raise ValueError(
        f"{object_name} must have type {expected_type!r}, got {check.dtype!r}"
    )


def ensure_dim(
    check: Any, expected_dim: int | tuple[int, ...], object_name: str
):
  expected = (expected_dim,) if isinstance(expected_dim, int) else expected_dim
  if len(check.shape) not in expected:
    raise ValueError(
        f"{object_name} must have dim in {expected!r}, got {check.shape!r}"
    )


def validate_abstract_eval_params(
    lhs_row_pointers: core.ShapedArray,
    lhs_local_embedding_ids: core.ShapedArray,
    lhs_local_sample_ids: core.ShapedArray,
    lhs_gains: core.ShapedArray,
    num_minibatches_per_physical_sparse_core: core.ShapedArray,
    embedding_table: core.ShapedArray,
    activations_grad: core.ShapedArray,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str,
    sharding_strategy: int,
    min_value: float | None = None,
    max_value: float | None = None,
):
  """Validate parameters common to all sparsecore grad primitives."""
  ensure_dtype(lhs_row_pointers, np.int32, "lhs_row_pointers")
  ensure_dtype(lhs_local_sample_ids, np.int32, "lhs_local_sample_ids")
  ensure_dtype(lhs_local_embedding_ids, np.int32, "lhs_local_embedding_ids")
  ensure_dtype(lhs_gains, np.float32, "lhs_gains")
  ensure_dtype(
      num_minibatches_per_physical_sparse_core,
      np.int32,
      "num_minibatches_per_physical_sparse_core",
  )
  ensure_dtype(embedding_table, np.float32, "embedding_table")
  ensure_dtype(activations_grad, np.float32, "activations_grad")
  ensure_dim(lhs_row_pointers, 1, "lhs_row_pointers")
  ensure_dim(embedding_table, (1, 2), "embedding_table")
  ensure_dim(
      num_minibatches_per_physical_sparse_core,
      0,
      "num_minibatches_per_physical_sparse_core",
  )
  ensure_dim(activations_grad, (1, 2), "activations_grad")
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

  embedding_dim = get_embedding_dim(embedding_table)
  activations_grad_dim = get_embedding_dim(activations_grad)
  if embedding_dim != activations_grad_dim:
    raise ValueError(
        "embedding_table and activations_grad must have equal feature (minor)"
        f" dimensions, got {embedding_table.shape}, {activations_grad.shape}"
    )

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

  if min_value is not None and max_value is not None and min_value > max_value:
    raise ValueError(
        "min_value must be less than or equal to max_value, got"
        f" {min_value} and {max_value}"
    )


def get_embedding_dim(val: core.ShapedArray) -> int:
  """Returns the feature (embedding) dimension for a 1D or 2D shaped array."""
  return val.shape[1] if len(val.shape) > 1 else 1


def to_value_sequence(results: Any) -> Sequence[ir.Value]:
  """Converts FFI lowering results to a Sequence of ir.Values."""
  if isinstance(results, ir.Value):
    return [results]
  typed_results: list[ir.Value] = []
  for r in results:
    assert isinstance(r, ir.Value)
    typed_results.append(r)
  return typed_results


def aval_to_ir_type(
    ctx: mlir.LoweringRuleContext, aval: core.AbstractValue
) -> ir.Type:
  """Converts an abstract value to an MLIR type with JAX version compatibility."""
  if jax.__version_info__ >= (0, 10, 1):
    return mlir.aval_to_ir_type(ctx.module_context, aval)
  else:
    return mlir.aval_to_ir_type(aval)  # pyrefly: ignore[missing-argument, bad-argument-type]


def maybe_squeeze_abstract_eval(
    val: core.ShapedArray | Sequence[core.ShapedArray],
    expected_dim: int,
) -> Any:
  """Squeezes trailing dimensions of size 1 until rank matches expected_dim."""
  if isinstance(val, core.ShapedArray):
    shape = list(val.shape)
    while len(shape) > expected_dim and shape and shape[-1] == 1:
      shape.pop()
    return core.ShapedArray(tuple(shape), val.dtype)
  return tuple(maybe_squeeze_abstract_eval(v, expected_dim) for v in val)


def maybe_squeeze_ir(val: ir.Value, expected_dim: int) -> ir.Value:
  """Squeezes trailing dimensions of size 1 until rank matches expected_dim."""
  tensor_type = ir.RankedTensorType(val.type)
  shape = list(tensor_type.shape)
  dtype = tensor_type.element_type
  changed = False
  while len(shape) > expected_dim and shape and shape[-1] == 1:
    shape.pop()
    changed = True
  if changed:
    target_type = ir.RankedTensorType.get(shape, dtype)
    return hlo.reshape(target_type, val)
  return val
