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
"""StableHLO lowering utilities and primitives for custom SparseCore optimizers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.extend as jex
from jax.extend.mlir import ir
from jax.extend.mlir.dialects import func as func_dialect
from jax.interpreters import mlir
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.nn import slot_shape

_call_stablehlo_p = jex.core.Primitive("call_stablehlo")
_call_stablehlo_p.multiple_results = True


def _call_stablehlo_abstract_eval(*args, out_avals, **kwargs):
  del args, kwargs
  return tuple(out_avals)


_call_stablehlo_p.def_abstract_eval(_call_stablehlo_abstract_eval)


def _call_stablehlo_lowering(ctx, *args, stablehlo_str, out_avals):
  """Lowers the _call_stablehlo_p primitive by calling the merged submodule."""
  submodule = ir.Module.parse(stablehlo_str, context=ctx.module_context.context)
  callee_name = mlir.merge_mlir_modules(
      ctx.module_context.module,
      "orig_custom_optimizer",
      submodule,
      dst_symtab=ctx.module_context.symbol_table,
  )
  result_types = [
      mlir.aval_to_ir_type(ctx.module_context, aval) for aval in out_avals
  ]

  call = func_dialect.CallOp(
      result_types,
      ir.FlatSymbolRefAttr.get(callee_name),
      list(args),
  )
  return tuple(call.results)


mlir.register_lowering(_call_stablehlo_p, _call_stablehlo_lowering)


def apply_clipping(
    val: jax.Array,
    min_value: float | None = None,
    max_value: float | None = None,
) -> jax.Array:
  """Clips val according to min_value and/or max_value if provided."""
  if min_value is not None and max_value is not None:
    return jnp.clip(val, min_value, max_value)
  if min_value is not None:
    return jnp.maximum(val, min_value)
  if max_value is not None:
    return jnp.minimum(val, max_value)
  return val


def _get_optimizer_avals(
    embedding_dim: int,
    num_slot_variables: int,
    num_hyperparameters: int,
    slot_shapes: Sequence[slot_shape.SlotShapeType] | None = None,
) -> tuple[list[jax.core.ShapedArray], tuple[jax.core.ShapedArray, ...]]:
  """Constructs input and output abstract arrays for custom optimizer lowering."""
  table_aval = jax.core.ShapedArray((1, embedding_dim), jnp.float32)
  if slot_shapes is not None:
    if len(slot_shapes) != num_slot_variables:
      raise ValueError(
          f"Length of slot_shapes ({len(slot_shapes)}) must match"
          f" num_slot_variables ({num_slot_variables})."
      )
    slot_avals = []
    for s in slot_shapes:
      if s in ("rowwise_1d", "1d"):
        slot_avals.append(jax.core.ShapedArray((1,), jnp.float32))
      elif s is None or s == "table":
        slot_avals.append(table_aval)
      elif isinstance(s, int):
        slot_avals.append(jax.core.ShapedArray((1, s), jnp.float32))
      else:
        raise ValueError(f"Unsupported slot shape specification: {s}")
  else:
    slot_avals = [table_aval] * num_slot_variables

  hyperparam_avals = [table_aval] * num_hyperparameters
  in_avals = [table_aval, table_aval, *slot_avals, *hyperparam_avals]
  out_avals = (table_aval, *slot_avals)
  return in_avals, out_avals


def lower_to_stablehlo(
    custom_computation_fn: Callable[..., Any],
    embedding_dim: int = 1,
    num_slot_variables: int = 0,
    num_hyperparameters: int = 1,
    min_value: float | None = None,
    max_value: float | None = None,
    slot_shapes: Sequence[slot_shape.SlotShapeType] | None = None,
) -> str:
  """Traces and lowers a custom optimizer function into StableHLO text.

  Args:
    custom_computation_fn: The custom optimizer callable accepting gradient,
      embedding table, slot variables, learning rate, and hyperparameters.
    embedding_dim: Dimension of the embedding table.
    num_slot_variables: Number of slot variables in the optimizer state.
    num_hyperparameters: Number of hyperparameters passed to the optimizer.
    min_value: Optional minimum value to clip the updated embedding table.
    max_value: Optional maximum value to clip the updated embedding table.
    slot_shapes: Optional shape specification per slot variable. Defaults to
      table-shaped slots. See `slot_shape.SlotShapeType`.

  Returns:
    A string containing the lowered StableHLO module text.
  """
  in_avals, _ = _get_optimizer_avals(
      embedding_dim=embedding_dim,
      num_slot_variables=num_slot_variables,
      num_hyperparameters=num_hyperparameters,
      slot_shapes=slot_shapes,
  )

  fn_to_lower = custom_computation_fn
  if min_value is not None or max_value is not None:
    orig_fn = fn_to_lower

    def _clipped_fn(*args):
      out = orig_fn(*args)
      if isinstance(out, (list, tuple)):
        clipped_table = apply_clipping(out[0], min_value, max_value)
        return (clipped_table, *out[1:])
      else:
        return apply_clipping(out, min_value, max_value)

    fn_to_lower = _clipped_fn

  lowered_opt = jax.jit(fn_to_lower).lower(*in_avals)
  return lowered_opt.as_text(dialect="stablehlo")


def wrap_stablehlo_with_limits(
    stablehlo: str | bytes,
    embedding_dim: int,
    num_slot_variables: int,
    num_hyperparameters: int,
    min_value: float | None = None,
    max_value: float | None = None,
    slot_shapes: Sequence[slot_shape.SlotShapeType] | None = None,
) -> str:
  """Wraps an existing StableHLO module in JAX and lowers again with limits.

  Args:
    stablehlo: The StableHLO module as a string or bytes.
    embedding_dim: Dimension of the embedding table.
    num_slot_variables: Number of slot variables in the optimizer state.
    num_hyperparameters: Number of hyperparameters passed to the optimizer.
    min_value: Optional minimum value to clip the updated embedding table.
    max_value: Optional maximum value to clip the updated embedding table.
    slot_shapes: Optional shape specification per slot variable. Defaults to
      table-shaped slots. See `slot_shape.SlotShapeType`.

  Returns:
    A string containing the wrapped and lowered StableHLO module text.
  """
  stablehlo_str = (
      stablehlo.decode("utf-8") if isinstance(stablehlo, bytes) else stablehlo
  )
  if min_value is None and max_value is None:
    return stablehlo_str

  in_avals, out_avals = _get_optimizer_avals(
      embedding_dim=embedding_dim,
      num_slot_variables=num_slot_variables,
      num_hyperparameters=num_hyperparameters,
      slot_shapes=slot_shapes,
  )

  def _wrapped(*args):
    out = _call_stablehlo_p.bind(
        *args,
        stablehlo_str=stablehlo_str,
        out_avals=out_avals,
    )
    clipped_table = apply_clipping(out[0], min_value, max_value)
    return (clipped_table, *out[1:])

  lowered = jax.jit(_wrapped).lower(*in_avals)
  return lowered.as_text(dialect="stablehlo")
