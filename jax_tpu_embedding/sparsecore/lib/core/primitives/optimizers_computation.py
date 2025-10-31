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
"""Defines common optimizers for embedding lookups."""

from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func as func_dialect
from jax._src.lib.mlir.dialects import hlo
from jax.interpreters import mlir


def sgd(
    ctx: mlir.LoweringRuleContext, computation_name: str, dim_size: int
) -> None:
  """A Callable SGD lowering."""
  optimizer_update = func_dialect.FuncOp(
      computation_name,
      (
          [
              ir.RankedTensorType.get(
                  [1, dim_size],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(
                  [1, dim_size],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(
                  [1, dim_size],
                  ir.F32Type.get(),
              ),
          ],
          [
              ir.TupleType.get_tuple(
                  [
                      ir.RankedTensorType.get(
                          [1, dim_size],
                          ir.F32Type.get(),
                      )
                  ]
              ),
          ],
      ),
      ip=ctx.module_context.ip,
      visibility="private",
  )
  entry_block = optimizer_update.add_entry_block()
  with ir.InsertionPoint(entry_block):
    # lr * grad
    gradient_update = hlo.multiply(
        entry_block.arguments[0],
        entry_block.arguments[2],
    )
    # updated_embedding_table = embedding_table - lr * grad
    updated_embedding_table = hlo.subtract(
        entry_block.arguments[1], gradient_update
    )
    updated_embedding_tables = hlo.tuple([updated_embedding_table])
    func_dialect.ReturnOp([updated_embedding_tables])


def adagrad(
    ctx: mlir.LoweringRuleContext, computation_name: str, dim_size: int
) -> None:
  """A callable Adagrad lowering.

  When using this optimizer, the expected ordering of the embedding variables is
    0. embedding_table
    1. accumulator

  Args:
    ctx: The lowering rule context.
    computation_name: The name of the computation.
    dim_size: The dimension of the embedding table.
  """
  optimizer_update = func_dialect.FuncOp(
      computation_name,
      (
          [
              ir.RankedTensorType.get(
                  [1, dim_size],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(
                  [1, dim_size],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(
                  [1, dim_size],
                  ir.F32Type.get(),
              ),
              ir.RankedTensorType.get(
                  [1, dim_size],
                  ir.F32Type.get(),
              ),
          ],
          [
              ir.TupleType.get_tuple([
                  ir.RankedTensorType.get(
                      [1, dim_size],
                      ir.F32Type.get(),
                  ),
                  ir.RankedTensorType.get(
                      [1, dim_size],
                      ir.F32Type.get(),
                  ),
              ]),
          ],
      ),
      ip=ctx.module_context.ip,
      visibility="private",
  )

  entry_block = optimizer_update.add_entry_block()
  with ir.InsertionPoint(entry_block):
    # new_accumulator = accumulator + grad * grad
    grad_squared = hlo.multiply(
        entry_block.arguments[0],
        entry_block.arguments[0],
    )
    new_accumulator = hlo.add(
        entry_block.arguments[2],
        grad_squared,
    )
    updated_embedding_table = hlo.subtract(
        entry_block.arguments[1],
        hlo.divide(
            hlo.multiply(
                entry_block.arguments[3],
                entry_block.arguments[0],
            ),
            hlo.sqrt(new_accumulator),
        ),
    )
    updated_embedding_tables = hlo.tuple(
        [updated_embedding_table, new_accumulator]
    )
    func_dialect.ReturnOp([updated_embedding_tables])
