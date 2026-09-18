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
"""Shape specifications for optimizer slot variables.

This module is intentionally dependency-free so that it can be shared by
`embedding_spec` (a pure data-specification module) and
`custom_optimizer_lowering` (which registers MLIR lowerings at import time)
without either having to import the other.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

SlotShapeType: TypeAlias = Literal["table", "rowwise_1d", "1d"] | int | None
"""Shape of a single slot variable, relative to its embedding table.

  - "table" (or None): same shape as the table, (vocab, embedding_dim).
  - "rowwise_1d" (or "1d"): one value per table row, (vocab,).
  - int: 2D with a custom width, (vocab, int).

`slot_shapes` holds one of these per slot variable, ordered to match
`slot_variables_initializers()`. Note that the outer sequence indexes slot
variables and is not itself a shape: for an Adam-style optimizer over a
(32, 8) table, `("rowwise_1d", 4)` means a (32,) momentum and a (32, 4)
velocity.
"""
