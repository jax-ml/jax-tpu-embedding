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

import enum


class SlotShape(enum.Enum):
  """Shape of a single slot variable, relative to its embedding table.

  `slot_shapes` holds one of these per slot variable, ordered to match
  `slot_variables_initializers()`. Note that the outer sequence indexes slot
  variables and is not itself a shape: for a two-slot optimizer over a
  (32, 8) table, `(SlotShape.ROWWISE_1D, SlotShape.TABLE)` means a (32,)
  first slot and a (32, 8) second slot.

  Attributes:
    TABLE: Same shape as the table, `(vocab, embedding_dim)`.
    ROWWISE_1D: One value per table row, `(vocab,)`.
  """

  TABLE = "table"
  ROWWISE_1D = "rowwise_1d"
