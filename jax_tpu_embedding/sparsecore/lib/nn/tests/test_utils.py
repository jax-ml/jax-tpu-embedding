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
"""Test utils for Jax Sparsecore."""

from typing import Sequence

import einops
import jax
import numpy

NUM_SC_PER_DEVICE = 4


def row_id_initializer(
    shape: Sequence[int],
    dtype: jax.typing.DTypeLike = jax.numpy.float32,
    offset: int = 0,
) -> jax.Array:
  """An initializer for an array where row values are function of row id.

  Used for testing.

  Args:
   shape: Shape of a jax.Array to be initialized.
   dtype: type of jax.Array.
   offset: An int offset added to all values in the jax.Array.

  Returns:
   A jax.Array for testing.
  """
  return (
      jax.numpy.ones(shape, dtype=dtype)
      * jax.numpy.arange(offset, offset + shape[0], dtype=dtype)[:, None]
  )


def formatted_array2string(arr: jax.Array) -> str:
  """Force float-like values to be formatted with 6 decimal places."""
  return numpy.array2string(arr, formatter={"float_kind": lambda x: "%.6f" % x})


def row_col_id_initializer_value(
    leading_value: int, row: int, col: int
) -> jax.numpy.float32:
  """Returns the value for row_col_id_initializer."""
  return leading_value + row / 1000.0 + col / 1000000.0


def row_col_id_initializer(
    leading_value: int = 0,
) -> jax.nn.initializers.Initializer:
  """Initializes a table with leading value + row id/1000 + col id/1000000."""

  def create_array(leading_value, shape):
    rows, cols = shape
    result = jax.numpy.zeros(shape, dtype=jax.numpy.float32)
    for i in range(rows):
      for j in range(cols):
        if j == 0:
          # Column 0 is process ID.
          col_value = jax.process_index()
        else:
          col_value = j
        result = result.at[i, j].set(
            row_col_id_initializer_value(leading_value, i, col_value)
        )

    return result

  def init(key, shape) -> jax.Array:
    del key
    # We should have 7 digits for fp32
    assert leading_value <= 9  # 1 digit for leading value
    assert len(shape) == 2
    assert shape[0] <= 999  # 3 digits for rows
    assert shape[1] <= 999  # 3 digits for cols
    return create_array(leading_value, shape)

  return init


def rotate_sharded_table(
    embedding_table: jax.Array, rotation: int
) -> jax.Array:
  """Return the rotated version of the input jax.Array.

  Args:
   embedding_table: jax.Array for the embedding table to be rotated.
   rotation: int value for amount the embedding table should be rotated.

  Return:
   A rotated jax.Array.
  """
  return jax.numpy.roll(embedding_table, shift=rotation, axis=0)


def create_per_device_sharded_stacked_tables(
    emb_tables: Sequence[jax.Array],
    num_devices: int,
    num_sparsecore_per_device: int = 4,
    rotation: int = 4,
) -> jax.Array:
  """Creates a array of stacked shards, one per device.

  Args:
    emb_tables: List of embedding tables that are to be sharded and stacked.
    num_devices: Number of devices for which the stacked shards are created.
    num_sparsecore_per_device: Number of sparse core there are on each device.
    rotation: An int rotation that should be applied between tables when
      stacking.

  Returns:
    A jax.Array for shards on devices.
  """
  dim = emb_tables[0].shape[1]
  assert all(table.shape[1] == dim for table in emb_tables)
  mod_sharded_tables = [
      einops.rearrange(
          emb_table,
          "(v t) f -> t (v) f",
          t=num_devices * num_sparsecore_per_device,
      )
      for emb_table in emb_tables
  ]
  rotations = numpy.arange(0, len(emb_tables)) * rotation
  rotated_tables = [
      rotate_sharded_table(emb_table, rot)
      for emb_table, rot in zip(mod_sharded_tables, rotations)
  ]
  sharded_stacked = jax.numpy.concatenate(rotated_tables, axis=1)

  return sharded_stacked.reshape(num_devices, -1, dim)
