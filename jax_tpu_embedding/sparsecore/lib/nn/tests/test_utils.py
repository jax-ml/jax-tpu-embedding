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
"""Test utils for JAX SparseCore."""

from typing import Sequence

from absl.testing import absltest
import einops
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
import numpy as np


def round_up_to_multiple(number: int, factor: int) -> int:
  """Returns the next largest multiple of factor greater than number."""
  return number if number % factor == 0 else (number // factor + 1) * factor


def row_id_initializer(
    shape: Sequence[int],
    dtype: jax.typing.DTypeLike = jnp.float32,
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
      jnp.ones(shape, dtype=dtype)
      * jnp.arange(offset, offset + shape[0], dtype=dtype)[:, None]
  )


def formatted_array2string(arr: jax.Array) -> str:
  """Force float-like values to be formatted with 6 decimal places."""
  return np.array2string(arr, formatter={"float_kind": lambda x: "%.6f" % x})


def row_col_id_initializer_value(
    leading_value: int, row: int, col: int
) -> jnp.float32:
  """Returns the value for row_col_id_initializer."""
  return leading_value + row / 1000.0 + col / 1000000.0


def row_col_id_initializer(
    leading_value: int = 0,
) -> jax.nn.initializers.Initializer:
  """Initializes a table with leading value + row id/1000 + col id/1000000."""

  def create_array(leading_value, shape):
    rows, cols = shape
    result = jnp.zeros(shape, dtype=jnp.float32)
    for i in range(rows):
      for j in range(cols):
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


def row_id_with_offset_initializer_value(
    offset_value: int, row: int
) -> jnp.float32:
  """Returns the value for row_col_id_initializer."""
  return offset_value + row


def row_id_with_offset_initializer(
    offset_value: int = 0,
) -> jax.nn.initializers.Initializer:
  """Initializes a table with offset value + row id."""

  def create_array(offset_value, shape):
    rows, cols = shape
    result = jnp.zeros(shape, dtype=jnp.float32)
    for i in range(rows):
      for j in range(cols):
        result = result.at[i, j].set(
            row_id_with_offset_initializer_value(offset_value, i)
        )

    return result

  def init(key, shape) -> jax.Array:
    del key
    return create_array(offset_value, shape)

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
  return jnp.roll(embedding_table, shift=rotation, axis=0)


def create_per_device_sharded_stacked_tables(
    emb_tables: Sequence[jax.Array],
    num_devices: int,
    num_sparsecore_per_device: int,
    rotation: int | None,
) -> jax.Array:
  """Creates a array of stacked shards, one per device.

  Args:
    emb_tables: List of embedding tables that are to be sharded and stacked.
    num_devices: Number of devices for which the stacked shards are created.
    num_sparsecore_per_device: Number of sparse core there are on each device.
    rotation: An int rotation that should be applied between tables when
      stacking.  If None, defaults to num_sparsecore_per_device.  Default: None.

  Returns:
    A jax.Array for shards on devices.
  """
  rotation = rotation if rotation is not None else num_sparsecore_per_device
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
  rotations = np.arange(0, len(emb_tables)) * rotation
  rotated_tables = [
      rotate_sharded_table(emb_table, rot)
      for emb_table, rot in zip(mod_sharded_tables, rotations)
  ]
  sharded_stacked = jnp.concatenate(rotated_tables, axis=1)

  return sharded_stacked.reshape(num_devices, -1, dim)


def skip_if_tpu_unavailable(f):
  """Skips the test if TPU initialization fails.

  See http://b/366070551#comment3 for more information.

  Args:
    f: The test function to wrap.

  Returns:
    A wrapper function that skips the test if TPU initialization fails.
  """
  def wrapper(*args, **kwargs):
    # Handle silent fallback to CPU.
    if jax.default_backend() == "cpu":
      raise absltest.SkipTest("TPU not available")
    try:
      return f(*args, **kwargs)
    except RuntimeError as e:
      if "TPU initialization failed" in str(e):
        raise absltest.SkipTest("TPU not available")
      raise e

  return wrapper


def row_initialize_with_padding(
    table_spec: embedding_spec.TableSpec,
    offset: int = 0,
    pad_value: float = -1,
) -> jax.Array:
  """Initializes an embedding table with padding."""
  shape = (
      table_spec.vocabulary_size,
      table_spec.embedding_dim,
  )
  padded_shape = (
      table_spec.setting_in_stack.padded_vocab_size,
      table_spec.setting_in_stack.padded_embedding_dim,
  )
  array = row_id_initializer(shape=shape, offset=offset)
  paddings = tuple((0, y - x) for x, y in zip(shape, padded_shape))
  return np.pad(array, paddings, mode="constant", constant_values=pad_value)
