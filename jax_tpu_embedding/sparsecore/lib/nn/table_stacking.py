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
"""Methods for table stacking."""

import collections
from collections.abc import Mapping
import hashlib
import typing
from typing import Any, Callable, Sequence, TypeAlias, TypeVar

from absl import logging
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.lib.nn.embedding_spec import StackedTableSpec
from jax_tpu_embedding.sparsecore.lib.nn.embedding_spec import TableSpec
from jax_tpu_embedding.sparsecore.lib.proto import embedding_spec_pb2
import numpy as np


T: TypeAlias = TypeVar("T")
Nested: TypeAlias = T | Sequence[T] | Mapping[str, T]
LimitsCallable: TypeAlias = Callable[[str, int], int]
# Any to support tf.Ragged without needing an explicit TF dependency.
ArrayLike: TypeAlias = jax.Array | np.ndarray | Any  # type: ignore
Shape: TypeAlias = tuple[int, ...]


def _next_largest_multiple(value: int, multiple: int) -> int:
  return ((value + multiple - 1) // multiple) * multiple


def _default_stacked_table_spec(
    table_spec: TableSpec, num_shards: int, batch_size: int
) -> StackedTableSpec:
  return StackedTableSpec(
      stack_name=table_spec.name,
      stack_vocab_size=_next_largest_multiple(
          table_spec.vocabulary_size, 8 * num_shards
      ),
      stack_embedding_dim=_next_largest_multiple(table_spec.embedding_dim, 8),
      optimizer=table_spec.optimizer,
      combiner=table_spec.combiner,
      total_sample_count=batch_size,
      max_ids_per_partition=table_spec.max_ids_per_partition,
      max_unique_ids_per_partition=table_spec.max_unique_ids_per_partition,
  )


def _get_stacked_table_spec(
    table_spec: TableSpec, num_shards: int, batch_size: int = 0
) -> StackedTableSpec:
  return table_spec.stacked_table_spec or _default_stacked_table_spec(
      table_spec, num_shards, batch_size
  )


def _pad_table(
    table_spec: TableSpec,
    table_values: jax.Array,
    num_shards: int,
    pad_value: jnp.float32 = jnp.nan,
) -> jax.Array:
  """Adds appropriate padding to a table to prepare for stacking.

  Args:
      table_spec: Table specification describing the table to pad.
      table_values: Table values array to pad.
      num_shards: Number of shards in the table (typically `global_device_count
        * num_sc_per_device`).
      pad_value: Value to use for padding.

  Returns:
      Padded table values.
  """
  vocabulary_size = table_spec.vocabulary_size
  embedding_dim = table_spec.embedding_dim
  padded_vocabulary_size = _next_largest_multiple(
      vocabulary_size, 8 * num_shards
  )
  stack_embedding_dim = _get_stacked_table_spec(
      table_spec, num_shards
  ).stack_embedding_dim
  return jnp.pad(
      table_values,
      (
          (0, padded_vocabulary_size - vocabulary_size),
          (0, stack_embedding_dim - embedding_dim),
      ),
      constant_values=pad_value,
  )


def _stack_and_shard_table(
    stacked_table: jax.Array,
    table_spec: TableSpec,
    table: jax.Array,
    num_shards: int,
    pad_value: jnp.float32,
) -> jax.Array:
  """Stacks and shards a single table for use in sparsecore lookups."""
  padded_values = _pad_table(table_spec, table, num_shards, pad_value)
  sharded_padded_vocabulary_size = padded_values.shape[0] // num_shards
  stack_embedding_dim = stacked_table.shape[-1]

  # Mod-shard vocabulary across devices.
  sharded_values = jnp.swapaxes(
      padded_values.reshape(-1, num_shards, stack_embedding_dim),
      0,
      1,
  )

  # Rotate shards.
  setting_in_stack = table_spec.setting_in_stack
  rotated_values = jnp.roll(
      sharded_values, setting_in_stack.shard_rotation, axis=0
  )

  # Insert table into the stack.
  table_row = setting_in_stack.row_offset_in_shard
  output_stacked_table = stacked_table.at[
      :, table_row : (table_row + sharded_padded_vocabulary_size), :
  ].set(rotated_values)

  return output_stacked_table


def stack_and_shard_tables(
    table_specs: Nested[TableSpec],
    tables: Nested[ArrayLike],
    num_shards: int,
    pad_value: jnp.float32 = jnp.nan,
) -> dict[str, Nested[jax.Array]]:
  """Stacks and shards tables for use in sparsecore lookups.

  Args:
      table_specs: Nested collection of unstacked table specifications.
      tables: Table values corresponding to the table_specs.
      num_shards: Number of shards in the table (typically `global_device_count
        * num_sc_per_device`).
      pad_value: Value to use for padding.

  Returns:
      A mapping of stacked table names to stacked table values.
  """

  # Gather stacked table information.
  stacked_table_map: dict[
      str,
      tuple[StackedTableSpec, list[TableSpec]],
  ] = {}

  def collect_stacked_tables(table_spec: TableSpec) -> None:
    stacked_table_spec = _get_stacked_table_spec(table_spec, num_shards)
    stacked_table_name = stacked_table_spec.stack_name
    if stacked_table_name not in stacked_table_map:
      stacked_table_map[stacked_table_name] = (stacked_table_spec, [])
    stacked_table_map[stacked_table_name][1].append(table_spec)

  jax.tree.map(collect_stacked_tables, table_specs)

  table_map: dict[str, Nested[jax.Array]] = {}

  def collect_tables(table_spec: TableSpec, table: Nested[jax.Array]) -> None:
    table_map[table_spec.name] = table

  jax.tree.map(collect_tables, table_specs, tables)

  stacked_tables: dict[str, Nested[jax.Array]] = {}
  for (
      stacked_table_spec,
      table_specs,
  ) in stacked_table_map.values():
    stack_vocab_size = stacked_table_spec.stack_vocab_size
    sharded_vocab_size = stack_vocab_size // num_shards
    stack_embedding_dim = stacked_table_spec.stack_embedding_dim

    # Allocate initial buffer.  The stacked table will be divided among
    # shards by splitting the vocabulary dimension:
    #   [ v, e ] -> [s, v/s, e]
    stacked_table_tree = jax.tree.map(
        lambda _, sharded_vocab_size=sharded_vocab_size, stack_embedding_dim=stack_embedding_dim: jnp.zeros(
            shape=(num_shards, sharded_vocab_size, stack_embedding_dim),
            dtype=jnp.float32,
        ),
        table_map[table_specs[0].name],
    )

    for table_spec in table_specs:
      table_tree = table_map[table_spec.name]
      stacked_table_tree = jax.tree.map(
          lambda stacked_table, table, table_spec=table_spec: _stack_and_shard_table(
              stacked_table,
              table_spec,
              table,
              num_shards,
              pad_value,
          ),
          stacked_table_tree,
          table_tree,
      )

    stacked_tables[stacked_table_spec.stack_name] = stacked_table_tree

  return stacked_tables


# jax.jit with in_shardings only supports positional arguments, so these
# arguments cannot be keyword-only.
def _unshard_and_unstack_single_table(
    stacked_table_shard: jax.Array,
    sharded_vocab_size: int,
    row_offset_in_shard: int,
    shard_rotation: int,
    vocab_size: int,
    embedding_dim: int,
) -> jax.Array:
  """Un-rotates, un-mod-shards, and un-pads a table shard."""
  stack_embedding_dim = stacked_table_shard.shape[-1]
  # Extract padded values from the stacked table.
  padded_values = stacked_table_shard[
      :, row_offset_in_shard : (row_offset_in_shard + sharded_vocab_size), :
  ]
  # Un-rotate shards.
  padded_values = jnp.roll(padded_values, -shard_rotation, axis=0)
  # Un-mod-shard.
  padded_values = jnp.swapaxes(padded_values, 0, 1).reshape(
      -1, stack_embedding_dim
  )
  # Un-pad.
  return padded_values[:vocab_size, :embedding_dim]


def _unshard_and_unstack_table(
    table_spec: TableSpec,
    stacked_table_tree: Nested[jax.Array],
    num_shards: int,
) -> Nested[jax.Array]:
  """Unshards and unstacks a single table."""
  sharded_vocabulary_size = (
      _next_largest_multiple(table_spec.vocabulary_size, 8 * num_shards)
      // num_shards
  )
  setting_in_stack = table_spec.setting_in_stack

  def _unshard_leaf(stacked_table: jax.Array) -> jax.Array:
    stacked_table = stacked_table.reshape(
        num_shards, -1, stacked_table.shape[-1]
    )
    return _unshard_and_unstack_single_table(
        stacked_table,
        sharded_vocab_size=sharded_vocabulary_size,
        row_offset_in_shard=setting_in_stack.row_offset_in_shard,
        shard_rotation=setting_in_stack.shard_rotation,
        vocab_size=table_spec.vocabulary_size,
        embedding_dim=table_spec.embedding_dim,
    )

  return jax.tree_util.tree_map(
      _unshard_leaf,
      stacked_table_tree,
  )


def _get_stack_name_from_table_spec(table_spec: TableSpec) -> str:
  """Returns the stack name associated with a single TableSpec."""
  if table_spec.stacked_table_spec:
    return table_spec.stacked_table_spec.stack_name
  return table_spec.name


def unshard_and_unstack_tables(
    table_specs: Nested[TableSpec],
    stacked_tables: Mapping[str, Nested[jax.Array]],
    num_shards: int,
) -> Nested[jax.Array]:
  """Unshards and unstacks a collection of tables.

  Args:
      table_specs: Nested collection of unstacked table specifications.
      stacked_tables: Mapping of stacked table names to stacked table values.
      num_shards: Number of shards in the table (typically `global_device_count
        * num_sc_per_device`).

  Returns:
      A mapping of table names to unstacked table values.
  """
  return jax.tree.map(
      lambda table_spec: _unshard_and_unstack_table(
          table_spec,
          stacked_tables[_get_stack_name_from_table_spec(table_spec)],
          num_shards,
      ),
      table_specs,
  )


def get_table_stacks(
    table_specs: Nested[TableSpec],
) -> dict[str, list[TableSpec]]:
  """Extracts lists of tables that are stacked together.

  Args:
      table_specs: Nested collection of table specifications.

  Returns:
      A mapping of stacked table names to lists of table specifications for
      each stack.
  """
  stacked_table_specs: dict[str, list[TableSpec]] = collections.defaultdict(
      list
  )
  flat_table_specs, _ = jax.tree.flatten(table_specs)
  for table_spec in flat_table_specs:
    table_spec = typing.cast(TableSpec, table_spec)
    stacked_table_spec = table_spec.stacked_table_spec
    if stacked_table_spec is not None:
      stacked_table_specs[stacked_table_spec.stack_name].append(table_spec)
    else:
      stacked_table_specs[table_spec.name].append(table_spec)

  return stacked_table_specs


def get_default_limits(name: str, batch_size: int) -> int:
  """A default implementation of limits for embedding tables.

  Args:
   name: Name of the embedding table
   batch_size: Vocabulary size of the embedding table

  Returns:
   Default limit(256)
  """
  del name, batch_size
  return 256


def round_up_dim_and_vocab_size(
    tables: Mapping[str, embedding_spec.TableSpec], num_sc: int
) -> tuple[Mapping[str, int], Mapping[str, int]]:
  """Rounds up the embedding dim and vocab size of the tables.

  The embedding dim is rounded up to the next largest multiple of 8.
  The vocab size is rounded up to the next largest multiple of 8 * num_sc.
  Args:
    tables: The tables to round up.
    num_sc: The number of sparsecores.

  Returns:
    A tuple of mappings from table name to the rounded up embedding dim and to
    vocab size.
  """
  table_to_padded_dim = {
      n: _next_largest_multiple(spec.embedding_dim, 8)
      for (n, spec) in tables.items()
  }
  table_to_padded_vocab_size = {
      n: _next_largest_multiple(spec.vocabulary_size, 8 * num_sc)
      for (n, spec) in tables.items()
  }
  return table_to_padded_dim, table_to_padded_vocab_size


def _get_stack_table_names(
    num_sc: int,
    flatten_tables: Mapping[str, embedding_spec.TableSpec],
    flatten_features: Sequence[embedding_spec.FeatureSpec],
    activation_mem_bytes_limit: int,
) -> Sequence[Sequence[str]]:
  """Returns the stack groups for the tables based on their specs."""
  original_table_names = set(flatten_tables.keys())

  table_to_padded_dim, _ = round_up_dim_and_vocab_size(flatten_tables, num_sc)
  table_name_map = collections.defaultdict(list)
  for table_name, dim in table_to_padded_dim.items():
    key = (
        dim,
        flatten_tables[table_name].optimizer,
        flatten_tables[table_name].combiner,
    )
    table_name_map[key].append(table_name)

  groups = list(table_name_map.values())

  # Calculate sample_count per sparsecore for each table.
  table_to_sample_count = collections.defaultdict(int)
  for feature in flatten_features:
    table_to_sample_count[feature.table_spec.name] += int(
        np.prod(feature.output_shape[:-1]) // num_sc
    )

  # Calculate and register the activation memory usage of this table.
  table_to_activation_mem_bytes = {
      tname: table_to_padded_dim[tname] * table_to_sample_count[tname] * 4
      for tname in flatten_tables.keys()
  }

  validated_groups = []
  for group in groups:
    # A list of groups that are split from the current group.
    split_groups = []
    # Iterate through all tables in the current group.
    for table_name in group:
      found = False
      # Iterate through all candidate groups to check if the table can be
      # joined.
      for candidate_group in split_groups:
        accumuated_activation_mem_bytes = 0
        # Sum up the activation memory usage of all tables in this candidate
        # group. We re-calculate because the tables could have been added into
        # the group in the previous iteration.
        for candidate_table in candidate_group:
          accumuated_activation_mem_bytes += table_to_activation_mem_bytes[
              candidate_table
          ]
        # Check for limit violation.
        if (
            accumuated_activation_mem_bytes
            + table_to_activation_mem_bytes[table_name]
        ) <= activation_mem_bytes_limit:
          # Append to this candidate group if no limit violation.
          candidate_group.append(table_name)
          found = True
          break
      if not found:
        # If the table cannot be joined with any existing group, create a new
        # group with only this table.
        split_groups.append([table_name])
        if split_groups:
          logging.info(
              "Table %s cannot be joined with any existing group, create a new"
              " one.",
              table_name,
          )
    # Add into the validated groups.
    validated_groups.extend(split_groups)

  grouped_table_names = set()
  for group in validated_groups:
    grouped_table_names.update(group)

  if original_table_names != grouped_table_names:
    raise ValueError(
        "Table names are not grouped correctly. Original table names:"
        f" {original_table_names}, grouped table names: {grouped_table_names}"
    )

  return validated_groups


def _verify_stack_tables(
    stack_name: str,
    table_names: Sequence[str],
    features: Sequence[embedding_spec.FeatureSpec],
    tables: Mapping[str, embedding_spec.TableSpec],
):
  """Verifies that the provided stacking groups are valid."""
  logging.vlog(
      2,
      "Verifying stack group: %s with tables: %s",
      stack_name,
      table_names,
  )

  # Check that each table is not stacked already.
  def _is_stacked_already(table: embedding_spec.TableSpec):
    if table.setting_in_stack is None:
      return False
    return table.setting_in_stack.stack_name != table.name

  for tname in table_names:
    if _is_stacked_already(tables[tname]):
      raise ValueError(
          f"Table {tname} is already stacked in group"
          f" {tables[tname].setting_in_stack.stack_name}."
      )

  for feature in features:
    if (
        _is_stacked_already(feature.table_spec)
        and feature.table_spec.setting_in_stack.stack_name == stack_name
    ):
      raise ValueError(
          f"Cannot use stack name {stack_name} since it's already used."
      )

  # A table should not be repeated in a group.
  counter = collections.Counter(table_names)
  for table, count in counter.items():
    if count > 1:
      raise ValueError(f"Table {table} is repeated in group {stack_name}.")

  # All tables in a group should have same optimizer.
  if not all([
      tables[name].optimizer == tables[table_names[0]].optimizer
      for name in table_names
  ]):
    raise ValueError(
        f"Tables {table_names} in group {stack_name} have different optimizers."
    )
  # All tables in a group should have same combiner.
  if not all([
      tables[t].combiner == tables[table_names[0]].combiner for t in table_names
  ]):
    raise ValueError(
        f"Tables {table_names} in group {stack_name} have different combiners."
    )


def _compute_table_to_setting_in_stack(
    stack_name: str,
    table_names: Sequence[str],
    padded_embedding_dim: int,
    table_to_padded_vocab_size: Mapping[str, int],
    global_device_count: int,
    num_sc_per_device: int,
    rotation: int,
) -> Mapping[str, embedding_spec.TableSettingInStack]:
  """Returns the table to setting in stack mapping."""
  table_name_to_setting_in_stack = {}
  row_offset_in_shard = 0
  shard_rotation = 0
  num_sc = num_sc_per_device * global_device_count
  for tname in table_names:
    if tname not in table_to_padded_vocab_size:
      raise ValueError(f"Padded vocab size for Table {tname} is missing.")

    num_rows_in_shard = table_to_padded_vocab_size[tname] // num_sc
    setting_in_stack = embedding_spec.TableSettingInStack(
        stack_name=stack_name,
        padded_vocab_size=table_to_padded_vocab_size[tname],
        padded_embedding_dim=padded_embedding_dim,
        row_offset_in_shard=row_offset_in_shard,
        shard_rotation=shard_rotation,
    )
    row_offset_in_shard += num_rows_in_shard
    # Rotate the shard by num_sc_per_device and then bound by the
    # total number of sparsecores.
    shard_rotation = (shard_rotation + rotation) % (num_sc)
    table_name_to_setting_in_stack[tname] = setting_in_stack
    logging.info("Table %s has setting in stack: %s", tname, setting_in_stack)
  return table_name_to_setting_in_stack


def _get_limits_for_stack(
    table_names: Sequence[str],
    table_name_to_feature_spec: Mapping[str, embedding_spec.FeatureSpec],
    default_max_ids_per_partition: int,
    default_max_unique_ids_per_partition: int,
) -> tuple[int, int]:
  """Returns the max_ids_per_partition and max_unique_ids_per_partition."""
  if len(table_names) == 1:
    # If the stack has only one table, then use the max_ids_per_partition of
    # that table's spec.
    return (
        table_name_to_feature_spec[
            table_names[0]
        ].table_spec.max_ids_per_partition,
        table_name_to_feature_spec[
            table_names[0]
        ].table_spec.max_unique_ids_per_partition,
    )
  return (
      default_max_ids_per_partition,
      default_max_unique_ids_per_partition,
  )


def _stack_feature_specs(
    stack_name: str,
    features: Nested[embedding_spec.FeatureSpec],
    table_names: Sequence[str],
    padded_embedding_dim: int,
    table_to_padded_vocab_size: Mapping[str, int],
    global_device_count: int,
    num_sc_per_device: int,
    rotation: int,
    stack_to_max_ids_per_partition: LimitsCallable = get_default_limits,
    stack_to_max_unique_ids_per_partition: LimitsCallable = get_default_limits,
) -> None:
  """Updated the feature spec based on provided groups and stacking logic."""

  table_name_to_feature_spec = {
      f.table_spec.name: f for f in jax.tree.leaves(features)
  }
  logging.info("Creating stack: %s with tables: %s", stack_name, table_names)
  table_name_to_setting_in_stack = _compute_table_to_setting_in_stack(
      stack_name=stack_name,
      table_names=table_names,
      padded_embedding_dim=padded_embedding_dim,
      table_to_padded_vocab_size=table_to_padded_vocab_size,
      global_device_count=global_device_count,
      num_sc_per_device=num_sc_per_device,
      rotation=rotation,
  )
  # Get the features for which the table is stacked in this group.
  stacked_features = [
      feature
      for feature in jax.tree.leaves(features)
      if feature.table_spec.name in table_names
  ]
  stack_sample_count = 0
  feature_to_row_offset = {}
  for feature in stacked_features:
    logging.info(
        "Feature %s has output shape %s", feature.name, feature.output_shape
    )
    logging.info("stack_sample_count %s", stack_sample_count)
    feature_to_row_offset[feature.name] = stack_sample_count
    # Last dimension is the table dimension which needs to be excluded in
    # calculating the sample count.
    stack_sample_count += int(np.prod(feature.output_shape[:-1]))

  max_ids_per_partition, max_unique_ids_per_partition = _get_limits_for_stack(
      table_names=table_names,
      table_name_to_feature_spec=table_name_to_feature_spec,
      default_max_ids_per_partition=stack_to_max_ids_per_partition(
          stack_name, stack_sample_count
      ),
      default_max_unique_ids_per_partition=stack_to_max_unique_ids_per_partition(
          stack_name, stack_sample_count
      ),
  )
  # StackedTableSpec for this stack.
  stacked_table_spec = embedding_spec.StackedTableSpec(
      stack_name=stack_name,
      stack_vocab_size=sum(
          table_to_padded_vocab_size[tname] for tname in table_names
      ),
      stack_embedding_dim=padded_embedding_dim,
      optimizer=stacked_features[0].table_spec.optimizer,
      combiner=stacked_features[0].table_spec.combiner,
      total_sample_count=stack_sample_count,
      max_ids_per_partition=max_ids_per_partition,
      max_unique_ids_per_partition=max_unique_ids_per_partition,
  )

  def _update_feature(
      feature: embedding_spec.FeatureSpec,
  ) -> None:
    # If the feature (that is its table) is not stacked here, then it need not
    # be updated.
    if feature.table_spec.name not in table_names:
      return
    feature.table_spec.stacked_table_spec = stacked_table_spec
    feature.table_spec.setting_in_stack = table_name_to_setting_in_stack[
        feature.table_spec.name
    ]
    feature.id_transformation = embedding_spec.FeatureIdTransformation(
        row_offset=feature_to_row_offset[feature.name],
        col_offset=table_name_to_setting_in_stack[
            feature.table_spec.name
        ].row_offset_in_shard
        * (num_sc_per_device * global_device_count),
        col_shift=table_name_to_setting_in_stack[
            feature.table_spec.name
        ].shard_rotation,
    )
    logging.info(
        "Feature '%s' refers to stacked table: %s with specs %s",
        feature.name,
        table_name_to_setting_in_stack[feature.table_spec.name].stack_name,
        stacked_table_spec,
    )
    logging.info(
        "Updating feature '%s' with id transformation: %s",
        feature.name,
        feature.id_transformation,
    )

  for feature in jax.tree.leaves(features):
    _update_feature(feature)


def _get_stack_name(
    table_names: Sequence[str],
    use_short_stack_names: bool = True,
    max_original_display_length: int = 50,
    hash_length: int = 12,
) -> str:
  """Returns the stack name for the given table names.

  Args:
    table_names: A list of table names to be stacked.
    use_short_stack_names: If `True`, a hash will be appended to the stack name
      to avoid long names. Otherwise, the stack name will be the concatenation
      of the table names.
    max_original_display_length: The maximum length of the original table names
      to display in the stack name.
    hash_length: The length of the hash to append to the stack name.

  Returns:
    The stack name.
  """
  stack_name = "_".join(table_names)
  if not use_short_stack_names:
    return stack_name
  shortened_name = stack_name[:max_original_display_length]
  if len(stack_name) > max_original_display_length:
    shortened_name += "..."
    shortened_name += hashlib.sha256(stack_name.encode("utf-8")).hexdigest()[
        :hash_length
    ]
  logging.info(
      "Creating short name %s for stack %s", shortened_name, stack_name
  )
  return shortened_name


def stack_tables(
    features: Nested[embedding_spec.FeatureSpec],
    table_names: Sequence[str],
    global_device_count: int,
    num_sc_per_device: int,
    rotation: int | None = None,
    stack_to_max_ids_per_partition: LimitsCallable = get_default_limits,
    stack_to_max_unique_ids_per_partition: LimitsCallable = get_default_limits,
    stack_name: str | None = None,
    fail_on_excess_padding: bool = False,
) -> None:
  """Creates new feature specs based on specified stacking groups.

  Checks that the tables in the groups have same dim, optimizer and combiner.
  Then creates new feature specs with updated table specs with relevant
  fields related to stacking setup. The features are updated in-place with the
  new table specs.
  Args:
    features: The input features.
    table_names: A list of table names to be stacked.
    global_device_count: The number of global devices (chips). Typically
      `mesh.size`.
    num_sc_per_device: The number of sparsecores per device.
    rotation: The shard rotation factor for each stacked table.  If None, sets
      to num_sc_per_device.  Default: None.
    stack_to_max_ids_per_partition: Override the max_ids_per_partition for each
      stack.
    stack_to_max_unique_ids_per_partition: Override the
      max_unique_ids_per_partition for each stack.
    stack_name: A unique name for the table stack. If None, a default name will
      be chosen.
    fail_on_excess_padding: If `True`, raises an error if the embedding
      dimensions of the tables to stack would lead to excessive padding (i.e. do
      not match when rounded up to the nearest multiple of 8 values).
  """
  if not stack_name:
    stack_name = _get_stack_name(table_names)
  flatten_features = jax.tree.leaves(features)
  tables_in_group = {
      feature.table_spec.name: feature.table_spec
      for feature in flatten_features
      if feature.table_spec.name in table_names
  }
  table_to_padded_dim, table_to_padded_vocab_size = round_up_dim_and_vocab_size(
      tables_in_group, num_sc_per_device * global_device_count
  )

  # Pad to maximum embedding dim.
  padded_embedding_dim = max(table_to_padded_dim.values())
  # All tables in a group _should_ have same embedding dimension after round up
  # to preserve memory - but this is not a hard requirement.
  if not all(
      [table_to_padded_dim[t] == padded_embedding_dim for t in table_names]
  ):
    excess_padding = 0
    for table_name in table_names:
      padded_dim = table_to_padded_dim[table_name]
      padded_vocab = table_to_padded_vocab_size[table_name]
      excess_padding += (padded_embedding_dim - padded_dim) * padded_vocab

    msg = (
        f"Excess padding detected for stacked table {stack_name}.\n"
        f"  Tables: {table_names},\n"
        f"  Padded sizes: {table_to_padded_dim.values()},\n"
        f"  Excess padding: {excess_padding} values.\n"
        "To reduce the memory footprint, stack tables that have consistent "
        "embedding dimensions when rounded up to the nearest multiple of 8 "
        "values."
    )

    if fail_on_excess_padding:
      raise ValueError(msg)
    else:
      logging.warning("WARNING during stack_tables:\n%s", msg)

  _verify_stack_tables(
      stack_name,
      table_names,
      flatten_features,
      tables_in_group,
  )

  rotation = rotation if rotation is not None else num_sc_per_device
  _stack_feature_specs(
      stack_name=stack_name,
      features=features,
      table_names=table_names,
      padded_embedding_dim=padded_embedding_dim,
      table_to_padded_vocab_size=table_to_padded_vocab_size,
      global_device_count=global_device_count,
      num_sc_per_device=num_sc_per_device,
      rotation=rotation,
      stack_to_max_ids_per_partition=stack_to_max_ids_per_partition,
      stack_to_max_unique_ids_per_partition=stack_to_max_unique_ids_per_partition,
  )


# TODO(b/359077239): Explore other ways to take limits as user input.
def auto_stack_tables(
    features: Nested[embedding_spec.FeatureSpec],
    global_device_count: int,
    num_sc_per_device: int,
    rotation: int | None = None,
    stack_to_max_ids_per_partition: LimitsCallable = get_default_limits,
    stack_to_max_unique_ids_per_partition: LimitsCallable = get_default_limits,
    *,
    use_short_stack_names: bool = True,
    activation_mem_bytes_limit=2048 * 1024,
) -> None:
  """Creates new feature specs based on auto stacking logic.

  All tables with same dimensions and optimizer/combiner are stacked together.
  The tables are stacked in the alphabetical order of their names.
  The new feature specs have updated table specs with relevant fields related to
  stacking setup. The features are updated in-place with the new table specs.

  Args:
    features: The input features.
    global_device_count: The number of global devices (chips). Typically
      `mesh.size`.
    num_sc_per_device: The number of sparsecores per device.
    rotation: The shard rotation factor for each stacked table.  If None, sets
      to num_sc_per_device.  Default: None.
    stack_to_max_ids_per_partition: Override the max_ids_per_partition for each
      stack.
    stack_to_max_unique_ids_per_partition: Override the
      max_unique_ids_per_partition for each stack.
    use_short_stack_names: If `True`, a hash will be appended to the stack name
      to avoid long names. Otherwise, the stack name will be the concatenation
      of the table names.
    activation_mem_bytes_limit: If the activation memory usage is larger than
      this limit, the table will not be stacked. Default is 2MB.
  """
  flatten_features = jax.tree.leaves(features)
  flatten_tables = {
      feature.table_spec.name: feature.table_spec
      for feature in flatten_features
  }
  groups = _get_stack_table_names(
      num_sc=num_sc_per_device * global_device_count,
      flatten_tables=flatten_tables,
      flatten_features=flatten_features,
      activation_mem_bytes_limit=activation_mem_bytes_limit,
  )

  updated_features = features
  for group in groups:
    logging.info("Stack group with tables: %s", group)
    stack_tables(
        features=updated_features,
        table_names=group,
        global_device_count=global_device_count,
        num_sc_per_device=num_sc_per_device,
        rotation=rotation,
        stack_to_max_ids_per_partition=stack_to_max_ids_per_partition,
        stack_to_max_unique_ids_per_partition=stack_to_max_unique_ids_per_partition,
        fail_on_excess_padding=False,  # Guaranteed to be satisfied.
        stack_name=_get_stack_name(group, use_short_stack_names),
    )


def _unstack_and_unshard_stacked_table(
    stacked_table: jax.Array,
    stacked_table_specs: embedding_spec_pb2.StackedTableSpecProto,
    donate: bool = False,
) -> dict[str, jax.Array]:
  """Unstack and unshard the stacked table."""

  stacked_table_sharding = stacked_table.sharding
  num_sparse_cores = stacked_table_specs.num_sparsecores
  stack_embedding_dim = stacked_table_specs.stack_embedding_dim

  # increase a rank and the first dimension is the number of sparse cores.
  stacked_table_3d = jax.jit(
      lambda x: x.reshape(num_sparse_cores, -1, stack_embedding_dim),
      in_shardings=stacked_table_sharding,
      out_shardings=stacked_table_sharding,
  )(stacked_table)

  if donate:
    # to save memory
    stacked_table.delete()

  unstack_and_unshard_fn = jax.jit(
      _unshard_and_unstack_single_table,
      static_argnames=(
          "sharded_vocab_size",
          "row_offset_in_shard",
          "shard_rotation",
          "vocab_size",
          "embedding_dim",
      ),
      in_shardings=stacked_table_sharding,
      out_shardings=stacked_table_sharding,
  )

  logging.info(
      "unstack_and_unshard_stacked_table: num_sparse_cores: %s,"
      " stack_embedding_dim: %s, stacked_table_shape: %s,"
      " stacked_table_sharding: %s",
      num_sparse_cores,
      stack_embedding_dim,
      stacked_table.shape,
      stacked_table.sharding,
  )

  ret = {}
  for table_setting_in_stack in stacked_table_specs.table_specs:
    row_offset = table_setting_in_stack.row_offset_in_shard
    chunk_size = table_setting_in_stack.padded_vocab_size // num_sparse_cores
    rotation = table_setting_in_stack.shard_rotation
    vocab_size = table_setting_in_stack.vocab_size
    embedding_dim = table_setting_in_stack.embedding_dim

    logging.info(
        "unstack_and_unshard_stacked_table: table_name: %s, row_offset: %s,"
        " chunk_size: %s, rotation: %s, vocab_size: %s, embedding_dim: %s",
        table_setting_in_stack.table_name,
        row_offset,
        chunk_size,
        rotation,
        vocab_size,
        embedding_dim,
    )

    ret[table_setting_in_stack.table_name] = unstack_and_unshard_fn(
        stacked_table_3d,
        chunk_size,
        row_offset,
        rotation,
        vocab_size,
        embedding_dim,
    )

    logging.info(
        "unstack_and_unshard_stacked_table: unstacked table_name: %s, shape:"
        " %s, sharding: %s",
        table_setting_in_stack.table_name,
        ret[table_setting_in_stack.table_name].shape,
        ret[table_setting_in_stack.table_name].sharding,
    )

  return ret


def unstack_and_unshard_stacked_tables(
    stacked_tables: dict[str, jax.Array],
    embedding_specs: embedding_spec_pb2.EmbeddingSpecProto,
    donate: bool = False,
) -> dict[str, jax.Array]:
  """Unstack and unshard the stacked tables.

  Args:
    stacked_tables: A dictionary of stacked tables. The keys are the stacked
      table names.
    embedding_specs: The embedding spec proto.
    donate: Whether the stacked tables are donated to reduce memory usage.

  Returns:
    A dictionary of unstacked tables with keys as the table names.
  """
  ret = {}
  for stacked_table_spec in embedding_specs.stacked_table_specs:
    if (
        stacked_table := stacked_tables.get(stacked_table_spec.stack_name)
    ) is None:
      raise ValueError(
          f"Stacked table '{stacked_table_spec.stack_name}' not found in"
          " `stacked_tables`."
      )
    ret.update(
        _unstack_and_unshard_stacked_table(
            stacked_table, stacked_table_spec, donate
        )
    )
  return ret


def _stack_and_shard_feature_table(
    feature_tables: dict[str, jax.Array],
    stacked_table_specs: embedding_spec_pb2.StackedTableSpecProto,
    delete_input: bool = False,
) -> jax.Array:
  """Stack and shard feature tables and return one stacked table."""

  def mod_shard(
      tbl,
      padded_vocab_size,
      padded_embedding_dim,
      num_sparsecores,
      shard_rotation,
  ):
    tbl_padded = jnp.pad(
        tbl,
        (
            (0, padded_vocab_size - tbl.shape[0]),
            (0, padded_embedding_dim - tbl.shape[1]),
        ),
    )
    chunk_size = padded_vocab_size // num_sparsecores

    tbl_3d = tbl_padded.reshape(
        chunk_size,
        -1,
        stacked_table_specs.stack_embedding_dim,
    )

    # mod sharding
    tbl_sharded = tbl_3d.transpose((1, 0, 2))

    tbl_rotated = jnp.roll(tbl_sharded, shard_rotation, axis=0)

    return tbl_rotated

  sharded_tables = []
  for table_setting_in_stack in stacked_table_specs.table_specs:
    # prepare each feature table
    tbl = feature_tables[table_setting_in_stack.table_name]

    if (
        stacked_table_specs.stack_embedding_dim
        != table_setting_in_stack.padded_embedding_dim
    ):
      raise ValueError(
          f"Embedding dim of table_spec {table_setting_in_stack.table_name} is"
          f" {table_setting_in_stack.padded_embedding_dim} but the stacked"
          f" table embedding dim is {stacked_table_specs.stack_embedding_dim}."
      )

    # mod shard and rotate the table
    tbl_rotated = jax.jit(
        mod_shard,
        static_argnames=(
            "padded_vocab_size",
            "padded_embedding_dim",
            "num_sparsecores",
            "shard_rotation",
        ),
        in_shardings=tbl.sharding,
        out_shardings=tbl.sharding,
    )(
        tbl,
        table_setting_in_stack.padded_vocab_size,
        table_setting_in_stack.padded_embedding_dim,
        stacked_table_specs.num_sparsecores,
        table_setting_in_stack.shard_rotation,
    )

    if delete_input:
      # to save memory
      tbl.delete()

    sharded_tables.append(tbl_rotated)

  # stack tables to create the final stacked table
  tbl_stacked_3d = jax.numpy.concatenate(sharded_tables, axis=1)
  tbl_stacked = jax.jit(
      lambda x: x.reshape(-1, stacked_table_specs.stack_embedding_dim),
      in_shardings=tbl_stacked_3d.sharding,
      out_shardings=tbl_stacked_3d.sharding,
  )(tbl_stacked_3d)

  return tbl_stacked


def stack_and_shard_feature_tables(
    feature_tables: dict[str, jax.Array],
    embedding_specs: embedding_spec_pb2.EmbeddingSpecProto,
    delete_input: bool = False,
) -> dict[str, jax.Array]:
  """Stack and shard the feature tables and return the stacked tables.

  This function can be run on both TPU or CPU backends. The stacked tables will
  be mod-sharded specifically for training TPU topologies, which is described in
  `embedding_specs`.

  The output sharding will be the same as the sharding of the input
  `feature_tables`, and all feature tables' shardings are required to be the
  same. For best restoration performance at the target TPU topologies, the
  number of shards of the input sharding should be the same as the number of TPU
  devices at the target topology.

  Args:
    feature_tables: A dictionary of feature tables. The keys are the table
      names.  Arrays are required to all have the same
      jax.sharding.NamedSharding.
    embedding_specs: The embedding spec proto.
    delete_input: Whether to delete the input feature tables to reduce peak
      memory usage.

  Returns:
    A dictionary of stacked tables with keys as the stacked table names.
  """

  in_sharding = list(feature_tables.values())[0].sharding
  for arr in feature_tables.values():
    if arr.sharding != in_sharding:
      raise ValueError(
          "All feature tables must have the same sharding. Found"
          f" {arr.sharding} and {in_sharding}."
      )

  ret = {}
  for stacked_table_spec in embedding_specs.stacked_table_specs:
    in_tables = {}
    for table_setting_in_stack in stacked_table_spec.table_specs:
      tbl_name = table_setting_in_stack.table_name
      if (tbl := feature_tables.get(tbl_name)) is None:
        raise ValueError(f"{tbl_name}' not found in `feature_tables`.")

      in_tables[tbl_name] = tbl

    ret[stacked_table_spec.stack_name] = _stack_and_shard_feature_table(
        in_tables, stacked_table_spec, delete_input
    )

    logging.info(
        "stack_and_shard_feature_tables: stacked table_name: %s, shape: %s,"
        " sharding: %s",
        stacked_table_spec.stack_name,
        ret[stacked_table_spec.stack_name].shape,
        ret[stacked_table_spec.stack_name].sharding,
    )

  return ret


def get_row_ids_in_stacked_table(
    stack_table_spec: embedding_spec_pb2.StackedTableSpecProto,
    table_spec: embedding_spec_pb2.TableSpecProto,
    row_ids: Sequence[int],
) -> Sequence[int]:
  """Returns the stacked table's row ids for the given unsharded table's row ids.

  Args:
    stack_table_spec: StackedTableSpecProto describing the stacked table
    table_spec: TableSpecProto of the unsharded table
    row_ids: Squence of row ids of the unsharded table

  Returns:
    Row ids of the stacked table
  """
  ret = []

  num_sparse_cores = stack_table_spec.num_sparsecores
  stack_shard_size = stack_table_spec.stack_vocab_size // num_sparse_cores

  for row_id in row_ids:
    assert (
        row_id < table_spec.vocab_size
    ), f"{row_id} execeeds available vocabulary size [{table_spec.vocab_size}]."
    shard_id = (
        row_id % num_sparse_cores + table_spec.shard_rotation
    ) % num_sparse_cores
    sharded_row_id = row_id // num_sparse_cores + table_spec.row_offset_in_shard
    ret.append(shard_id * stack_shard_size + sharded_row_id)

  return ret
