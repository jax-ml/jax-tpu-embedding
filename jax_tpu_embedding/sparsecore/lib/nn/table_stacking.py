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
import functools
from typing import Callable, Dict, Mapping, Sequence, TypeAlias, TypeVar

from absl import logging
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.lib.proto import embedding_spec_pb2
import numpy as np
import tree


T: TypeAlias = TypeVar("T")
Nested: TypeAlias = T | Sequence[T] | Mapping[str, T]
LimitsCallable: TypeAlias = Callable[[str, int], int]


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


def _next_largest_multiple(number: int, divisor: int) -> int:
  """Returns the next largest multiple of y that is greater than x."""
  return number if number % divisor == 0 else (number // divisor + 1) * divisor


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
  tables_to_padded_vocab_size = {
      n: _next_largest_multiple(spec.vocabulary_size, 8 * num_sc)
      for (n, spec) in tables.items()
  }
  return table_to_padded_dim, tables_to_padded_vocab_size


def _get_stack_table_names(
    tables: Mapping[str, embedding_spec.TableSpec], num_sc: int
) -> Sequence[Sequence[str]]:
  """Returns the stack groups for the tables based on their specs."""
  table_to_padded_dim, _ = round_up_dim_and_vocab_size(tables, num_sc)

  table_name_map = collections.defaultdict(list)
  for table_name, dim in table_to_padded_dim.items():
    key = (dim, tables[table_name].optimizer, tables[table_name].combiner)
    table_name_map[key].append(table_name)

  return list(table_name_map.values())


def _verify_stack_tables(
    stack_name: str,
    table_names: Sequence[str],
    features: Sequence[embedding_spec.FeatureSpec],
    tables: Mapping[str, embedding_spec.TableSpec],
    table_to_padded_dim: Mapping[str, int],
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
  # All tables in a group should have same embedding dimension after round up.
  if not all([
      table_to_padded_dim[t] == table_to_padded_dim[table_names[0]]
      for t in table_names
  ]):
    raise ValueError(
        f"Tables {table_names} in group {stack_name} have different"
        " embedding dimensions after round up."
    )


def _compute_table_to_setting_in_stack(
    stack_name: str,
    table_names: Sequence[str],
    table_to_padded_dim: Mapping[str, int],
    table_to_padded_vocab_size: Mapping[str, int],
    global_device_count: int,
    num_sc_per_device: int = 4,
) -> Mapping[str, embedding_spec.TableSettingInStack]:
  """Returns the table to setting in stack mapping."""
  table_name_to_setting_in_stack = {}
  row_offset_in_shard = 0
  shard_rotation = 0
  num_sc = num_sc_per_device * global_device_count
  for tname in table_names:
    if tname not in table_to_padded_vocab_size:
      raise ValueError(f"Padded vocab size for Table {tname} is missing.")
    if tname not in table_to_padded_dim:
      raise ValueError(f"Padded dimension for Table {tname} is missing.")
    num_rows_in_shard = table_to_padded_vocab_size[tname] // num_sc
    setting_in_stack = embedding_spec.TableSettingInStack(
        stack_name=stack_name,
        padded_vocab_size=table_to_padded_vocab_size[tname],
        padded_embedding_dim=table_to_padded_dim[tname],
        row_offset_in_shard=row_offset_in_shard,
        shard_rotation=shard_rotation,
    )
    row_offset_in_shard += num_rows_in_shard
    # Rotate the shard by num_sc_per_device and then bound by the
    # total number of sparsecores.
    shard_rotation = (shard_rotation + num_sc_per_device) % (num_sc)
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
    table_to_padded_dim: Mapping[str, int],
    table_to_padded_vocab_size: Mapping[str, int],
    global_device_count: int,
    num_sc_per_device: int = 4,
    stack_to_max_ids_per_partition: LimitsCallable = get_default_limits,
    stack_to_max_unique_ids_per_partition: LimitsCallable = get_default_limits,
) -> None:
  """Updated the feature spec based on provided groups and stacking logic."""

  table_name_to_feature_spec = {
      f.table_spec.name: f for f in tree.flatten(features)
  }
  logging.info("Creating stack: %s with tables: %s", stack_name, table_names)
  table_name_to_setting_in_stack = _compute_table_to_setting_in_stack(
      stack_name=stack_name,
      table_names=table_names,
      table_to_padded_dim=table_to_padded_dim,
      table_to_padded_vocab_size=table_to_padded_vocab_size,
      global_device_count=global_device_count,
      num_sc_per_device=num_sc_per_device,
  )
  # Get the features for which the table is stacked in this group.
  stacked_features = [
      feature
      for feature in tree.flatten(features)
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
      stack_embedding_dim=table_to_padded_dim[table_names[0]],
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

  for feature in tree.flatten(features):
    _update_feature(feature)


def stack_tables(
    features: Nested[embedding_spec.FeatureSpec],
    table_names: Sequence[str],
    global_device_count: int,
    num_sc_per_device: int = 4,
    stack_to_max_ids_per_partition: LimitsCallable = get_default_limits,
    stack_to_max_unique_ids_per_partition: LimitsCallable = get_default_limits,
    stack_name: str | None = None,
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
    stack_to_max_ids_per_partition: Override the max_ids_per_partition for each
      stack.
    stack_to_max_unique_ids_per_partition: Override the
      max_unique_ids_per_partition for each stack.
    stack_name: A unique name for the table stack. If None, a default name will
      be chosen.
  """
  if stack_name is None:
    # TODO(b/355289256): Consider better name for the stack.
    stack_name = "_".join(table_names)
  flatten_features = tree.flatten(features)
  tables_in_group = {
      feature.table_spec.name: feature.table_spec
      for feature in flatten_features
      if feature.table_spec.name in table_names
  }
  table_to_padded_dim, tables_to_padded_vocab_size = (
      round_up_dim_and_vocab_size(
          tables_in_group, num_sc_per_device * global_device_count
      )
  )
  _verify_stack_tables(
      stack_name,
      table_names,
      flatten_features,
      tables_in_group,
      table_to_padded_dim,
  )
  _stack_feature_specs(
      stack_name=stack_name,
      features=features,
      table_names=table_names,
      table_to_padded_dim=table_to_padded_dim,
      table_to_padded_vocab_size=tables_to_padded_vocab_size,
      global_device_count=global_device_count,
      num_sc_per_device=num_sc_per_device,
      stack_to_max_ids_per_partition=stack_to_max_ids_per_partition,
      stack_to_max_unique_ids_per_partition=stack_to_max_unique_ids_per_partition,
  )


# TODO(b/359077239): Explore other ways to take limits as user input.
def auto_stack_tables(
    features: Nested[embedding_spec.FeatureSpec],
    global_device_count: int,
    num_sc_per_device: int = 4,
    stack_to_max_ids_per_partition: LimitsCallable = get_default_limits,
    stack_to_max_unique_ids_per_partition: LimitsCallable = get_default_limits,
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
    stack_to_max_ids_per_partition: Override the max_ids_per_partition for each
      stack.
    stack_to_max_unique_ids_per_partition: Override the
      max_unique_ids_per_partition for each stack.
  """
  flatten_features = tree.flatten(features)
  flatten_tables = {
      feature.table_spec.name: feature.table_spec
      for feature in flatten_features
  }
  groups = _get_stack_table_names(
      flatten_tables, num_sc=num_sc_per_device * global_device_count
  )
  updated_features = features
  for group in groups:
    logging.info("Stack group with tables: %s", group)
    stack_tables(
        features=updated_features,
        table_names=group,
        global_device_count=global_device_count,
        num_sc_per_device=num_sc_per_device,
        stack_to_max_ids_per_partition=stack_to_max_ids_per_partition,
        stack_to_max_unique_ids_per_partition=stack_to_max_unique_ids_per_partition,
    )


def _unstack_and_unshard_stacked_table(
    stacked_table: jax.Array,
    stacked_table_specs: embedding_spec_pb2.StackedTableSpecProto,
    donate: bool = False,
) -> Dict[str, jax.Array]:
  """Unstack and unshard the stacked table."""

  stacked_table_sharding = stacked_table.sharding
  num_sparse_cores = stacked_table_specs.num_sparsecores
  stack_embedding_dim = stacked_table_specs.stack_embedding_dim

  # increase a rank and the first dimension is the number of sparse cores.
  stacked_table_3d = jax.jit(
      fun=lambda x: x.reshape(num_sparse_cores, -1, stack_embedding_dim),
      in_shardings=stacked_table_sharding,
      out_shardings=stacked_table_sharding,
  )(stacked_table)

  if donate:
    # to save memory
    stacked_table.delete()

  @functools.partial(
      jax.jit,
      static_argnames=(
          "row_offset",
          "chunk_size",
          "rotation",
          "stack_embedding_dim",
          "vocab_size",
          "embedding_dim",
      ),
      in_shardings=stacked_table_sharding,
      out_shardings=stacked_table_sharding,
  )
  def _unstack_and_unshard(
      stacked_table_3d,
      row_offset,
      chunk_size,
      rotation,
      stack_embedding_dim,
      vocab_size,
      embedding_dim,
  ):
    # From each shard get the chunk for 'this' table
    shards = stacked_table_3d[:, row_offset : row_offset + chunk_size, :]

    # Undo the shard rotation (note '-' for reverse direction)
    shards = jnp.roll(shards, -rotation, axis=0)

    # Undo the mod sharding
    un_mod_shard = shards.transpose((1, 0, 2))

    # Remove the first dimension
    ret = un_mod_shard.reshape(-1, stack_embedding_dim)

    # Remove paddings
    ret = ret[:vocab_size, :embedding_dim]

    return ret

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

    ret[table_setting_in_stack.table_name] = _unstack_and_unshard(
        stacked_table_3d,
        row_offset,
        chunk_size,
        rotation,
        stack_embedding_dim,
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
    stacked_tables: Dict[str, jax.Array],
    embedding_specs: embedding_spec_pb2.EmbeddingSpecProto,
    donate: bool = False,
) -> Dict[str, jax.Array]:
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
