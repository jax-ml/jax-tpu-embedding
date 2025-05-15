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
  table_to_padded_vocab_size = {
      n: _next_largest_multiple(spec.vocabulary_size, 8 * num_sc)
      for (n, spec) in tables.items()
  }
  return table_to_padded_dim, table_to_padded_vocab_size


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
      f.table_spec.name: f for f in tree.flatten(features)
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

  for feature in tree.flatten(features):
    _update_feature(feature)


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
  if stack_name is None:
    # TODO(b/355289256): Consider better name for the stack.
    stack_name = "_".join(table_names)
  flatten_features = tree.flatten(features)
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
    activation_mem_bytes_limit=2024 * 1024,
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
    rotation: The shard rotation factor for each stacked table.  If None,
      sets to num_sc_per_device.  Default: None.
    stack_to_max_ids_per_partition: Override the max_ids_per_partition for each
      stack.
    stack_to_max_unique_ids_per_partition: Override the
      max_unique_ids_per_partition for each stack.
    activation_mem_bytes_limit: If the activation memory
      usage is larger than this limit, the table will not be stacked.
  """
  flatten_features = tree.flatten(features)
  flatten_tables = {
      feature.table_spec.name: feature.table_spec
      for feature in flatten_features
  }
  groups = _get_stack_table_names(
      flatten_tables, num_sc=num_sc_per_device * global_device_count
  )

  # We do not need the vocab size information for auto stacking.
  num_sc = global_device_count * num_sc_per_device
  table_to_padded_dim, _ = round_up_dim_and_vocab_size(
      flatten_tables, num_sc
  )

  # Calculate sample_count per sparsecore.
  table_to_sample_count = {}
  for feature in flatten_features:
    if feature.table_spec.name in table_to_sample_count:
      sample_count = table_to_sample_count[feature.table_spec.name]
    else:
      sample_count = 0
    table_to_sample_count[feature.table_spec.name] = sample_count + (
        int(np.prod(feature.output_shape[:-1]) // num_sc)
    )

  table_to_activation_mem_bytes = {}
  validated_groups = []
  for group in groups:
    # A list of groups that are split from the current group.
    split_groups = []
    # Iterate through all tables in the current group.
    for table_name in group:
      # Calculate and register the activation memory usage of this table.
      table_to_activation_mem_bytes[table_name] = (
          table_to_padded_dim[table_name]
          * table_to_sample_count[table_name]
          * 4  # 4 bytes per f32
      )
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
        if (
            accumuated_activation_mem_bytes
            + table_to_activation_mem_bytes[table_name]
        ) <= activation_mem_bytes_limit:
          # Append to this candidate group.
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

  updated_features = features
  for group in validated_groups:
    logging.info("Stack group with tables: %s", group)
    stack_tables(
        features=updated_features,
        table_names=group,
        global_device_count=global_device_count,
        num_sc_per_device=num_sc_per_device,
        rotation=rotation,
        stack_to_max_ids_per_partition=stack_to_max_ids_per_partition,
        stack_to_max_unique_ids_per_partition=stack_to_max_unique_ids_per_partition,
        fail_on_excess_padding=False  # Guaranteed to be satisfied.
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
      lambda x: x.reshape(num_sparse_cores, -1, stack_embedding_dim),
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


def _stack_and_shard_feature_table(
    feature_tables: Dict[str, jax.Array],
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
    feature_tables: Dict[str, jax.Array],
    embedding_specs: embedding_spec_pb2.EmbeddingSpecProto,
    delete_input: bool = False,
) -> Dict[str, jax.Array]:
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
