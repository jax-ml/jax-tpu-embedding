# Copyright 2022 The jax_tpu_embedding Authors.
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

"""Utils support load or retrieve TPU embedding variables."""

from typing import List, Mapping

from absl import flags
from absl import logging
import tensorflow as tf
# pylint: disable=g-direct-tensorflow-import
from tensorflow..dtensor.python import accelerator_util
from tensorflow..dtensor.python import gen_dtensor_ops
from tensorflow..python.eager import context
# pylint: enable=g-direct-tensorflow-import

TableConfig = tf.tpu.experimental.embedding.TableConfig


def init_tpu_system():
  """Initialize tpu system for tpu embedding."""
  # As Jax TPUEmbedding also use coordination service to initialize
  # embedding engine, setting `enable_coordination_service` to `False` 
  # in dtensor to avoid potential conflict.
  tf.experimental.dtensor.initialize_accelerator_system(
      enable_coordination_service=False)
  # TODO(b/259131699): delete this when add mlir bridge support.
  tf.config.experimental.disable_mlir_bridge()


def is_tpu_use_tfrt() -> bool:
  """Checks if tpu backend is using TFRT-TPU Runtime."""
  return 'tpu_use_tfrt' in flags.FLAGS and flags.FLAGS['tpu_use_tfrt'].value


def shutdown_tpu_system():
  """Shuts down the TPU system."""
  context.async_wait()

  @tf.function
  def _shutdown_tpu_system():
    return gen_dtensor_ops.shutdown_tpu_system()

  success = _shutdown_tpu_system() if is_tpu_use_tfrt() else True
  if success:
    logging.info('TPU system shut down.')
  else:
    logging.warning('TPU system fails to shut down.')

  accelerator_util.set_initialized(None)

  # reset TF context to stop gRPC servers.
  context._reset_context()  # pylint: disable=protected-access
  context.context()._clear_caches()  # pylint: disable=protected-access


def _local_shard_size(total_rows: int, num_shards: int, shard_id: int) -> int:
  """Compute local shard size from a global total row number.

  Args:
    total_rows: global size of rows overall.
    num_shards: number of shard to split.
    shard_id: the id of local shard size we need.

  Returns:
    local shard size of given shard_id.
  """
  residual = total_rows % num_shards
  shard_size = total_rows // num_shards

  # Divides residual rows
  if shard_id < residual:
    return shard_size + 1
  return shard_size


def create_tables_variables(
    shard_id: int, num_shards: int, table_config_list: List[TableConfig]
) -> Mapping[str, Mapping[str, tf.Tensor]]:
  """Create all table variables of parameters and slot_variables.

  Args:
    shard_id: shard id of variables to create.
    num_shards: total number of shards.
    table_config_list: A list of all table config.

  Returns:
    A nested dictionary of tensors. Outer dictionary is indexed by table's name
    while the inner is indexed by variable names (`parameters` and slot names).
  """

  def _create_per_table(table_config: TableConfig) -> Mapping[str, tf.Tensor]:
    """Create variables for a table."""

    local_shard_size = _local_shard_size(
        total_rows=table_config.vocabulary_size,
        num_shards=num_shards,
        shard_id=shard_id)

    # pylint: disable=protected-access
    slot_names = table_config.optimizer._slot_names()
    slot_initalizars = table_config.optimizer._slot_initializers()
    # pylint: enable=protected-access

    # If the expected shard size for this table is zero on this host, we ignore
    # what the user passes in and just pass a zeros tensor with the shape
    # (1, table.dim).
    if local_shard_size == 0:
      zeros = tf.zeros(shape=(1, table_config.dim), dtype=tf.float32)
      variable_names = ['parameters'] + slot_names
      return {var_name: zeros for var_name in variable_names}

    local_shape = (local_shard_size, table_config.dim)

    # parameters and slot variable for this table.
    per_table_variables = {}
    per_table_variables['parameters'] = table_config.initializer(
        local_shape, dtype=tf.float32)

    for slot_name, slot_initializer in zip(slot_names, slot_initalizars):
      per_table_variables[slot_name] = slot_initializer(
          local_shape, dtype=tf.float32)

    return per_table_variables

  tables_variables = {}
  for table_config in table_config_list:
    tables_variables[table_config.name] = _create_per_table(table_config)

  return tables_variables


def load_embedding_tables_impl(table_config_list: List[TableConfig],
                               config_proto_str: bytes, host_id: int,
                               num_hosts: int) -> None:
  """Load parameters and slot variables of embedding tables.

  Args:
    table_config_list: A list of tf.tpu.experimental.embedding.TableConfig.
    config_proto_str: Serialized string of tpu embedding configuration proto.
    host_id: This also represents shard id of table variables to load, as each
      host will only have one shard.
    num_hosts: This also represents num of shards of table variables.
  """
  embedding_variables = create_tables_variables(
      shard_id=host_id,
      num_shards=num_hosts,
      table_config_list=table_config_list)

  @tf.function
  def load_fn(table_config_list: List[TableConfig],
              table_variables: Mapping[str, Mapping[str, tf.Tensor]],
              config_proto_str: str, host_id: int, num_hosts: int) -> None:
    """Load embedding from CPU onto HBM of embedding engine."""

    for table_config in table_config_list:
      table_config.optimizer._load()(  # pylint: disable=protected-access
          table_name=table_config.name,
          num_shards=num_hosts,
          shard_id=host_id,
          config=config_proto_str,
          **table_variables[table_config.name])
      # Ensure that only the first table gets a config so that we don't bloat
      # graph by attaching this large string to each op. When there is a large
      # number of tables, this can be an issue.
      config_proto_str = None

  load_fn(
      table_config_list=table_config_list,
      table_variables=embedding_variables,
      config_proto_str=config_proto_str,
      host_id=host_id,
      num_hosts=num_hosts)


def retrieve_embedding_tables_impl(
    table_config_list: List[TableConfig], config_proto_str: bytes, host_id: int,
    num_hosts: int) -> Mapping[str, Mapping[str, tf.Tensor]]:
  """Retrieve embeddings variables from embedding engine to CPU host.

  Args:
    table_config_list: A list of tf.tpu.experimental.embedding.TableConfig.
    config_proto_str: Serialized string of tpu embedding configuration proto.
    host_id: This also represents shard id of table variables to retrieve
    num_hosts: This also represents num of shards of table variables.

  Returns:
    A nested dictionary of tensors. Outer dictionary is indexed by table's name
    while the inner is indexed by variable names (`parameters` and slot names).
  """

  @tf.function
  def retrieve_fn(table_config_list: List[TableConfig], config_proto_str: str,
                  host_id: int,
                  num_hosts: int) -> Mapping[str, Mapping[str, tf.Tensor]]:
    """Retrieve embedding from embedding engine."""

    # Dictionary of all embedding tables' variables.
    retrieved_tables = {}
    for table_config in table_config_list:
      retrieved = table_config.optimizer._retrieve()(  # pylint: disable=protected-access
          table_name=table_config.name,
          num_shards=num_hosts,
          shard_id=host_id,
          config=config_proto_str)
      # Only include the config in one op, to reduce graph size.
      config_proto_str = None

      # When there is no slot variables.
      if not isinstance(retrieved, tuple):
        retrieved = (retrieved,)

      local_shard_size = _local_shard_size(
          total_rows=table_config.vocabulary_size,
          num_shards=num_hosts,
          shard_id=host_id)

      if local_shard_size == 0:
        retrieved = (tf.zeros(
            (0, table_config.dim), dtype=tf.float32),) * len(retrieved)
      else:
        retrieved = [
            tf.ensure_shape(var, (local_shard_size, table_config.dim))
            for var in retrieved
        ]

      variable_names = ['parameters'] + table_config.optimizer._slot_names()  # pylint: disable=protected-access

      table_variables = {}
      for retrieved_var, name in zip(retrieved, variable_names):
        table_variables[name] = retrieved_var

      retrieved_tables[table_config.name] = table_variables
    return retrieved_tables

  return retrieve_fn(
      table_config_list=table_config_list,
      config_proto_str=config_proto_str,
      host_id=host_id,
      num_hosts=num_hosts)
