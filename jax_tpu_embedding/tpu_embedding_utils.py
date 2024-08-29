# Copyright 2024 The jax_tpu_embedding Authors.
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

import copy
from typing import List, Mapping, Optional, Union

from absl import flags
from absl import logging
import jax
from jax.experimental import multihost_utils
from jax.experimental import jax2tf
import jax.numpy as jnp
from jax_tpu_embedding import config_utils
from jax_tpu_embedding import pytype_utils
import numpy as np
import tensorflow as tf
import orbax.checkpoint as ocp

# pylint: disable=g-direct-tensorflow-import
from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.python.eager import context
# pylint: enable=g-direct-tensorflow-import

from tensorflow.python.tpu.ops import gen_tpu_embedding_ops as tpu_ops  # pylint: disable=g-direct-tensorflow-import

JaxArrayRestoreArgs = ocp.ArrayRestoreArgs
GlobalHostArray = pytype_utils.GlobalHostArray
NestedStruct = pytype_utils.NestedStruct
GlobalHostArrayRestoreArgs = pytype_utils.RestoreArgs
TableConfig = pytype_utils.TableConfig
TPUEmbeddingConfigurationProto = pytype_utils.TPUEmbeddingConfigurationProto


def init_tpu_system(enable_megacore=False):
  """Initialize tpu system for tpu embedding."""
  # As Jax TPUEmbedding also use coordination service to initialize
  # embedding engine, setting `enable_coordination_service` to `False`
  # in dtensor to avoid potential conflict.
  tf.experimental.dtensor.initialize_accelerator_system(
      enable_coordination_service=False,
      experimental_enable_megcore=enable_megacore)


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


def get_tuple_mask(config_str: bytes) -> pytype_utils.TensorProto:
  """Gets deduplication data tuple mask.

  Deduplication data is a xla tuple of integer or float 1-D tensors. This op is
  to generate a mask to respresent type and span size of these tensors.

  Args:
    config_str: A serialized string of TPUEmbeddingConfiguration.

  Returns:
    A tensor proto of tuple mask, whose first column is 0 or 1 to represent
    integer or float correspondingly, and second column is span size of each
    element in this tuple.
  """

  def _compute_dedup_tuple_mask():
    return tpu_ops.compute_dedup_data_tuple_mask(config_str)

  tuple_mask = jax2tf.call_tf(
      _compute_dedup_tuple_mask, has_side_effects=False
  )()
  dedup_tuple_mask_proto = config_utils.create_mask_proto(tuple_mask)
  return dedup_tuple_mask_proto


def get_tuple_mask_pmap(
    config_str: bytes,
    embedding_partitions: bytes,
    hbm_buffers_config: bytes,
    tpu_topology: bytes,
) -> pytype_utils.TensorProto:
  """Gets deduplication data tuple mask through pmap.

  Args:
    config_str: A serialized string of TPUEmbeddingConfiguration.
    embedding_partitions: A serialized string of EmbeddingPartitionsProto.
    hbm_buffers_config: A serialized string of HbmBuffersConfig.
    tpu_topology: A serialized string of TpuTopologyArgsProto.

  Returns:
    A tensor proto of tuple mask, whose first column is 0 or 1 to represent
    integer or float correspondingly, and second column is span size of each
    element in this tuple.
  """

  def _compute_dedup_data_size_tf():
    return tpu_ops.compute_dedup_data_size_v2(
        config_str, embedding_partitions, hbm_buffers_config, tpu_topology
    )

  def _compute_dedup_tuple_mask_tf():
    return tpu_ops.compute_dedup_data_tuple_mask_v2(
        config_str, embedding_partitions, hbm_buffers_config, tpu_topology
    )

  def _compute_dedup_data_size(_):
    return jax2tf.call_tf(
        _compute_dedup_data_size_tf,
        has_side_effects=False,
    )()

  def _compute_dedup_tuple_mask(dummy):
    return jax2tf.call_tf(
        _compute_dedup_tuple_mask_tf,
        has_side_effects=False,
        output_shape_dtype=jax.ShapeDtypeStruct(
            shape=(dummy.shape[0], 2), dtype=jnp.int32
        ),
    )()

  device = jax.devices()[0]
  dummy = np.zeros(1)
  num_elements = jax.pmap(_compute_dedup_data_size, devices=[device])(dummy)
  dummy = np.array([np.zeros(num_elements)])
  tuple_mask = jax.pmap(_compute_dedup_tuple_mask, devices=[device])(dummy)
  dedup_tuple_mask_proto = config_utils.create_mask_proto(tuple_mask[0])
  return dedup_tuple_mask_proto


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


def create_tables_variables_from_config(
    shard_id: int, num_shards: int, table_config_list: List[TableConfig]
) -> NestedStruct[tf.Tensor]:
  """Create table variables of parameters and slot_variables from table config.

  Args:
    shard_id: shard id of variables to create.
    num_shards: total number of shards.
    table_config_list: A list of all table config.

  Returns:
    A nested dictionary of tf tensors. Outer dictionary is indexed by table's 
    name while the inner is indexed by variable names (`parameters` and slot 
    names).
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


def create_table_variables_from_gha(
    table_gha_variables: NestedStruct[GlobalHostArray],
    shard_id: int,
    num_shards: int,
    table_config_list: List[TableConfig],
) -> NestedStruct[Union[tf.Tensor, GlobalHostArray]]:
  """Create table variables of parameters and slot_variables from table config.

  Args:
    table_gha_variables: A nested structure of table variables.
    shard_id: shard id of variables to create.
    num_shards: total number of shards.
    table_config_list: A list of all table config.

  Returns:
    A nested dictionary of tf.Tensor. Outer dictionary is indexed by table's
    name while the inner is indexed by variable names (`parameters` and slot
    names).
  """

  table_gha_variables = dict(table_gha_variables)
  for tb_cfg in table_config_list:
    global_shape = (tb_cfg.vocabulary_size, tb_cfg.dim)

    variable_names = ['parameters'] + tb_cfg.optimizer._slot_names()  # pylint: disable=protected-access
    for var_name in variable_names:
      assert table_gha_variables[tb_cfg.name][var_name] is not None
      if isinstance(
          table_gha_variables[tb_cfg.name][var_name], GlobalHostArray
      ):
        tvar = table_gha_variables[tb_cfg.name][var_name]
        assert tvar.shard_id == shard_id
        assert tvar.num_shards == num_shards
        assert tvar.global_shape == global_shape
      else:
        for _, value in table_gha_variables[tb_cfg.name][var_name].items():
          assert value.shard_id == shard_id
          assert value.num_shards == num_shards
          assert value.global_shape == global_shape

    table_gha_variables[tb_cfg.name] = jax.tree.map(
        lambda x: tf.constant(x.data, dtype=tf.float32),
        table_gha_variables[tb_cfg.name],
    )

  return table_gha_variables


def create_table_variables_from_jax_array(
    table_jax_variables: NestedStruct[jax.Array],
    num_shards: int,
    table_config_list: List[TableConfig],
) -> NestedStruct[Union[tf.Tensor, jax.Array]]:
  """Create table variables of parameters and slot_variables from table config.

  Args:
    table_jax_variables: A nested structure of table variables.
    num_shards: total number of shards.
    table_config_list: A list of all table config.

  Returns:
    A nested dictionary of tf.Tensor. Outer dictionary is indexed by table's
    name while the inner is indexed by variable names (`parameters` and slot
    names).
  """

  table_jax_variables = dict(table_jax_variables)
  for tb_cfg in table_config_list:
    global_shape = (tb_cfg.vocabulary_size, tb_cfg.dim)

    variable_names = ['parameters'] + tb_cfg.optimizer._slot_names()  # pylint: disable=protected-access
    for var_name in variable_names:
      assert table_jax_variables[tb_cfg.name][var_name] is not None
      if isinstance(table_jax_variables[tb_cfg.name][var_name], jax.Array):
        tvar = table_jax_variables[tb_cfg.name][var_name]
        assert len(tvar.global_shards) == num_shards
        assert tvar.shape == global_shape
      else:
        for _, value in table_jax_variables[tb_cfg.name][var_name].items():
          assert len(value.global_shards) == num_shards
          assert value.shape == global_shape

    table_jax_variables[tb_cfg.name] = jax.tree.map(
        lambda x: tf.constant(x.addressable_data(0), dtype=tf.float32),
        table_jax_variables[tb_cfg.name],
    )

  return table_jax_variables


def create_tables_global_host_array_restore_args(
    shard_id: int,
    num_shards: int,
    table_config_list: List[TableConfig],
    create_all_shards: bool = False,
) -> NestedStruct[GlobalHostArrayRestoreArgs]:
  """Creates RestoreArgs for tables and slot variables.

  Args:
    shard_id: shard id of variables to create.
    num_shards: total number of shards.
    table_config_list: A list of all table config.
    create_all_shards: If True, create RestoreArgs for all shards.

  Returns:
    A nested dictionary of checkpoint.RestoreArgs.
  """

  embed_restore_args = {}
  for tb_cfg in table_config_list:
    if create_all_shards:
      embed_restore_args[tb_cfg.name] = {}
      for slot_name in tb_cfg.optimizer._slot_names():  # pylint: disable=protected-access
        embed_restore_args[tb_cfg.name][slot_name] = {}
        for shard_id in range(num_shards):
          embed_restore_args[tb_cfg.name][slot_name][f'shard_{shard_id}'] = (
              GlobalHostArrayRestoreArgs(
                  restore_type=GlobalHostArray,
                  shard_id=shard_id,
                  num_shards=num_shards,
              )
          )
    else:
      restore_arg = GlobalHostArrayRestoreArgs(
          restore_type=GlobalHostArray,
          shard_id=shard_id,
          num_shards=num_shards,
      )
      embed_restore_args[tb_cfg.name] = {'parameters': restore_arg}
      for slot_name in tb_cfg.optimizer._slot_names():  # pylint: disable=protected-access
        embed_restore_args[tb_cfg.name][slot_name] = restore_arg

  return embed_restore_args


def create_tables_jax_array_restore_args(
    sharding: jax.sharding.NamedSharding,
    table_config_list: List[TableConfig],
) -> NestedStruct[ocp.ArrayRestoreArgs]:
  """Creates RestoreArgs for tables and slot variables.

  Args:
    sharding: NamedSharding corresponding to the embedding table shards
    table_config_list: A list of all table config.

  Returns:
    A nested dictionary of checkpoint.RestoreArgs.
  """
  embed_restore_args = {}
  for tb_cfg in table_config_list:
    restore_arg = JaxArrayRestoreArgs(
        sharding=sharding,
    )
    embed_restore_args[tb_cfg.name] = {'parameters': restore_arg}
    for slot_name in tb_cfg.optimizer._slot_names():  # pylint: disable=protected-access
      embed_restore_args[tb_cfg.name][slot_name] = restore_arg

  return embed_restore_args


def create_table_shape_dtype_struct(
    table_config_list: List[TableConfig],
    num_shards: int | None = None,
) -> NestedStruct[jax.ShapeDtypeStruct]:
  """Creates RestoreArgs for all table and slot variables.

  Args:
    table_config_list: A list of all table config.
    num_shards: If not None, it is the number of shards of table which stays in
      the leaf nodes of the shape dtype struct.

  Returns:
    A nested dictionary of jax.ShapeDtyeStruct.
  """
  embed_shape_dtypes = {}
  for tb_cfg in table_config_list:
    global_shape = (tb_cfg.vocabulary_size, tb_cfg.dim)

    shape_and_dtype = jax.ShapeDtypeStruct(global_shape, jnp.float32)
    if num_shards is not None:
      embed_shape_dtypes = {tb_cfg.name: {'parameters': {}}}
      for shard_id in range(num_shards):
        embed_shape_dtypes[tb_cfg.name]['parameters'][
            f'shard_{shard_id}'
        ] = shape_and_dtype
    else:
      embed_shape_dtypes[tb_cfg.name] = {'parameters': shape_and_dtype}
    for slot_name in tb_cfg.optimizer._slot_names():  # pylint: disable=protected-access
      embed_shape_dtypes[tb_cfg.name][slot_name] = shape_and_dtype

  return embed_shape_dtypes


def _get_single_distinct_tree_leaf_type(
    tree: NestedStruct[Union[GlobalHostArray, jax.Array]],
) -> type[Union[GlobalHostArray, jax.Array]]:
  """Gets the type of leaf nodes of the tree.

  Raise an error if the tree has multiple types of leaf nodes.

  Args:
    tree: A nested structure of GlobalHostArray or jax.Array.

  Returns:
    type(GlobalHostArray) if all leaf nodes are GlobalHostArray.
    type(jax.Array) if all leaf nodes are jax.Array.
  """
  types = set()
  jax.tree.map(lambda x: types.add(type(x)), tree)
  if len(types) != 1:
    raise ValueError(
        f'Expected a tree with a single type of leaf nodes, got {types}'
    )
  return types.pop()


def load_embedding_tables_impl(
    table_config_list: List[TableConfig],
    config_proto_str: bytes,
    host_id: int,
    num_hosts: int,
    embedding_variables: Optional[
        NestedStruct[Union[GlobalHostArray, jax.Array]]
    ] = None,
) -> None:
  """Load parameters and slot variables of embedding tables.

  Args:
    table_config_list: A list of tf.tpu.experimental.embedding.TableConfig.
    config_proto_str: Serialized string of tpu embedding configuration proto.
    host_id: This also represents shard id of table variables to load, as each
      host will only have one shard.
    num_hosts: This also represents num of shards of table variables.
    embedding_variables: If not None, it should be a nested dictionary of
      GlobalHostArray.
  """
  if embedding_variables is None:
    embedding_variables = create_tables_variables_from_config(
        shard_id=host_id,
        num_shards=num_hosts,
        table_config_list=table_config_list)
  else:
    var_type = _get_single_distinct_tree_leaf_type(embedding_variables)

    if issubclass(var_type, GlobalHostArray):
      embedding_variables = create_table_variables_from_gha(
          embedding_variables,
          shard_id=host_id,
          num_shards=num_hosts,
          table_config_list=table_config_list,
      )
    elif issubclass(var_type, jax.Array):
      embedding_variables = create_table_variables_from_jax_array(
          embedding_variables,
          num_shards=num_hosts,
          table_config_list=table_config_list,
      )
    else:
      raise ValueError(
          'Leaf nodes are expected either GlobalHostArray or jax.Array, got'
          f' {var_type}'
      )

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
    num_hosts: int) -> NestedStruct[tf.Tensor]:
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
                  num_hosts: int) -> NestedStruct[tf.Tensor]:
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


def get_new_config(
    config_proto: TPUEmbeddingConfigurationProto,
) -> bytes:
  """Gets the new config proto.

     The new config proto has additional fields set.

  Args:
    config_proto: A configuration proto.

  Returns:
    A new config in bytes.
  """
  # As input config proto needs field populating in `populate_config`, this
  # copy is to avoid to change original config proto.
  copied_proto = copy.deepcopy(config_proto)
  config_utils.set_additional_fields(copied_proto)

  logging.info('TPU Embedding Configuration: %s', copied_proto)
  new_config_str = copied_proto.SerializeToString()
  return new_config_str


def get_embedding_sharding(local_shard_id: int) -> jax.sharding.NamedSharding:
  """Broadcast local shard_id and collect global shard_id to create a NamedSharding.

  TPU Embedding shards and Jax devices are not ordered in the same way. This
  function aligns them across all hosts by broadcasting each local shard_id and
  its device_id, then creates a Jax NamedSharding with its CPU devices ordered
  by embedding shard_ids.

  Args:
    local_shard_id: The local shard id of the current host. This should be the
      same used by TPUEmbedding.

  Returns:
    jax.sharding.NamedSharding with its "cpu" device mesh ordered by shard id.
  """
  backend = 'cpu'
  if jax.process_count(backend=backend) != jax.device_count(backend=backend):
    raise ValueError(
        'Only support when process_count == device_count but got'
        f' process_count={jax.process_count(backend=backend)} and '
        f' device_count={jax.device_count(backend=backend)}'
    )

  if jax.process_count(backend=backend) == 1:
    # just return a local sharding
    return jax.sharding.NamedSharding(
        jax.sharding.Mesh(jax.devices(backend=backend), 'device'),
        jax.sharding.PartitionSpec('device'),
    )

  devices = jax.devices(backend=backend)
  local_device = jax.local_devices(backend=backend)[0]
  logging.info(
      'devices[%s]: %s, devices:%s, local_shard_id: %s',
      backend,
      local_device,
      devices,
      local_shard_id,
  )

  # broadcast and collect global shard_id to device_id
  local_array = jax.device_put(
      np.asarray([local_shard_id, local_device.id]), local_device
  )
  logging.info('local_array: %s', local_array)

  global_shard_to_device_id = multihost_utils.process_allgather(
      local_array, tiled=False
  )
  logging.info('global_shard_id_arr: %s', global_shard_to_device_id)

  # order by shard id
  ordered_shard_device_id = sorted(
      global_shard_to_device_id, key=lambda x: x[0]
  )
  logging.info('devices ordered by shard id: %s', ordered_shard_device_id)

  id_device_map = {device.id: device for device in jax.devices(backend=backend)}
  logging.info('id_device_map: %s', id_device_map)

  embedding_mesh = jax.sharding.Mesh(
      [id_device_map[device_id] for _, device_id in ordered_shard_device_id],
      'device',
  )
  logging.info(
      'embedding_mesh.devices: %s, embedding_mesh.axis_names: %s',
      embedding_mesh.devices,
      embedding_mesh.axis_names,
  )

  embedding_sharding = jax.sharding.NamedSharding(
      embedding_mesh, jax.sharding.PartitionSpec('device')
  )
  logging.info('NamedSharding: %s', embedding_sharding)

  return embedding_sharding
