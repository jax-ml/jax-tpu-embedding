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

"""Utils for using Coordination Service for TPU embedding."""

from typing import List, Tuple

from absl import flags
from absl import logging
from jax._src import distributed
from jax._src.lib import xla_extension
from jax_tpu_embedding.google import borg_utils
import tensorflow as tf

from tensorflow.python.tpu.ops import gen_tpu_embedding_ops as tpu_ops  # pylint: disable=g-direct-tensorflow-import

global_state = distributed.global_state
DistributedRuntimeClient = xla_extension.DistributedRuntimeClient
DistributedRuntimeService = xla_extension.DistributedRuntimeService

_COORDINATION_SERVICE_TIMEOUT = flags.DEFINE_integer(
    "coordination_service_timeout",
    100,
    "The timeout value in seconds for the API of getting key-value pair in"
    " Coordination Service.",
)


def init_coordination_service(
    shard_index: int,
    num_shards: int,
    coordinator_shard: int,
    coordinator_bind_address: str,
    coordinator_address: str,
) -> Tuple[
    DistributedRuntimeService | None,
    DistributedRuntimeClient,
]:
  """Initializes Coordination Service.

  The service is started at `coordinator_shard`, and clients are created and
  connect to the service in all the shards.

  Args:
    shard_index: Index of the shard in Pathways.
    num_shards: Number of shards in Pathways.
    coordinator_shard: The shard index of the coordinator.
    coordinator_bind_address: The network address of the coordinator to bind to.
    coordinator_address: The network address of the coordinator task which all
      the coordination clients can connect to.

  Returns:
    A pair of Coordination Service and client.
  """
  coordination_service = None
  start_coordination_service = False
  if shard_index == coordinator_shard:
    start_coordination_service = True
  if start_coordination_service:
    logging.info(
        "Starting Coordination Service at %s", coordinator_bind_address
    )
    coordination_service = xla_extension.get_distributed_runtime_service(
        coordinator_bind_address, num_shards
    )
  logging.info("Connecting to Coordination Service at %s", coordinator_address)
  coordination_client = xla_extension.get_distributed_runtime_client(
      coordinator_address, shard_index
  )
  coordination_client.connect()
  return coordination_service, coordination_client


def _all_gather_configs(
    config_type: str,
    local_config_bytes: bytes,
    timeout_in_sec: int,
    current_task: int,
    num_tasks: int,
    client: DistributedRuntimeClient,
) -> List[bytes]:
  """All gather configs from each client.

  Args:
    config_type: A string that to claim config type like `memory` or `network`
      it will be used to create key of config by client id to store and lookup
      in coordination service.
    local_config_bytes: A bytes to store in coordination service.
    timeout_in_sec: Timeout seconds when get value from coordination service.
    current_task: ID of the current task.
    num_tasks: Number of tasks with Coordination Service deployed.
    client: The client of Coordination Service.

  Returns:
    A list all `local_config_bytes` value put into coordination service by each
    client.
  """

  def _get_config_key(config_type: str, pid: int) -> str:
    """Create configure key for each process."""
    return "{type}_config_by_process_{id}".format(type=config_type, id=pid)

  # Get value for a given key store into coordination service.
  def _get_key_value(key: str) -> bytes:
    # Checking `blocking_key_value_get_bytes` API for backwards compatibility
    # And falling back to blocking_key_value_get.
    if hasattr(client, "blocking_key_value_get_bytes"):
      return client.blocking_key_value_get_bytes(
          key=key, timeout_in_ms=timeout_in_sec * 1000
      )
    else:
      # TODO(b/344016223): remove blocking_key_value_get fallback when most
      # users migrate to Jax 0.4.5+.
      gathered_config_str = client.blocking_key_value_get(
          key=key, timeout_in_ms=timeout_in_sec * 1000
      )
      # Here and following we encode local_config_bytes to `cp437` due to utf-8
      # or ascii decoding would not return results decoded same as original.
      return gathered_config_str.encode("cp437")

  # Add key value store into coordination service.
  def _set_key_value(key: str, value: bytes) -> None:
    # TODO(b/344016223): Remove the fallbacks when the min supported Jax version
    # is 0.4.5+.
    if hasattr(client, "key_value_set_bytes"):
      client.key_value_set_bytes(key=key, value=value)
    elif hasattr(client, "blocking_key_value_get_bytes"):
      client.key_value_set(key=key, value=value)  # pytype: disable=wrong-arg-types
    else:
      client.key_value_set(key=key, value=local_config_bytes.decode("cp437"))

  _set_key_value(
      key=_get_config_key(config_type, current_task), value=local_config_bytes
  )

  all_configs = [b"" for _ in range(num_tasks)]
  for pid in range(num_tasks):
    if pid == current_task:
      all_configs[pid] = local_config_bytes
    else:
      all_configs[pid] = _get_key_value(key=_get_config_key(config_type, pid))

  return all_configs


def maybe_all_gather_configs(
    config_type: str,
    local_config: bytes,
    current_task: int,
    num_tasks: int,
    client: DistributedRuntimeClient | None = None,
) -> List[bytes]:
  """When there is more than one process, apply `_all_gather_configs`.

  Args:
    config_type: A string that to claim config type
    local_config: A bytes from local client.
    current_task: ID of the current task.
    num_tasks: Number of tasks with Coordination Service deployed.
    client: The client of Coordination Service.

  Returns:
    A list of gathered local_config from all processes, when there is only one
    process, the list only has input local_config.
  """
  if client is None:
    client = global_state.client
  if num_tasks > 1:
    return _all_gather_configs(
        config_type=config_type,
        local_config_bytes=local_config,
        timeout_in_sec=_COORDINATION_SERVICE_TIMEOUT.value,
        current_task=current_task,
        num_tasks=num_tasks,
        client=client,
    )
  return [local_config]


def maybe_update_task_id_and_global_core_array(
    current_task: int,
    num_tasks: int,
    client: DistributedRuntimeClient | None = None,
):
  """When there is more than one process, update task ID and global core array.

  Args:
    current_task: ID of the current task.
    num_tasks: Number of tasks with Coordination Service deployed.
    client: The client of Coordination Service.
  """
  if client is None:
    client = global_state.client
  if num_tasks > 1:
    this_tpu_task_id = tpu_ops.get_tpu_task_id()
    this_tpu_task_id = str(this_tpu_task_id.numpy()).encode("cp437")
    shard_id_to_tpu_task_id = _all_gather_configs(
        config_type="task_id",
        local_config_bytes=this_tpu_task_id,
        timeout_in_sec=_COORDINATION_SERVICE_TIMEOUT.value,
        current_task=current_task,
        num_tasks=num_tasks,
        client=client,
    )
    tpu_task_id_to_shard_id = [0] * num_tasks
    for shard_id, tpu_task_id in enumerate(shard_id_to_tpu_task_id):
      tpu_task_id = int(tpu_task_id.decode("cp437"))
      assert tpu_task_id >= 0 and tpu_task_id < num_tasks
      tpu_task_id_to_shard_id[tpu_task_id] = shard_id
    tpu_ops.update_task_id_and_global_core_array(tpu_task_id_to_shard_id)


def initialize_fn(
    config_str: bytes,
    current_task: int,
    num_tasks: int,
    client: xla_extension.DistributedRuntimeClient | None = None,
    use_v2: bool = False,
) -> Tuple[tf.Tensor, tf.Tensor] | None:
  """TF function for initializing tpu embedding rewrite.

  Reusing logic from `tpu/graph_rewrite/configure_tpu_embedding_rewrite_pass.cc`
  to lower tpu_embedding_initialize to:
  1) execute_tpu_embedding_partitioner: to compute common config for each host.
  2) configure_tpu_embedding_memory: hbm memory configuration.
  3) collate_tpu_embedding_memory to merge memory configuration of each host.
  4) configure_tpu_embedding_host: configure tpu embedding on each host.
  5) connect_tpu_embedding_hosts: connect hosts with output from step 4) which
    describe metadata of that host.
  6) finalize_tpu_embedding: update tpu system with results of initialization.
  As step 3) and 5) needs intermediate configurations from other hosts, we need
  apply all gather for these configurations on each host.

  Also, for better performance, these ops need to run in graph/tf.function.
  Therefore, we have three inner functions to execute all 6 steps above.

  Args:
    config_str: Serialized tpu embedding config proto string.
    current_task: ID of the current task.
    num_tasks: Number of tasks with Coordination Service deployed.
    client: The client of Coordination Service.
    use_v2: Whether to use V2 Ops. In this case the Op for finalizing TPU
      embedding.

  Returns:
    Embedding partitions and HBM buffers config when using
    `finalize_tpu_embedding_v2`.
  """

  @tf.function
  def create_memory_config(config_str):
    """Execute embedding partitioner and configure memory for embedding.

    `execute_tpu_embedding_partitioner` is to run the embedding engine
    partitioner as well as calculate the HBM size (in bytes) required for
    embedding engine operation.
    `configure_tpu_embedding_memory` is to initialize the HBM memory addresses
    and segments on each host, allocating HBM memory used by embedding engine.

    Args:
      config_str: Serialized tpu embedding configuration string.

    Returns:
      common_config: An encoded string  proto containing meta data about TPU
        Embedding partitioner output and HBM size required.
      memory_config: HbmBuffer configuration containing metadata about memory
        allocations reserved for tpu embedding.
    """
    common_config = tpu_ops.execute_tpu_embedding_partitioner(config_str)
    memory_config = tpu_ops.configure_tpu_embedding_memory(common_config)
    return common_config, memory_config

  @tf.function
  def create_network_config(common_config, memory_configs, config_str):
    """Merge memory configs and configure TPUEmbedding host software.

    `collate_tpu_embedding_memory` merges the memory configurations of all hosts
    into one. `configure_tpu_embedding_host` is to set up the embedding engine
    host software on a given host.

    Args:
      common_config: An encoded string proto contains meta data about TPU
        Embedding partitioner output and HBM size required.
      memory_configs: A list of HbmBuffer configuration from all hosts.
      config_str: Serialized tpu embedding configuration string.

    Returns:
      merged_memory_config: An encoded string of HbmBuffer configuration protos
        containing metadata about memory allocations for TPUEmbedding across
        all hosts.
      network_config: A string contains metadata about the hostname and RPC port
        used for communication with this host.
    """
    merged_memory_config = tpu_ops.collate_tpu_embedding_memory(memory_configs)
    network_config = tpu_ops.configure_tpu_embedding_host(
        common_config, merged_memory_config, config_str
    )
    return merged_memory_config, network_config

  @tf.function
  def connect_embedding_hosts(
      network_configs, common_config, merged_mem_config, use_v2
  ) -> Tuple[tf.Tensor, tf.Tensor] | None:
    """Connect each host and update global tpu embedding setting.

    `connect_tpu_embedding_hosts` is to set up gRPC connections between host
    software of embedding engine on each host. `finalize_tpu_embedding` is used
    to update TpuMeshCommonState and TpuSystemConfiguration objects with the
    results of the TPU embedding initialization.

    Args:
      network_configs: A list of network configs on each host.
      common_config: An encoded string proto contains meta data about TPU
        Embedding partitioner output and HBM size required. This is to update
        TPU embedding engine setup in `finalize_tpu_embedding`.
      merged_mem_config: A encoded string proto containing metadata about the
        memory allocations reserved for TPUEmbedding over all hosts.
      use_v2: Whether to use V2 Ops. In this case the Op for finalizing TPU
        embedding.

    Returns:
      Embedding partitions and HBM buffers config if `use_v2` is true.
    """
    tpu_ops.connect_tpu_embedding_hosts(network_configs)
    if use_v2:
      return tpu_ops.finalize_tpu_embedding_v2(common_config, merged_mem_config)
    else:
      tpu_ops.finalize_tpu_embedding(common_config, merged_mem_config)

  if flags.FLAGS.create_tpu_embedding_states_from_global_tpu_system:
    maybe_update_task_id_and_global_core_array(
        current_task=current_task,
        num_tasks=num_tasks,
        client=client,
    )
  common_config, mem_config = create_memory_config(config_str)

  # Gather other memory configs to merge when there are multi clients.
  all_mem_configs = maybe_all_gather_configs(
      config_type="memory",
      local_config=mem_config.numpy(),
      current_task=current_task,
      num_tasks=num_tasks,
      client=client,
  )

  merged_memory_config, network_config = create_network_config(
      common_config, all_mem_configs, config_str
  )

  # Gather other network configs to connect when there are multi clients.
  all_network_configs = maybe_all_gather_configs(
      config_type="network",
      local_config=network_config.numpy(),
      current_task=current_task,
      num_tasks=num_tasks,
      client=client,
  )

  return connect_embedding_hosts(
      all_network_configs, common_config, merged_memory_config, use_v2
  )
