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

"""Jax Mid level API for TPU Embedding."""

import copy
import os
import re
from typing import Optional, Mapping, Sequence, List, Tuple

from absl import logging
import jax
from jax.experimental import jax2tf
from jax_tpu_embedding import config_utils
from jax_tpu_embedding import input_utils
from jax_tpu_embedding import pytype_utils
from jax_tpu_embedding import tpu_embedding_utils
import tensorflow as tf

from tensorflow..python.tpu.ops import gen_tpu_embedding_ops as tpu_ops  # pylint: disable=g-direct-tensorflow-import

TensorType = pytype_utils.TensorType
NestedTfTensor = pytype_utils.NestedTfTensor
Nested = pytype_utils.Nested

# Regular expression handler to extract task id from environment variable.
_TASK_HANDLE_RE = re.compile(r"(?:logs\.)?(\d+)\.(.*)\.([^.]+)\.\d+")


def _add_key_attr(op: tf.Operation, name: str):
  """Set tpu embedding layer attribute name in given op node.

  Args:
    op: an op to set tpu embedding layer attribute name.
    name: A name for the underlying op.
  """
  op._set_attr(  # pylint: disable=protected-access
      "_tpu_embedding_layer",
      tf.compat.v1.AttrValue(s=tf.compat.as_bytes(name)))


def _initialize_fn(config_str: bytes) -> None:
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
        common_config, merged_memory_config, config_str)
    return merged_memory_config, network_config

  @tf.function
  def connect_embedding_hosts(network_configs, common_config,
                              merged_mem_config):
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
    """
    tpu_ops.connect_tpu_embedding_hosts(network_configs)
    tpu_ops.finalize_tpu_embedding(common_config, merged_mem_config)

  common_config, mem_config = create_memory_config(config_str)

  # Gather other memory configs to merge when there are multi clients.
  all_mem_configs = config_utils.maybe_all_gather_configs(
      config_type="memory", local_config=mem_config.numpy())

  merged_memory_config, network_config = create_network_config(
      common_config, all_mem_configs, config_str)

  # Gather other network configs to connect when there are multi clients.
  all_network_configs = config_utils.maybe_all_gather_configs(
      config_type="network", local_config=network_config.numpy())

  connect_embedding_hosts(all_network_configs, common_config,
                          merged_memory_config)


def _get_task_id() -> int:
  """Get TPU task id."""

  task_id = None
  if "BORG_TASK_HANDLE" in os.environ:
    handle = os.getenv("BORG_TASK_HANDLE")
    task_id_str, _, _ = _TASK_HANDLE_RE.match(handle).groups()
    task_id = int(task_id_str)
  else:
    logging.warning("`BORG_TASK_HANDLE` is not found, using "
                    "`tf.experimental.dtensor.client_id()`.")
    task_id = tf.experimental.dtensor.client_id()
  return task_id


class TPUEmbedding(object):
  """The TPUEmbedding mid level API for Jax users."""

  def __init__(self,
               feature_configs: config_utils.NestedFeatureConfig,
               optimizer: Optional[config_utils.Optimizer] = None,
               pipeline_execution_with_tensor_core: bool = False,
               cores_per_replica: int = 1):
    """Creates jax TPUEmbedding object.

    Args:
      feature_configs: A nested structure or a standalone instance of
        `tf.tpu.experimental.embedding.FeatureConfig` configs.
      optimizer: An instance of one of embedding optimizers like
        `tf.tpu.experimental.embedding.SGD`,
        `tf.tpu.experimental.embedding.Adam` etc.
      pipeline_execution_with_tensor_core: If True, the TPU embedding
        computations will overlap with the TensorCore computations (and hence
        will be one step old). Set to True for improved performance.
      cores_per_replica: default 1 using pmap. If it is greater than, it will
        enable embedding engine spmd. Note that range of is
        [1, total_tensor_cores], i.e. the maximum is all avaibale cores.
        For pjit users, always set this based on cores for each model replica.
        For pjit data parallelism user, it should be set as jax.device_count().

    Raises:
      ValueError: when cores_per_replica is not legal.
    """
    if cores_per_replica > jax.device_count() or cores_per_replica < 1:
      raise ValueError("`cores_per_replica = {}` is not allowed, legal range is"
                       "1 <= cores_per_replica <= {}".format(
                           cores_per_replica, jax.device_count()))

    # Get current host id.
    self._host_id = _get_task_id()
    self._num_hosts = jax.process_count()

    # Create config_utils.TpuEmbeddingConfig instance.
    tpu_embedding_config = config_utils.create_tpu_embedding_config(
        feature_configs=feature_configs,
        optimizer=optimizer,
        pipeline_execution_with_tensor_core=pipeline_execution_with_tensor_core,
        num_hosts=self._num_hosts,
        num_tensor_cores=jax.device_count(),
        cores_per_replica=cores_per_replica)

    self._cores_per_replica = cores_per_replica
    self._table_config_list = tpu_embedding_config.table_config_list
    self._config_proto = config_utils.create_config_proto(tpu_embedding_config)
    self._feature_configs = feature_configs
    self._output_shapes = tpu_embedding_config.output_shapes
    self._dynamic_learning_rates = tpu_embedding_config.dynamic_learning_rates

  def initialize_tpu_embedding(self):
    """Initializae tpu embedding.

    Check if tpu embedding system is already initialized. If not, initialize
    with `initialize_fn`.

    Note: Currently jax tpu system initialization does not include
    1) tpu_mesh_state for storing TPU topology and HBM buffer configurations,
    also 2) tpu_embedding_engine_state. Both of them needs to be set before call
    this method, otherwise tpu system precondition errors will be raised.
    Therefore at this point, it requests to call dtensor's tpu system initialize
    which is also multi-client setup before calling this method.

    It is important to disable coordination service on from DTensor to avoid
    double initialization of the service.

    The initialization would be like:

    ```python
    tf.experimental.dtensor.initialize_accelerator_system('TPU',
      enable_coordination_service=False)
    initialize_tpu_embedding(config_proto)
    ```

    Raises:
      RuntimeError: If tpu embedding is already initialized on TPU.
    """
    if tpu_ops.is_tpu_embedding_initialized():
      raise RuntimeError(
          "TPU is already initialized for embeddings. This may be caused by "
          "using multiple TPUEmbedding instances in a TPU scope which is "
          "unsupported.")

    # As input config proto needs field populating in `populate_config`, this
    # copy is to avoid to change original config proto.
    copied_proto = copy.deepcopy(self._config_proto)
    config_utils.set_additional_fields(copied_proto)

    logging.info("TPU Embedding Configuration: %s", str(copied_proto))
    _initialize_fn(copied_proto.SerializeToString())

  def load_embedding_tables(self) -> None:
    """Load tables' variables on the embedding engine to given set of tensors.
    """
    tpu_embedding_utils.load_embedding_tables_impl(
        table_config_list=self._table_config_list,
        config_proto_str=self._config_proto.SerializeToString(),
        host_id=self._host_id,
        num_hosts=self._num_hosts)

  def retrieve_embedding_tables(self) -> Mapping[str, Mapping[str, TensorType]]:
    """Retrieve tables variables from embedding engine.

    Returns:
      A dict of dict of list of tensors. The outer dict is indexed by table's
      name. The inner dict is indexed by slot names and the final list is the
      list of per host tensors.
    """
    return tpu_embedding_utils.retrieve_embedding_tables_impl(
        table_config_list=self._table_config_list,
        config_proto_str=self._config_proto.SerializeToString(),
        host_id=self._host_id,
        num_hosts=self._num_hosts)

  def apply_gradients(self,
                      gradients: NestedTfTensor,
                      name: Optional[str] = None):
    """Applies the gradient update to the embedding tables.

    Args:
      gradients: A nested structure of gradients, with structure matching the
        `feature_config` passed to this object.
      name: A name for the underlying op.

    Raises:
      ValueError: If a non-`tf.Tensor` non-`None` gradient is passed in, or a
        `tf.Tensor` of the incorrect shape is passed in. Also if
        the size of any sequence in `gradients` does not match corresponding
        sequence in `feature_config`.
      TypeError: If the type of any sequence in `gradients` does not match
        corresponding sequence in `feature_config`.
    """

    def _gradients_fn(gradients: NestedTfTensor):
      tf.nest.assert_same_structure(self._feature_configs, gradients)
      updated_gradients = []
      gradients_with_names, _ = input_utils.tree_flatten_with_names(gradients)
      flatten_feature_configs, _ = jax.tree_util.tree_flatten(
          self._feature_configs)
      for (path,
           gradient), feature, output_shape in zip(gradients_with_names,
                                                   flatten_feature_configs,
                                                   self._output_shapes):
        full_output_shape = list(output_shape) + [feature.table.dim]
        if gradient is not None and not isinstance(gradient, tf.Tensor):
          raise ValueError(
              f"found non-tensor type: {type(gradient)} at path {path}.")
        if gradient is not None:
          local_shape = gradient.shape.as_list()

          # When self._core_per_replica is not None, it uses BC spmd.
          if self._cores_per_replica:
            local_shape[0] = local_shape[0] // self._cores_per_replica
          if local_shape != full_output_shape:
            raise ValueError("Found gradient of shape {} at path {}. Expected "
                             "shape {}.".format(gradient.shape, path,
                                                full_output_shape))
        else:
          # No gradient for this feature, since we must give a gradient for all
          # features, pass in a zero tensor here. Note that this is not correct
          # for all optimizers.
          logging.warning(
              "No gradient passed for feature %s, sending zero "
              "gradient. This may not be correct behavior for certain "
              "optimizers like Adam.", path)
          gradient = tf.zeros(full_output_shape, dtype=tf.float32)
        # Some gradients can be passed with op which shape is not correctly set.
        # This ensures that the shape of the gradient is correctly set.
        updated_gradients.append(tf.reshape(gradient, shape=gradient.shape))

      # Here we need compute deduplication data for apply_gradients and dequeue
      # as we are using call2tf to compile, to assure input is not out of scope.
      deduplication_data = tpu_ops.xla_recv_tpu_embedding_deduplication_data(
          config=self._config_proto.SerializeToString())
      op = tpu_ops.xla_send_tpu_embedding_gradients(
          gradients=updated_gradients,
          learning_rates=[
              tf.cast(fn(), dtype=tf.float32)
              for fn in self._dynamic_learning_rates
          ],
          deduplication_data=deduplication_data,
          config=self._config_proto.SerializeToString())

      # Apply the name tag to the op.
      if name is not None:
        _add_key_attr(op, name)

    jax2tf.call_tf(_gradients_fn)(gradients)

  def dequeue(self, name: Optional[str] = None) -> NestedTfTensor:
    """Gets the embedding results with inputs non-duplicated.

    Args:
      name: A name for the underlying op.

    Returns:
      A nested structure of tensors, returned by jax2tf.call_tf call, with the
      same structure as `feature_config` passed to this instance of
      `TPUEmbedding` object.
    """

    def _dequeue_fn() -> NestedTfTensor:
      deduplication_data = tpu_ops.xla_recv_tpu_embedding_deduplication_data(
          config=self._config_proto.SerializeToString())

      # The activations returned by this op are per feature.
      # here num_tables is num_outputs when using feature descriptor
      activations = tpu_ops.xla_recv_tpu_embedding_activations(
          deduplication_data=deduplication_data,
          num_tables=len(self._config_proto.feature_descriptor),
          config=self._config_proto.SerializeToString())

      # Apply the name tag to the op.
      if name is not None:
        _add_key_attr(activations[0].op, name)

      # Pack the list back into the same nested structure as the features.
      return tf.nest.pack_sequence_as(self._feature_configs, activations)

    return jax2tf.call_tf(_dequeue_fn)()

  def enqueue(self,
              features: List[Nested[TensorType]],
              weights: Optional[List[Nested[TensorType]]] = None,
              is_training: bool = True,
              name: Optional[str] = None) -> None:
    """Enqueues id tensors for embedding lookup for all devices.

    This function enqueues a list of nested structure of features to be looked
    up in embedding tables. We expect the input shape of each feature to matche
    the `output_shape` defined in FeatureConfig.

    Args:
      features: A list of nested structure of `tf.Tensor`s, `tf.SparseTensor`s
        with the **same** structure as `feature_config` for each device to
        enqueue, for example if `feature_config` is tuple or list, each one of
        `features[device_id]` should be in same order.
      weights: If not `None`, a list nested structure of `tf.Tensor`s,
        `tf.SparseTensor`s matching `features above, for each device as well.
      is_training: Defaults to `True`. If `False`, enqueue the batch as
        inference batch (forward pass only). Do not call `apply_gradients` when
        this is `False` as this may lead to a deadlock.
       name: An optional name for the underlying op.
    """

    def _generate_enqueue_op(flat_inputs: List[TensorType],
                             flat_weights: Optional[List[TensorType]],
                             flat_features: Sequence[Tuple[
                                 str, config_utils.FeatureConfig]],
                             device_ordinal: int,
                             mode_override: str) -> tf.Operation:
      """Generate correspoding enqueue op for given device."""
      # Combiners of each table, listed in the same order as table configs.
      combiners = [
          table_config.combiner for table_config in self._table_config_list
      ]

      # sample_indices for sparse.
      enqueue_inputs = input_utils.prepare_data_to_enqueue(
          flat_inputs, flat_weights, flat_features)
      indices, values, weights = (list(attrs) for attrs in zip(*enqueue_inputs))

      return tpu_ops.enqueue_tpu_embedding_arbitrary_tensor_batch(
          sample_indices_or_row_splits=indices,
          embedding_indices=values,
          aggregation_weights=weights,
          mode_override=mode_override,
          device_ordinal=device_ordinal,
          combiners=combiners)

    mode_override = "train" if is_training else "inference"
    flat_feature_with_names, configs_treedef = input_utils.tree_flatten_with_names(
        self._feature_configs)
    for device_id in range(jax.local_device_count()):
      flat_inputs, inputs_treedef = jax.tree_util.tree_flatten(
          features[device_id])

      if inputs_treedef != configs_treedef:
        raise ValueError("Expects `flat_inputs` has the same tree structure as "
                         "`self.feature_configs`.")

      flat_weights = [None] * len(flat_inputs)
      if weights is not None:
        flat_weights, weights_treedef = jax.tree_util.tree_flatten(
            weights[device_id])
        assert inputs_treedef == weights_treedef

      enqueue_op = _generate_enqueue_op(
          flat_inputs=flat_inputs,
          flat_weights=flat_weights,
          flat_features=flat_feature_with_names,
          device_ordinal=device_id,
          mode_override=mode_override)
      # Apply the name tag to the op.
      if name is not None:
        _add_key_attr(enqueue_op, name)

  @property
  def config_proto(self) -> config_utils.TPUEmbeddingConfigurationProto:
    return self._config_proto
