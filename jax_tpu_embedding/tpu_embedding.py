# Copyright 2023 The jax_tpu_embedding Authors.
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
import dataclasses
import os
import re
from typing import List, Mapping, Optional, Sequence, Tuple

from absl import logging
import jax
from jax.experimental import jax2tf
import jax.numpy as jnp
from jax_tpu_embedding import config_utils
from jax_tpu_embedding import input_utils
from jax_tpu_embedding import pytype_utils
from jax_tpu_embedding import tpu_embedding_utils
import tensorflow as tf

from tensorflow.python.tpu.ops import gen_tpu_embedding_ops as tpu_ops  # pylint: disable=g-direct-tensorflow-import

TensorType = pytype_utils.TensorType
NestedTfTensor = pytype_utils.NestedTfTensor
Nested = pytype_utils.Nested
NestedFeatureConfig = config_utils.NestedFeatureConfig
TPUEmbeddingOptimizer = config_utils.Optimizer

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
    task_id_str, _, _ = _TASK_HANDLE_RE.match(handle).groups()  # pytype: disable=attribute-error  # re-none
    task_id = int(task_id_str)
  else:
    logging.warning("`BORG_TASK_HANDLE` is not found, using "
                    "`tf.experimental.dtensor.client_id()`.")
    task_id = tf.experimental.dtensor.client_id()
  return task_id


def _get_tuple_mask(config_str: bytes) -> pytype_utils.TensorProto:
  """Get deduplication data tuple mask.

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


@dataclasses.dataclass
class DeduplicationTuple:
  """Deduplication Tuple."""

  integers: tf.Tensor
  floats: tf.Tensor


@dataclasses.dataclass
class DequeueOutput:
  """Utility class to represent `TPUEmbedding.dequeue` output.

  User may consider to use TPUEmbedding software deduplication, as result of
  `tpu_ops.xla_recv_tpu_embedding_deduplication_data`. However, the result needs
  to be converted to integer and float tf.Tensor(s) as part of
  `TPUEmbedding.dequeue` output, which will be correspondingly used in
  `TPUEmbedding.apply_gradient`.

  Attributes:
    activations: TPUEmbedding activations results.
    deduplication_tuple: A `DeduplicationTuple` instance.
  """

  activations: NestedTfTensor
  deduplication_tuple: DeduplicationTuple


class TPUEmbedding(object):
  """The TPUEmbedding mid level API for Jax users."""

  def __init__(self,
               feature_configs: Optional[NestedFeatureConfig] = None,
               optimizer: Optional[TPUEmbeddingOptimizer] = None,
               pipeline_execution_with_tensor_core: bool = False,
               cores_per_replica: Optional[int] = None):
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
    if cores_per_replica and (
        cores_per_replica > jax.device_count() or cores_per_replica < 1):
      raise ValueError("`cores_per_replica = {}` is not allowed, legal range is"
                       "1 <= cores_per_replica <= {}".format(
                           cores_per_replica, jax.device_count()))
    # Setup device specs.
    self._host_id = _get_task_id()
    self._num_hosts = jax.process_count()

    self._feature_configs = feature_configs
    self._optimizer = optimizer

    # Create config_utils.TpuEmbeddingConfig instance.
    self._tpu_embedding_config = config_utils.create_tpu_embedding_config(
        feature_configs=self._feature_configs,
        optimizer=self._optimizer,
        pipeline_execution_with_tensor_core=pipeline_execution_with_tensor_core,
        num_hosts=self._num_hosts,
        num_tensor_cores=jax.device_count(),
        cores_per_replica=cores_per_replica)

    self._cores_per_replica = cores_per_replica
    self._table_config_list = self._tpu_embedding_config.table_config_list
    self._output_shapes = self._tpu_embedding_config.output_shapes
    self._dynamic_learning_rates = (
        self._tpu_embedding_config.dynamic_learning_rates
        )
    self._is_initialized = False

    # TPUEmbedding software deduplication tuple mask and data.
    self._dedup_tuple_mask = None
    self._dedup_tuple_tensors = None

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

    Also note that, when user need to make output shape inference, user should
    not explicitly call `initialize_tpu_embedding`. It will be called in side
    of `enqueue` after inferring output shapes from first input batch.

    The initialization would be like:

    ```python
    # Example 1: When not using shape inference `initialize_tpu_embedding` is .
    from jax_tpu_embedding import tpu_embedding_utils
    tpu_embedding_utils.init_tpu_system()
    initialize_tpu_embedding()

    # Example 2: When using shape inference, initialize_tpu_embedding is not
    explicit called, it will be executed after output shape inferred during
    first batch of data enqueued.

    tpu_embedding_utils.init_tpu_system()
    tpu_embedding_layer = TPUEmbedding(...)

    tpu_embedding_layer.enqueue(...)
    ```

    Raises:
      RuntimeError: If tpu embedding is already initialized on TPU.
    """
    self._config_proto = config_utils.create_config_proto(
        self._tpu_embedding_config)
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
    self._is_initialized = True
    logging.info("Successfully Initialized TPUEmbedding devices.")

    self._dedup_tuple_mask = _get_tuple_mask(
        self._config_proto.SerializeToString()
    )
    logging.info("Get deduplication tuple mask : %s", self._dedup_tuple_mask)

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

  def _is_empty_dedup_mask(self):
    return self._dedup_tuple_mask.tensor_shape.dim[0].size == 0

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

    def _gradients_fn(
        gradients: NestedTfTensor,
        dedup_tuple_integers: Optional[tf.Tensor] = None,
        dedup_tuple_floats: Optional[tf.Tensor] = None,
    ):
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

      # Merging integer tensor and float tensor in `dedup_tuple` to create
      # deduplication data. When tuple mask is empty, create empty tensors with
      # default types.
      if self._is_empty_dedup_mask():
        logging.info("Creates empty dedup data when dedup mask is empty.")
        dedup_tuple_integers = tf.constant((), dtype=tf.uint32)
        dedup_tuple_floats = tf.constant((), dtype=tf.float32)

      deduplication_data = tpu_ops.merge_dedup_data(
          integer_tensor=dedup_tuple_integers,
          float_tensor=dedup_tuple_floats,
          tuple_mask=self._dedup_tuple_mask.SerializeToString(),
          config=self._config_proto.SerializeToString(),
      )

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

    jax2tf.call_tf(_gradients_fn, has_side_effects=False)(
        gradients,
        self._dedup_tuple_tensors.integers,
        self._dedup_tuple_tensors.floats,
    )

  def dequeue(self, name: Optional[str] = None) -> NestedTfTensor:
    """Gets the embedding results with inputs non-duplicated.

    Args:
      name: A name for the underlying op.

    Returns:
      A nested structure of tensors, returned by jax2tf.call_tf call, with the
      same structure as `feature_config` passed to this instance of
      `TPUEmbedding` object.
    """

    def _dedup_fn():
      deduplication_data = tpu_ops.xla_recv_tpu_embedding_deduplication_data(
          config=self._config_proto.SerializeToString()
      )
      tuple_integers, tuple_floats = tpu_ops.split_dedup_data(
          deduplication_data,
          integer_type=tf.uint32,
          float_type=tf.float32,
          tuple_mask=self._dedup_tuple_mask.SerializeToString(),
          config=self._config_proto.SerializeToString(),
      )
      return tuple_integers, tuple_floats

    if self._is_empty_dedup_mask():
      # When dedup_tuple_mask is empty, `tpu_ops.split_dedup_data` will give two
      # shape zero tensors, which are not traced by jax XLA input tracing.
      # Therefore we create two dummy inputs with shape (1,) to return for
      # methods using these two as inputs.
      tuple_integers = jnp.zeros([1], dtype=jnp.uint32)
      tuple_floats = jnp.zeros([1], dtype=jnp.float32)
    else:
      tuple_integers, tuple_floats = jax2tf.call_tf(
          _dedup_fn, has_side_effects=False
      )()

    self._dedup_tuple_tensors = DeduplicationTuple(
        integers=tuple_integers, floats=tuple_floats
    )

    def _dequeue_fn(
        dedup_tuple_integers: Optional[tf.Tensor] = None,
        dedup_tuple_floats: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
      # Merging integer tensor and float tensor in `dedup_tuple` to create
      # deduplication data. When tuple mask is empty, create empty tensors with
      # default types.
      if self._is_empty_dedup_mask():
        logging.info("Creates empty dedup data when dedup mask is empty.")
        dedup_tuple_integers = tf.constant((), dtype=tf.uint32)
        dedup_tuple_floats = tf.constant((), dtype=tf.float32)

      deduplication_data = tpu_ops.merge_dedup_data(
          integer_tensor=dedup_tuple_integers,
          float_tensor=dedup_tuple_floats,
          tuple_mask=self._dedup_tuple_mask.SerializeToString(),
          config=self._config_proto.SerializeToString(),
      )

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

    return jax2tf.call_tf(_dequeue_fn, has_side_effects=False)(
        self._dedup_tuple_tensors.integers, self._dedup_tuple_tensors.floats
    )

  def enqueue(self,
              features: List[Nested[TensorType]],
              weights: Optional[List[Nested[TensorType]]] = None,
              is_training: bool = True,
              name: Optional[str] = None) -> None:
    """Enqueues id tensors for embedding lookup for all devices.

    This function enqueues a list of nested structure of features to be looked
    up in embedding tables. We expect the input shape of each feature to matche
    the `output_shape` defined in FeatureConfig. When TPU Embedding is not
    initialized, then when first time enqueue is called, it will initialize
    TPU Embedding as part of this `enqueue` function.

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
    if not self._is_initialized:
      # Use output shapes inference mode when feature_configs did not provide
      # output_shape. `output_shape` of each feature_config, i.e. each instance
      # of tf.tpu.experimental.embedding.FeatureConfig, is to configure output
      # shape of the feature activation but without last embedding dimension.
      # For example, if we expect output feature activation shape is [B, N, E]
      # where E is embedding dimension, the output shape should be [B, N].

      inferred_output_shapes = input_utils.infer_output_shapes(
          features[0], self._feature_configs
      )
      self._tpu_embedding_config.output_shapes = inferred_output_shapes
      self._output_shapes = inferred_output_shapes
      logging.info(
          "Using shape inference mode, initializing embedding layer and loading"
          " embedding weights in first enqueue call."
      )
      self.initialize_tpu_embedding()
      self.load_embedding_tables()
      self._is_initialized = True

    self.enqueue_fn_call(features=features,
                         weights=weights,
                         is_training=is_training,
                         name=name)

  @tf.function
  def enqueue_fn_call(self,
                      features: List[Nested[TensorType]],
                      weights: Optional[List[Nested[TensorType]]] = None,
                      is_training: bool = True,
                      name: Optional[str] = None) -> None:  # pylint:disable=g-doc-args
    """tf.function call of enqueue().

    This function enqueues a list of nested structure of features to be looked
    up in embedding tables. We expect the input shape of each feature to matche
    the `output_shape` defined in FeatureConfig.

    Args are same as the `enqueue` function.
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
    flat_feature_with_names, configs_treedef = (
        input_utils.tree_flatten_with_names(self._feature_configs)
    )
    for device_id in range(jax.local_device_count()):
      flat_inputs, inputs_treedef = jax.tree_util.tree_flatten(
          features[device_id])

      if inputs_treedef != configs_treedef:
        raise ValueError(f"Expects `flat_inputs` has the same tree structure as"
                         f" `self.feature_configs`, inputs = {inputs_treedef}"
                         f" feature configs = {configs_treedef}")

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
    return self._config_proto  # pytype: disable=attribute-error

  @property
  def feature_configs(self) -> Optional[NestedFeatureConfig]:
    return self._feature_configs

  @property
  def optimizer(self) -> Optional[TPUEmbeddingOptimizer]:
    return self._optimizer
