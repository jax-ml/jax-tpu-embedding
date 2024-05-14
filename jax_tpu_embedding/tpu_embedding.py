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

"""Jax Mid level API for TPU Embedding."""

import dataclasses
import functools
import os
import re
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from absl import logging
import jax
from jax.experimental import jax2tf
import jax.numpy as jnp
from jax_tpu_embedding import config_utils
from jax_tpu_embedding import coordination_service_utils
from jax_tpu_embedding import input_utils
from jax_tpu_embedding import pytype_utils
from jax_tpu_embedding import tpu_embedding_utils
from jax_tpu_embedding.google import tpu_embedding_pathways_utils
import numpy as np
import tensorflow as tf

from tensorflow.python.tpu.ops import gen_tpu_embedding_ops as tpu_ops  # pylint: disable=g-direct-tensorflow-import

GlobalHostArray = pytype_utils.GlobalHostArray
TensorType = pytype_utils.TensorType
NestedTfTensor = pytype_utils.NestedTfTensor
Nested = pytype_utils.Nested
NestedStruct = pytype_utils.NestedStruct
TPUEmbeddingConfigurationProto = pytype_utils.TPUEmbeddingConfigurationProto
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


def _get_task_id() -> int:
  """Get TPU task id."""

  if "BORG_TASK_HANDLE" in os.environ:
    handle = os.getenv("BORG_TASK_HANDLE")
    task_id_str, _, _ = _TASK_HANDLE_RE.match(handle).groups()  # pytype: disable=attribute-error  # re-none
    task_id = int(task_id_str)
  else:
    logging.info(
        "`BORG_TASK_HANDLE` is not found, using `jax.process_index()`."
    )
    task_id = jax.process_index()
  return task_id


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

  def __init__(
      self,
      feature_configs: Optional[NestedFeatureConfig] = None,
      optimizer: Optional[TPUEmbeddingOptimizer] = None,
      pipeline_execution_with_tensor_core: bool = False,
      cores_per_replica: Optional[int] = None,
      use_pathways: bool = False,
      coordinator_address: str | None = None,
      input_split_fn: Callable[..., dict[str, Any]] | None = None,
  ):
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
        enable embedding engine spmd. Note that range of is [1,
        total_tensor_cores], i.e. the maximum is all avaibale cores. For pjit
        users, always set this based on cores for each model replica. For pjit
        data parallelism user, it should be set as jax.device_count().
      use_pathways: Whether to use Pathways as the backend.
      coordinator_address: The network address of the coordinator task which all
        the coordination clients can connect to.
      input_split_fn: A callable function takes elements from iterator, yields
        splits pytree of host and device batches in a dictionary. This should be
        supplied if users want to call `experimental_get_next`.

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
    self._num_tensor_cores = jax.device_count()

    self._feature_configs = feature_configs
    self._optimizer = optimizer
    self._pipeline_execution_with_tensor_core = (
        pipeline_execution_with_tensor_core
    )
    self._cores_per_replica = cores_per_replica

    # Create config_utils.TpuEmbeddingConfig instance.
    self._tpu_embedding_config = config_utils.create_tpu_embedding_config(
        feature_configs=self._feature_configs,
        optimizer=self._optimizer,
        pipeline_execution_with_tensor_core=self._pipeline_execution_with_tensor_core,
        num_hosts=self._num_hosts,
        num_tensor_cores=self._num_tensor_cores,
        cores_per_replica=self._cores_per_replica)

    self._cores_per_replica = cores_per_replica
    self._table_config_list = self._tpu_embedding_config.table_config_list
    self._output_shapes = self._tpu_embedding_config.output_shapes
    self._output_shapes_ready = False
    self._dynamic_learning_rates = (
        self._tpu_embedding_config.dynamic_learning_rates
        )
    self._is_initialized = False
    self._config_proto = None
    self._config_proto_ready = False

    # TPUEmbedding software deduplication tuple mask and data.
    self._dedup_tuple_mask = None
    self._dedup_tuple_tensors = None

    # Create restore args for checkpoint restoration.
    self._embedding_restore_args = (
        tpu_embedding_utils.create_tables_restore_args(
            shard_id=self._host_id,
            num_shards=self._num_hosts,
            table_config_list=self._table_config_list,
        )
    )
    # Create shape and dtype for checkpoint restoration.
    self._embedding_shape_dtypes = (
        tpu_embedding_utils.create_table_shape_dtype_struct(
            table_config_list=self._table_config_list
        )
    )
    self._use_pathways = use_pathways
    self._coordinator_address = coordinator_address
    self._num_local_devices = jax.local_device_count()
    self._tpu_topology = tpu_embedding_pathways_utils.get_tpu_topology()
    self._embedding_partitions = None
    self._embedding_partitions_ready = False
    self._hbm_buffers_config = None
    self._hbm_buffers_config_ready = False
    self._input_split_fn = input_split_fn
    self._packer = None

  def _remote_initialize_tpu_embedding(
      self,
      config_proto: TPUEmbeddingConfigurationProto,
      embedding_manager: tpu_embedding_pathways_utils.EmbeddingManager,
  ):
    """Initializes TPU embedding using remote Python.

    Args:
      config_proto: A configuration proto.
      embedding_manager: The object to manage the remote Python execution.

    Raises:
      RuntimeError: If TPU embedding is already initialized.
    """
    assert self._coordinator_address is not None
    new_config_str = tpu_embedding_utils.get_new_config(config_proto)
    embedding_manager.init_embedding_config(
        new_config_str,
        self._num_hosts,
        self._coordinator_address,
    )
    if embedding_manager.is_initialized()[0]:
      raise RuntimeError(
          "TPU is already initialized for embeddings. This may be caused by "
          "using multiple TPUEmbedding instances in a TPU scope which is "
          "unsupported."
      )

    embedding_manager.initialize()
    self._embedding_partitions = np.asanyarray(
        embedding_manager.embedding_partitions()[0], dtype=np.bytes_
    ).tobytes()
    self._hbm_buffers_config = np.asanyarray(
        embedding_manager.hbm_buffers_config()[0], dtype=np.bytes_
    ).tobytes()

  def initialize_tpu_embedding(
      self,
      start_remote_python: bool = False,
      embedding_manager: (
          tpu_embedding_pathways_utils.EmbeddingManager | None
      ) = None,
  ):
    """Initialize tpu embedding.

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
    not explicitly call `initialize_tpu_embedding`. It will be called inside
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

    We do not create deduplication tuple mask when we use remote Python since
    doing that means running an Op on TPU which should go through an XLA
    component in Pathways.

    Args:
      start_remote_python: Whether to start remote Python to initialize the
        embedding configuration from this function. This can be False even if
        the backend is Pathways since we have the case where this is done inside
        `enqueue` (Example 2) which is already placed in a remote Python. But if
        it is True, the backend has to be Pathways.
      embedding_manager: The object to manage the remote Python execution. This
        is only applicable if `start_remote_python` is True.

    Raises:
      RuntimeError: If tpu embedding is already initialized on TPU.
    """
    self._config_proto = config_utils.create_config_proto(
        self._tpu_embedding_config,
        self._use_pathways and not start_remote_python,
    )
    self._config_proto_ready = True
    if start_remote_python:
      assert embedding_manager is not None
      self._remote_initialize_tpu_embedding(
          self._config_proto, embedding_manager
      )
      self._is_initialized = True
      logging.info("Successfully Initialized TPUEmbedding devices.")
    else:
      if tpu_ops.is_tpu_embedding_initialized():
        raise RuntimeError(
            "TPU is already initialized for embeddings. This may be caused by "
            "using multiple TPUEmbedding instances in a TPU scope which is "
            "unsupported."
        )

      new_config_str = tpu_embedding_utils.get_new_config(self._config_proto)
      if self._use_pathways:
        assert self._coordinator_address is not None
        coordination_service, coordination_client = (
            coordination_service_utils.init_coordination_service(
                self._host_id, self._num_hosts, self._coordinator_address
            )
        )
        result = coordination_service_utils.initialize_fn(
            new_config_str,
            self._host_id,
            self._num_hosts,
            coordination_client,
            True,
        )
        assert result is not None
        self._embedding_partitions, self._hbm_buffers_config = (
            result[0].numpy(),
            result[1].numpy(),
        )
        self._embedding_partitions_ready = True
        self._hbm_buffers_config_ready = True
        coordination_client.shutdown()
        if coordination_service is not None:
          coordination_service.shutdown()
      else:
        coordination_service_utils.initialize_fn(
            new_config_str, self._host_id, self._num_hosts,
        )
      self._is_initialized = True
      logging.info("Successfully Initialized TPUEmbedding devices.")
      if not self._use_pathways:
        self._dedup_tuple_mask = tpu_embedding_utils.get_tuple_mask(
            self._config_proto.SerializeToString()
        )
        logging.info(
            "Get deduplication tuple mask : %s", self._dedup_tuple_mask
        )

  def create_dedup_tuple_mask(self):
    if self._dedup_tuple_mask:
      return
    self._dedup_tuple_mask = tpu_embedding_utils.get_tuple_mask_pmap(
        self._config_proto.SerializeToString(),
        self._embedding_partitions,
        self._hbm_buffers_config,
        self._tpu_topology,
    )
    logging.info("Get deduplication tuple mask : %s", self._dedup_tuple_mask)

  def init_embedding_tables(
      self,
      embedding_manager: tpu_embedding_pathways_utils.EmbeddingManager,
      retrieve_specs: NestedStruct[jax.ShapeDtypeStruct],
      embedding_variables: Optional[NestedStruct[GlobalHostArray]] = None,
  ):
    """Initialize embedding tables.

    This is only meaningful when using Pathways.

    Args:
      embedding_manager: The object to manage the remote Python execution.
      retrieve_specs: The specs for embedding tables.
      embedding_variables: If not None, initialize variables of embeddings in
        GlobalHostArray; if None, initialize embedding variables created by
        initializers.
    """
    embedding_manager.init_embedding_tables(
        table_config_list=self._table_config_list,
        config_proto_str=self._config_proto.SerializeToString(),
        num_shards=self._num_hosts,
        retrieve_specs=retrieve_specs,
        embedding_variables=embedding_variables,
    )

  def load_embedding_tables(
      self,
      embedding_variables: Optional[NestedStruct[GlobalHostArray]] = None,
      start_remote_python: bool = False,
      embedding_manager: (
          tpu_embedding_pathways_utils.EmbeddingManager | None
      ) = None,
  ) -> None:
    """Load tables' variables on the embedding engine to given set of tensors.

    When using remote Python to load embedding tables, the remote Python must
    have been initialized before running this function.

    Args:
      embedding_variables: If not None, load variables of embeddings in
        GlobalHostArray; if None, load embedding variables created by
        initializers.
      start_remote_python: See the comment in `initialize_tpu_embedding`.
      embedding_manager: The object to manage the remote Python execution. This
        is only applicable if `start_remote_python` is True.
    """
    if start_remote_python:
      assert embedding_manager is not None
      assert embedding_manager.embedding_tables_initialized()
      embedding_manager.load_embedding_tables()
    else:
      tpu_embedding_utils.load_embedding_tables_impl(
          table_config_list=self._table_config_list,
          config_proto_str=self._config_proto.SerializeToString(),
          host_id=self._host_id,
          num_hosts=self._num_hosts,
          embedding_variables=embedding_variables,
      )

  def retrieve_embedding_tables(
      self,
      as_gha=False,
      start_remote_python: bool = False,
      embedding_manager: (
          tpu_embedding_pathways_utils.EmbeddingManager | None
      ) = None,
  ) -> Union[
      NestedStruct[np.ndarray],
      NestedStruct[TensorType],
      NestedStruct[GlobalHostArray],
  ]:
    """Retrieve tables variables from embedding engine.

    Args:
      as_gha: If True, converts returned tables and slots to GlobalHostArray to
        save in Orbax checkpoints.
      start_remote_python: See the comment in `initialize_tpu_embedding`.
      embedding_manager: The object to manage the remote Python execution. This
        is only applicable if `start_remote_python` is True.

    Returns:
      A dict of dict of list of tensors. The outer dict is indexed by table's
      name. The inner dict is indexed by slot names and the final list is the
      list of per host tensors.
    """
    if start_remote_python:
      assert embedding_manager is not None
      assert embedding_manager.embedding_tables_initialized()
      retrieved_tables = embedding_manager.retrieve_embedding_tables()
    else:
      retrieved_tables = tpu_embedding_utils.retrieve_embedding_tables_impl(
          table_config_list=self._table_config_list,
          config_proto_str=self._config_proto.SerializeToString(),
          host_id=self._host_id,
          num_hosts=self._num_hosts,
      )

    if not as_gha:
      return retrieved_tables

    def _create_gha(
        data: Union[TensorType, np.ndarray, jax.Array],
        global_shape: Tuple[int, int],
    ):
      if isinstance(data, TensorType):
        return GlobalHostArray(
            data=data.numpy(),
            global_shape=global_shape,
            shard_id=self._host_id,
            num_shards=self._num_hosts,
        )
      elif isinstance(data, np.ndarray):
        return GlobalHostArray(
            data=data,
            global_shape=global_shape,
            shard_id=self._host_id,
            num_shards=self._num_hosts,
        )
      else:
        return GlobalHostArray(
            data=np.asarray(data),
            global_shape=global_shape,
            shard_id=self._host_id,
            num_shards=self._num_hosts,
        )

    retrieved_tables = dict(retrieved_tables)
    for tb_cfg in self._table_config_list:
      if self._use_pathways:
        global_shape = (self._num_hosts, tb_cfg.vocabulary_size, tb_cfg.dim)
      else:
        global_shape = (tb_cfg.vocabulary_size, tb_cfg.dim)
      gha_creator = functools.partial(_create_gha, global_shape=global_shape)
      retrieved_tables[tb_cfg.name] = jax.tree.map(
          gha_creator, retrieved_tables[tb_cfg.name])

    return retrieved_tables

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

      if self._use_pathways:
        op = tpu_ops.xla_send_tpu_embedding_gradients_v2(
            gradients=updated_gradients,
            learning_rates=[
                tf.cast(fn(), dtype=tf.float32)
                for fn in self._dynamic_learning_rates
            ],
            deduplication_data=deduplication_data,
            config=self._config_proto.SerializeToString(),
            embedding_partitions=self._embedding_partitions,
            hbm_buffers_config=self._hbm_buffers_config,
            tpu_topology=self._tpu_topology,
        )
      else:
        op = tpu_ops.xla_send_tpu_embedding_gradients(
            gradients=updated_gradients,
            learning_rates=[
                tf.cast(fn(), dtype=tf.float32)
                for fn in self._dynamic_learning_rates
            ],
            deduplication_data=deduplication_data,
            config=self._config_proto.SerializeToString(),
        )

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
      if self._use_pathways:
        deduplication_data = (
            tpu_ops.xla_recv_tpu_embedding_deduplication_data_v2(
                config=self._config_proto.SerializeToString(),
                embedding_partitions=self._embedding_partitions,
                hbm_buffers_config=self._hbm_buffers_config,
                tpu_topology=self._tpu_topology,
            )
        )
      else:
        deduplication_data = tpu_ops.xla_recv_tpu_embedding_deduplication_data(
            config=self._config_proto.SerializeToString(),
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
      if self._use_pathways:
        activations = tpu_ops.xla_recv_tpu_embedding_activations_v2(
            deduplication_data=deduplication_data,
            num_tables=len(self._config_proto.feature_descriptor),
            config=self._config_proto.SerializeToString(),
            embedding_partitions=self._embedding_partitions,
            hbm_buffers_config=self._hbm_buffers_config,
            tpu_topology=self._tpu_topology,
        )
      else:
        activations = tpu_ops.xla_recv_tpu_embedding_activations(
            deduplication_data=deduplication_data,
            num_tables=len(self._config_proto.feature_descriptor),
            config=self._config_proto.SerializeToString(),
        )

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
      self._output_shapes_ready = True
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
                             flat_weights: List[Optional[TensorType]],
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
    for device_id in range(self._num_local_devices):
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
  def dedup_tuple_mask(self) -> pytype_utils.TensorProto:
    return self._dedup_tuple_mask

  @property
  def output_shapes(self) -> List[config_utils.OutputShape]:
    return self._output_shapes

  @property
  def embedding_partitions(self) -> bytes:
    return self._embedding_partitions

  @property
  def hbm_buffers_config(self) -> bytes:
    return self._hbm_buffers_config

  @property
  def feature_configs(self) -> Optional[NestedFeatureConfig]:
    return self._feature_configs

  @property
  def optimizer(self) -> Optional[TPUEmbeddingOptimizer]:
    return self._optimizer

  @property
  def is_initialized(self) -> bool:
    return self._is_initialized

  @property
  def restore_args(self) -> NestedStruct[pytype_utils.RestoreArgs]:
    return self._embedding_restore_args

  @property
  def shape_and_dtypes(self) -> NestedStruct[jax.ShapeDtypeStruct]:
    return self._embedding_shape_dtypes

  @property
  def table_config_list(self) -> List[config_utils.TableConfig]:
    return self._table_config_list

  @property
  def num_local_devices(self) -> int:
    return self._num_local_devices

  @property
  def tpu_topology(self) -> bytes:
    return self._tpu_topology

  @property
  def config_proto_ready(self) -> bool:
    return self._config_proto_ready

  @property
  def output_shapes_ready(self) -> bool:
    return self._output_shapes_ready

  def embedding_partitions_ready(self) -> bool:
    return self._embedding_partitions_ready

  def hbm_buffers_config_ready(self) -> bool:
    return self._hbm_buffers_config_ready

  def set_config_proto(
      self, config_proto: config_utils.TPUEmbeddingConfigurationProto
  ):
    self._config_proto = config_proto

  def set_dedup_tuple_mask(self, dedup_tuple_mask: pytype_utils.TensorProto):
    self._dedup_tuple_mask = dedup_tuple_mask

  def set_output_shapes(self, output_shapes: List[config_utils.OutputShape]):
    self._tpu_embedding_config.output_shapes = output_shapes
    self._output_shapes = output_shapes

  def set_embedding_partitions(self, embedding_partitions: bytes):
    self._embedding_partitions = embedding_partitions

  def set_hbm_buffers_config(self, hbm_buffers_config: bytes):
    self._hbm_buffers_config = hbm_buffers_config

  def set_packer(self, pack_spec: input_utils.PackSpec):
    self._packer = input_utils.Packer(pack_spec, self._num_local_devices)

  @tf.function
  def experimental_get_next(
      self,
      iterator: tf.data.Iterator,
      num_local_devices: int,
      is_training: bool,
  ) -> NestedTfTensor:
    """Get next batch of data, enqueue the sparse part and pack the dense part.

    This API is experimental and its behavior may change in the future.

    Args:
      iterator: an iterator that yields batched inputs for host and devices,
        first dimension of each element is batch size.
      num_local_devices: Number of local devices.
      is_training: See the comment in `enqueue`.

    Returns:
      The dense part of data with the dense features packed.
    """
    assert self._input_split_fn is not None
    iterator = map(self._input_split_fn, iterator)
    data = next(iterator)
    features = input_utils.shard_inputs(
        data[input_utils.EMBED_PLACEMENT], num_local_devices
    )
    self.enqueue(features, is_training=is_training)
    dense_features = data[input_utils.NON_EMBED_PLACEMENT]
    if self._packer:
      dense_features = self._packer.shard_and_pack_features(
          data[input_utils.NON_EMBED_PLACEMENT]
      )
    return dense_features

  def unpack_single_shard_features(
      self, packed_features: dict[str, jax.Array]
  ) -> Any:
    assert self._packer is not None
    return self._packer.unpack_single_shard_features(packed_features)
