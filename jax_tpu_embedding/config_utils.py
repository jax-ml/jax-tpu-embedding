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

"""TPU Embedding configuration utils support."""

import collections
import dataclasses
import math
from typing import Callable, Dict, List, Optional, Union

import jax
from jax_tpu_embedding import pytype_utils
import numpy as np
import tensorflow as tf

LearningRate = Union[float, Callable[[], float]]
Optimizer = pytype_utils.TPUEmbeddingOptimizer
AdagradMomentumOptimizer = pytype_utils.TPUEmbeddingOptimizerAdagradMomentum
AdagradOptimizer = pytype_utils.TPUEmbeddingOptimizerAdagrad
AdamOptimizer = pytype_utils.TPUEmbeddingOptimizerAdam
FTRLOptimizer = pytype_utils.TPUEmbeddingOptimizerFTRL
SGDOptimizer = pytype_utils.TPUEmbeddingOptimizerSGD

TableConfig = pytype_utils.TableConfig
TensorProto = pytype_utils.TensorProto
FeatureConfig = pytype_utils.FeatureConfig
NestedFeatureConfig = pytype_utils.NestedFeatureConfig
TPUEmbeddingConfigurationProto = pytype_utils.TPUEmbeddingConfigurationProto

OutputShape = Union[List[int], tf.TensorShape]


@dataclasses.dataclass
class TpuEmbeddingConfigSpecs:
  """Configurations and specs to create tpu embedding configuration proto.

  Attributes:
    feature_config: A nested structure of feature config.
    output_shapes: A list of output shapes.
    table_config_list: A list of table configs.
    dynamic_learning_rates: A list of learning rates or a callable mapping to
      learning rate tags.
    pipeline_execution_with_tensor_core: If the TPU embedding computations will
      overlap with the TensorCore computations.
    num_hosts: number of hosts.
    num_tensor_cores: number of tensor cores.
    cores_per_replica: number of cores per replica, use for spmd when it's not
      None.
    use_manual_partitioning: If True, the tensors are manually partitioned.
      Otherwise, use the automatic SPMD partitioning.
  """
  feature_config: NestedFeatureConfig
  output_shapes: List[OutputShape]
  table_config_list: List[TableConfig]
  dynamic_learning_rates: Optional[List[LearningRate]]
  pipeline_execution_with_tensor_core: bool
  num_hosts: int
  num_tensor_cores: int
  cores_per_replica: Optional[int]
  use_manual_partitioning: bool


def create_tpu_embedding_config(
    feature_configs: NestedFeatureConfig,
    optimizer: Optional[Optimizer],
    pipeline_execution_with_tensor_core: bool,
    num_hosts: int,
    num_tensor_cores: int,
    cores_per_replica: Optional[int] = None,
    use_manual_partitioning: bool = False,
) -> TpuEmbeddingConfigSpecs:
  """Creates TpuEmbeddingConfigSpecs.

  Args:
    feature_configs: A nested structure of feature configs.
    optimizer: An instance of tpu embedding optimizer or None for inference.
    pipeline_execution_with_tensor_core: If True, the TPU embedding computations
      will overlap with the TensorCore computations (and hence will be one step
      old). Set to True for improved performance.
    num_hosts: number of hosts.
    num_tensor_cores: number of tensor cores.
    cores_per_replica: number of cores for one replica. If None, config would be
      for data parallelism only, if not None config will be set for SPMD.
    use_manual_partitioning: If True, the tensors are manually partitioned.
      Otherwise, use the automatic SPMD partitioning.

  Raises:
      ValueError: If optimizer is not one of tf.tpu.experimental.embedding.(SGD,
      Adam or Adagrad) or None.

  Returns:
    An instance of TpuEmbeddingConfig.
  """

  output_shapes = []
  flatten_feature_configs, _ = jax.tree_util.tree_flatten(feature_configs)
  for feature in flatten_feature_configs:
    output_shapes.append(feature.output_shape)

  table_config_list = []
  for feature in flatten_feature_configs:
    if feature.table not in table_config_list:
      table_config_list.append(feature.table)

  # Ensure tables have unique names. Also error check the optimizer as we
  # specifically don't do that in the TableConfig class to allow high level
  # APIs that are built on this to use strings/other classes to represent
  # optimizers (before they are passed to this class).
  table_names = []
  for i, table in enumerate(table_config_list):
    if table.optimizer is None:
      table.optimizer = optimizer
    if ((table.optimizer is not None) and
        not isinstance(table.optimizer, Optimizer)):
      raise ValueError("{} is an unsupported optimizer class. Please pass an "
                       "instance of one of the optimizer classes under "
                       "tf.tpu.experimental.embedding.".format(
                           type(table.optimizer)))
    if table.name is None:
      table.name = "table_{}".format(i)
    if table.name in table_names:
      raise ValueError("Tables must have a unique name. "
                       f"Multiple tables with name {table.name} found.")
    table_names.append(table.name)

  dynamic_learning_rates = list({
      table.optimizer.learning_rate
      for table in table_config_list
      if callable(table.optimizer.learning_rate)
  })

  return TpuEmbeddingConfigSpecs(
      feature_config=feature_configs,
      output_shapes=output_shapes,
      table_config_list=table_config_list,
      dynamic_learning_rates=dynamic_learning_rates,
      pipeline_execution_with_tensor_core=pipeline_execution_with_tensor_core,
      num_hosts=num_hosts,
      num_tensor_cores=num_tensor_cores,
      cores_per_replica=cores_per_replica,
      use_manual_partitioning=use_manual_partitioning,
  )


def create_tpu_embedding_configs(
    feature_configs: List[NestedFeatureConfig],
    optimizer: Optional[Optimizer],
    pipeline_execution_with_tensor_core: bool,
    num_hosts: int,
    num_tensor_cores: int,
    cores_per_replica: Optional[int] = None,
    use_manual_partitioning: bool = False,
) -> List[TpuEmbeddingConfigSpecs]:
  """Creates a list of TpuEmbeddingConfigSpecs.

  Args:
    feature_configs: A list of nested structure of feature configs.
    optimizer: An instance of tpu embedding optimizer or None for inference.
    pipeline_execution_with_tensor_core: If True, the TPU embedding computations
      will overlap with the TensorCore computations (and hence will be one step
      old). Set to True for improved performance.
    num_hosts: number of hosts.
    num_tensor_cores: number of tensor cores.
    cores_per_replica: number of cores for one replica. If None, config would be
      for data parallelism only, if not None config will be set for SPMD.
    use_manual_partitioning: If True, the tensors are manually partitioned.
      Otherwise, use the automatic SPMD partitioning.

  Raises:
      ValueError: If optimizer is not one of tf.tpu.experimental.embedding.(SGD,
      Adam or Adagrad) or None.

  Returns:
    A list of TpuEmbeddingConfig.
  """

  tpu_embedding_configs = []
  for feature_config in feature_configs:
    tpu_embedding_config = create_tpu_embedding_config(
        feature_config,
        optimizer,
        pipeline_execution_with_tensor_core,
        num_hosts,
        num_tensor_cores,
        cores_per_replica,
        use_manual_partitioning,
    )
    tpu_embedding_configs.append(tpu_embedding_config)
  return tpu_embedding_configs


def compute_batch_size_per_core(
    config_proto: TPUEmbeddingConfigurationProto) -> int:
  """Computes tensor core batch size as the gcd of all input feature batch size.

  Each feature descriptor input_shape field is `TpuEmbeddingConfig.OutputShapes`
  without last dimension, i.e., for output_shape = [x, y, embedding_dim],
  corresponding input_shape added input config_proto is [x, y].
  For embedding engine, it always reshape input_shape to 1D as [x * y,]
  for low level enqueue ops.
  It needs input batch_size that can divide all input features on host. Hence,
  this method is to compute gcd number for all input features batch_size. This
  gcd number will be used to set `batch_size_per_core`. The referenced logic is
  method `ComputeBatchSizePerTensorCore` in
  `third_party/tensorflow/core/tpu/tpu_embedding_configuration_proto_rewrite.cc`
  .

  Args:
    config_proto: A given tpu embedding config proto file includes all input
      features.

  Returns:
    gcd value as batch_size_per_core for tpu embedding configuration.
  """
  num_all_features = len(config_proto.feature_descriptor)
  if num_all_features == 0:
    raise ValueError("config_proto's feature_descripor list is empty.")

  batch_size_per_core = 0
  for feature_id in range(num_all_features):
    # Get input_shape for each feature and get 1D reshaped input batch_size.
    input_shape = config_proto.feature_descriptor[feature_id].input_shape
    batch_size_per_core = math.gcd(batch_size_per_core, np.prod(input_shape))

  return batch_size_per_core


def count_table_feature_numbers(
    config_proto: TPUEmbeddingConfigurationProto) -> Dict[int, int]:
  """Count number of features each table is used by.

  This is a field need to be added into configuration proto before tpu embedding
  initialization.

  Args:
    config_proto: A tpu embedding configuration proto file which should be
      created from user provided TpuEmbeddingConfig instance.

  Returns:
    A counter dictionary gives {table_id : num_features}
  """
  table_num_features = collections.defaultdict(int)
  for feature_id in range(len(config_proto.feature_descriptor)):
    table_id = config_proto.feature_descriptor[feature_id].table_id
    table_num_features[table_id] += 1
  return table_num_features


def create_config_proto(
    tpu_embedding_config: TpuEmbeddingConfigSpecs,
    recreate_table_config_list: bool = False,
) -> TPUEmbeddingConfigurationProto:
  """Creates TPUEmbeddingConfiguration proto initialize TPU embedding engine.

  Args:
    tpu_embedding_config: A TpuEmbeddingConfig instance. It includes info to be
      used to create configuration proto.
    recreate_table_config_list: Whether to recreate table_config_list. This is
      for the case where the contents in table_config_list have been changed.
      For example, when the object holding table_config_list has gone through a
      deep copy.

  Raises:
    ValueError: When table_config_list or output_shapes is empty.
  Returns:
    A configuration proto.
  """

  config_proto = TPUEmbeddingConfigurationProto()

  # Map each callable dynamic learning rate to its in index in the list.
  # The learning rate index is the index of the dynamic learning rate for this
  # table (if it exists) in the list we created at initialization. We don't
  # simply create one learning rate index per table as this has extremely bad
  # performance characteristics. The more separate optimization configurations
  # we have, the worse the performance will be.
  learning_rate_index = {
      learning_rate: i for i, learning_rate in enumerate(
          tpu_embedding_config.dynamic_learning_rates)
  }

  if not tpu_embedding_config.table_config_list:
    raise ValueError("tpu_embedding_config.table_config_list is empty.")

  for table in tpu_embedding_config.table_config_list:
    # For small tables, we pad to the number of hosts so that at least one
    # id will be assigned to each host.
    vocab_size = max(table.vocabulary_size, tpu_embedding_config.num_hosts)
    table_descriptor = TPUEmbeddingConfigurationProto.TableDescriptor(
        name=table.name, vocabulary_size=vocab_size, dimension=table.dim)

    parameters = table_descriptor.optimization_parameters

    # We handle the learning rate separately here and don't allow the
    # optimization class to handle this, as it doesn't know about dynamic
    # rates.
    if callable(table.optimizer.learning_rate):
      parameters.learning_rate.dynamic.tag = (
          learning_rate_index[table.optimizer.learning_rate])
    else:
      parameters.learning_rate.constant = table.optimizer.learning_rate

    # Use optimizer to handle the rest of the parameters.
    table.optimizer._set_optimization_parameters(parameters)  # pylint: disable=protected-access
    config_proto.table_descriptor.append(table_descriptor)

  if recreate_table_config_list:
    flatten_feature_config, _ = jax.tree_util.tree_flatten(
        tpu_embedding_config.feature_config
    )
    tpu_embedding_config.table_config_list = []
    for feature in flatten_feature_config:
      if feature.table not in tpu_embedding_config.table_config_list:
        tpu_embedding_config.table_config_list.append(feature.table)
  table_to_id = {
      table: i for i, table in enumerate(tpu_embedding_config.table_config_list)
  }

  # Set feature descriptor field in the config proto.
  if not tpu_embedding_config.output_shapes:
    raise ValueError("tpu_embedding_config.output_shapes is empty.")

  for feature, output_shape in zip(
      tf.nest.flatten(tpu_embedding_config.feature_config),
      tpu_embedding_config.output_shapes):
    feature_descriptor = config_proto.feature_descriptor.add()

    if feature.name:
      feature_descriptor.name = feature.name

    feature_descriptor.table_id = table_to_id[feature.table]
    # The input shape of the feature is the actual shape of the input tensor
    # except the last dimension because the last dimension will always be
    # reduced.
    feature_descriptor.input_shape.extend(output_shape.as_list())

  # Always set mode to training, we override the mode during enqueue.
  config_proto.mode = TPUEmbeddingConfigurationProto.TRAINING

  config_proto.num_hosts = tpu_embedding_config.num_hosts
  config_proto.num_tensor_cores = tpu_embedding_config.num_tensor_cores

  config_proto.sharding_strategy = TPUEmbeddingConfigurationProto.DIV_DEFAULT
  config_proto.pipeline_execution_with_tensor_core = (
      tpu_embedding_config.pipeline_execution_with_tensor_core)

  # When cores_per_replica is not None, set for SPMD.
  if tpu_embedding_config.cores_per_replica:
    config_proto.spmd_sharding.enabled = True
    config_proto.spmd_sharding.num_cores_per_replica = (
        tpu_embedding_config.cores_per_replica)
    config_proto.spmd_sharding.use_manual_partitioning = (
        tpu_embedding_config.use_manual_partitioning
    )

  return config_proto


def set_additional_fields(config_proto: TPUEmbeddingConfigurationProto):
  """Set additional fields that are needed for tpu embedding initialization.

  For tpu embedding initialization, we need to aet additional fields of config
  proto. The fields to set are `batch_size_per_tensor_core` and
  `num_features` under each table descriptor.

  These fields are only needed for initialization, must not exist when
  `config_proto` is used for load/retrieve embedding parameters and slot
  variables.

  Args:
    config_proto: tpu embedding configuration proto which should not set values
      of `batch_size_per_tensor_core` and `num_features` for each
      table_descriptor.

  Raises:
    ValueError: When any of fields to set has non-zero value.
  """

  if config_proto.batch_size_per_tensor_core != 0:
    raise ValueError("config_proto already has `batch_size_per_tensor_core`.")

  config_proto.batch_size_per_tensor_core = compute_batch_size_per_core(
      config_proto)
  table_num_features = count_table_feature_numbers(config_proto)
  for table_id in range(len(config_proto.table_descriptor)):
    table_descriptor = config_proto.table_descriptor[table_id]
    if table_descriptor.num_features != 0:
      raise ValueError(f"Table {table_id} already has `num_features`.")
    table_descriptor.num_features = table_num_features[table_id]


def create_mask_proto(tuple_mask: np.ndarray) -> TensorProto:
  """Create TensorProto for Tuple mask.

  When using software deduplication, the output of deduplication is XLA tuple,
  including integer elements and floating point elements. `tuple_mask` consists
  of 0 or 1. 1 means floating point type, 0 means integer type.

  Args:
    tuple_mask: A np.ndarray represents tuple element is floating point type.

  Returns:
    A TensorProto of `tuple_mask`.
  """
  mask_shape = tf.TensorShape(tuple_mask.shape)
  mask_tensor_proto = TensorProto(
      dtype=tf.int32.as_datatype_enum, tensor_shape=mask_shape.as_proto()
  )
  mask_tensor_proto.int_val.extend(tuple_mask.ravel())
  return mask_tensor_proto
