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

"""Jax Embedding API Test Utils."""

from typing import Any, Optional, Sequence, Tuple

import jax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec
from jax_tpu_embedding import pytype_utils
import numpy as np
import tensorflow as tf

AxisResourcesRegexes = Sequence[Tuple[str, PartitionSpec]]
PyTree = Any

FeatureConfig = pytype_utils.FeatureConfig
TableConfig = pytype_utils.TableConfig
Nested = pytype_utils.Nested

PER_CORE_BATCH_SIZE = 2


def dummy_features() -> Tuple[Nested[tf.SparseTensor], Nested[tf.Tensor]]:
  """Create dummy test features made by sparse tensors.

  Returns:
    A tuple of sparse tensors.
  """
  feature_watched_indices = [[0, 0], [1, 0], [1, 1], [2, 0], [2, 1], [3, 0]]
  feature_watched_values = [0, 0, 1, 0, 1, 1]

  feature_favorited_indices = [[0, 0], [0, 1], [1, 0], [2, 0], [3, 0], [3, 1]]
  feature_favorited_values = [0, 1, 1, 0, 0, 1]

  feature_friends_indices = [[0, 0], [1, 0], [1, 1], [1, 2], [2, 0], [3, 0],
                             [3, 1], [3, 2]]
  feature_friends_values = [3, 0, 1, 2, 3, 0, 1, 2]

  data_batch_size = 4

  dense_features = {'dense': tf.convert_to_tensor(
      [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]], dtype=tf.float32)}

  sparse_features = (tf.SparseTensor(
      indices=feature_watched_indices,
      values=feature_watched_values,
      dense_shape=[data_batch_size, 10]),
                     tf.SparseTensor(
                         indices=feature_favorited_indices,
                         values=feature_favorited_values,
                         dense_shape=[data_batch_size, 2]),
                     tf.SparseTensor(
                         indices=feature_friends_indices,
                         values=feature_friends_values,
                         dense_shape=[data_batch_size, 3]))
  return (sparse_features, dense_features)


def create_dummy_dataset(batch_size: int) -> tf.data.Dataset:
  """Create a dataset of sparse features."""
  sparse_features = dummy_features()
  dataset = tf.data.Dataset.from_tensors(sparse_features)
  dataset = dataset.unbatch().repeat().batch(batch_size, drop_remainder=True)
  return dataset


def create_feature_config(
    batch_size: int,
    use_shape_inference: bool = False,
    init_values: Optional[np.ndarray] = None) -> tuple[FeatureConfig, ...]:
  """Create A tuple of feature config.

  Args:
    batch_size: local batch size of input, which in FeatureConfig object, will
      be used for output shape inference.
    init_values: A list of table embedding initialization value.

  Returns:
    A tuple of FeatureConfig.
  """
  if init_values is None:
    init_values = np.array(list(range(32)), dtype=np.float64)

  initializer = tf.constant_initializer(init_values)
  table_video = TableConfig(
      vocabulary_size=8,
      dim=4,
      initializer=initializer,
      combiner='sum',
      name='video',
  )
  table_user = TableConfig(
      vocabulary_size=16,
      dim=2,
      initializer=initializer,
      combiner='mean',
      name='user',
  )
  output_shape = None if use_shape_inference else [batch_size]
  return (
      FeatureConfig(
          table=table_video, name='watched', output_shape=output_shape
      ),
      FeatureConfig(
          table=table_video, name='favorited', output_shape=output_shape
      ),
      FeatureConfig(
          table=table_user, name='friends', output_shape=output_shape
      ),
  )


def create_global_mesh(
    mesh_shape: tuple[int, ...],
    axis_names: Sequence[jax.interpreters.pxla.MeshAxisName],
) -> Mesh:
  size = np.prod(mesh_shape)
  if len(jax.devices()) < size:
    raise ValueError(f'Test requires {size} global devices.')
  # Sorts devices by host_id and core on chip for contiguous devices.
  devices = sorted(jax.devices(), key=lambda d: (d.host_id, d.core_on_chip))
  mesh_devices = np.array(devices[:size]).reshape(mesh_shape)
  global_mesh = Mesh(mesh_devices, axis_names)
  return global_mesh
