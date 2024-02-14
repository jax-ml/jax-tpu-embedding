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

"""Tests for input_utils."""

from absl.testing import parameterized
import jax
from jax_tpu_embedding import input_utils
from jax_tpu_embedding.tests import test_utils
import tensorflow as tf

GLOBAL_BATCH_SIZE = 4
VOCAB_SIZE = 10


def dummy_inputs():
  """Create dummy test features."""
  feature_watched_indices = [[0, 0], [1, 0], [1, 1], [2, 0], [2, 1], [3, 0]]
  feature_watched_values = [0, 0, 1, 0, 1, 1]

  watched = tf.SparseTensor(
      indices=feature_watched_indices,
      values=feature_watched_values,
      dense_shape=[GLOBAL_BATCH_SIZE, VOCAB_SIZE])
  targets = (tf.constant([0, 1, 2, 3], dtype=tf.int32),)
  return watched, targets


def dummy_dataset() -> tf.data.Dataset:
  """Create a dummy dataset for test."""

  features = dummy_inputs()
  dataset = tf.data.Dataset.from_tensors(features)
  dataset = dataset.unbatch().repeat()
  return dataset


class InputUtilsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    feature_configs = test_utils.create_feature_config(
        batch_size=test_utils.PER_CORE_BATCH_SIZE)
    self._flatten_feature_configs, _ = input_utils.tree_flatten_with_names(
        feature_configs)



  @parameterized.named_parameters(
      ('_unnested', tf.ones(shape=[
          8,
      ])),
      ('_nested', [
          tf.ones(shape=[
              8,
          ]),
          tf.SparseTensor(
              indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[4, 4]),
          tf.ones(shape=[4, 3, 2])
      ]),
  )
  def test_valid_inputs_to_shard(self, inputs):
    """Test valid sharding inputs."""
    num_shards = 2
    sharded_inputs = input_utils.shard_inputs(inputs, num_shards=num_shards)

    self.assertLen(sharded_inputs, num_shards)
    flatten_inputs, input_tree = jax.tree_util.tree_flatten(inputs)
    for per_shard in sharded_inputs:
      flatten_shard, shard_tree = jax.tree_util.tree_flatten(per_shard)
      self.assertEqual(shard_tree, input_tree)
      for shard_tensor, input_tensor in zip(flatten_shard, flatten_inputs):
        shard_shape = shard_tensor.shape.as_list()
        global_shape = input_tensor.shape.as_list()
        self.assertListEqual(shard_shape,
                             [global_shape[0] // num_shards] + global_shape[1:])

  @parameterized.named_parameters(
      ('_unnested', tf.ones(shape=[
          7,
      ])),
      ('_nested', [
          tf.ones(shape=[
              7,
          ]),
          tf.SparseTensor(
              indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[7, 4]),
          tf.ones(shape=[4, 3, 2])
      ]),
  )
  def test_invalid_inputs_to_shard(self, inputs):
    """Test invalid sharding inputs."""
    num_shards = 2

    with self.assertRaisesRegex(
        ValueError,
        'Number of shards should evenly divide the first dimension'):
      input_utils.shard_inputs(inputs, num_shards)

  def test_tree_flatten_with_names(self):
    """Test python tree flatten with name of path."""
    tree = {'k1': 1, 'k2': 2, 'k34': (3, 4)}
    names_and_vals, _ = input_utils.tree_flatten_with_names(tree)
    expected_flatten = [('k1', 1), ('k2', 2), ('k34/0', 3), ('k34/1', 4)]
    self.assertListEqual(names_and_vals, expected_flatten)

    tree = {'ff_0': {'kernel': 0, 'bias': 1}, 'ff_1': {'kernel': 2, 'bias': 3}}
    names_and_vals, _ = input_utils.tree_flatten_with_names(tree)
    expected_flatten = [('ff_0/bias', 1), ('ff_0/kernel', 0), ('ff_1/bias', 3),
                        ('ff_1/kernel', 2)]
    self.assertListEqual(names_and_vals, expected_flatten)

    tree = (1, (3, 2))
    names_and_vals, _ = input_utils.tree_flatten_with_names(tree)
    expected_flatten = [('0', 1), ('1/0', 3), ('1/1', 2)]
    self.assertListEqual(names_and_vals, expected_flatten)

  @parameterized.product(
      (
          {
              # Unweighted input.
              'weight': None,
              'expected_weight': tf.zeros((0,), dtype=tf.float32)
          }, {
              # Weighted input.
              'weight':
                  tf.SparseTensor(
                      indices=[[0, 0], [1, 2]],
                      values=[0.5, 0.5],
                      dense_shape=[4, 4]),
              'expected_weight':
                  tf.constant([0.5, 0.5], dtype=tf.float32)
          },
      ),
      (
          {
              # Feature configuration has max_sequence_length equals 0.
              'max_sequence_length': 0,
              'expected_indices': tf.constant([[0, 0], [1, 2]], dtype=tf.int32)
          },
          {
              # Feature configuration has max_sequence_length greater than 0.
              'max_sequence_length': 2,
              'expected_indices':
                  tf.constant([[0, 0, 0], [1, 2, 0]], dtype=tf.int32)
          }))
  def test_process_sparse_data(self, weight, expected_weight,
                               max_sequence_length,
                               expected_indices):
    """Test input_utils._process_sparse_data method."""
    sparse_input = tf.SparseTensor(
        indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[4, 4])
    feature_config = test_utils.FeatureConfig(
        table=test_utils.TableConfig(vocabulary_size=8, dim=4),
        max_sequence_length=max_sequence_length)
    enqueue_input = input_utils._process_sparse_data(
        sparse_input=sparse_input,
        weight=weight,
        feature_config=feature_config,
        path_name='0')
    self.assertAllEqual(enqueue_input.indices, expected_indices)
    self.assertAllEqual(enqueue_input.values, tf.constant([1, 2],
                                                          dtype=tf.int64))
    self.assertAllEqual(enqueue_input.aggregation_weight, expected_weight)

  def test_invalid_sparse_input_weight(self):
    """Test sparse input has unexpected dense weight."""
    sparse_input = tf.SparseTensor(
        indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[4, 4])
    weight = tf.ones(shape=[4, 4])
    path_name, feature_config = self._flatten_feature_configs[0]

    with self.assertRaisesRegex(
        ValueError, 'Weight for 0 is not NoneType or tf.SparseTensor'):
      _ = input_utils._process_sparse_data(
          sparse_input=sparse_input,
          weight=weight,
          feature_config=feature_config,
          path_name=path_name)

  @parameterized.named_parameters(
      {
          'testcase_name': '_none_weights',
          'flatten_weights': [None] * 3,
          'expected_weights': [tf.zeros((0,), dtype=tf.float32)] * 3
      }, {
          'testcase_name':
              '_spares_weights',
          'flatten_weights': [
              tf.SparseTensor(
                  indices=[[0, 0], [1, 2]],
                  values=[0.5, 0.5],
                  dense_shape=[4, 4]),
              None,
              tf.SparseTensor(
                  indices=[[0, 0], [1, 2]],
                  values=[0.1, 0.1],
                  dense_shape=[4, 4]),
          ],
          'expected_weights': [
              tf.constant([0.5, 0.5], dtype=tf.float32),
              tf.zeros((0,), dtype=tf.float32),
              tf.constant([0.1, 0.1], dtype=tf.float32)
          ]
      })
  def test_valid_enqueue_inputs(self, flatten_weights, expected_weights):
    """Test valid data composed to enqueue."""
    sparse_input = tf.SparseTensor(
        indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[4, 4])
    dense_input = tf.ones(shape=[4, 3])
    feature_inputs = (sparse_input, (dense_input, sparse_input))
    flatten_inputs, _ = jax.tree_util.tree_flatten(feature_inputs)

    expected_sparse_indices = tf.constant([[0, 0], [1, 2]], dtype=tf.int32)
    expected_indices = [
        expected_sparse_indices,
        tf.zeros((0,), dtype=tf.int32), expected_sparse_indices
    ]
    expected_sparse_values = tf.constant([1, 2], dtype=tf.int64)
    expected_values = [
        expected_sparse_values,
        tf.ones(shape=(12,), dtype=tf.int64), expected_sparse_values
    ]

    expected_inputs = zip(expected_indices, expected_values, expected_weights)

    # Test all none weights.
    enqueue_inputs = input_utils.prepare_data_to_enqueue(
        flatten_inputs=flatten_inputs,
        flatten_weights=flatten_weights,
        flatten_feature_configs=self._flatten_feature_configs)

    for inputs, expected_inputs in zip(enqueue_inputs, expected_inputs):

      expected_ind, expected_val, expected_weight = expected_inputs

      self.assertAllEqual(inputs.indices, expected_ind)
      self.assertAllEqual(inputs.values, expected_val)
      self.assertAllEqual(inputs.aggregation_weight, expected_weight)

  @parameterized.named_parameters(
      {
          'testcase_name': 'weighted_dense',
          'flatten_inputs': [
              tf.ones(shape=[4, 3]),
              tf.SparseTensor(
                  indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[4, 4]),
              tf.SparseTensor(
                  indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[4, 4]),
          ],
          'flatten_weights': [
              tf.ones(shape=[4, 3]) * 0.1,
              tf.SparseTensor(
                  indices=[[0, 0], [1, 2]],
                  values=[0.5, 0.5],
                  dense_shape=[4, 4]),
              tf.SparseTensor(
                  indices=[[0, 0], [1, 2]],
                  values=[0.5, 0.5],
                  dense_shape=[4, 4]),
          ],
          'expected_error_regex':
              'Dense input should not have specified weight',
      }, {
          'testcase_name': 'unmatched_length',
          'flatten_inputs': [tf.ones(shape=[4, 3])] * 3,
          'flatten_weights': [None, None],
          'expected_error_regex': 'All flatten data should have same length',
      }, {
          'testcase_name': 'ragged_tensor',
          'flatten_inputs': [
              tf.ragged.constant([[1, 1], [], [2], []]),
              tf.ones(shape=[4, 3]),
              tf.ones(shape=[4, 3]),
          ],
          'flatten_weights': [None, None, None],
          'expected_error_regex':
              'Input tensor is neither tf.Tensor nor tf.SparseTensor.'
      })
  def test_invalid_enqueue_inputs(self, flatten_inputs, flatten_weights,
                                  expected_error_regex):
    """Test invalid input to enqueue."""
    with self.assertRaisesRegex(ValueError, expected_error_regex):
      enqueue_inputs_iter = input_utils.prepare_data_to_enqueue(
          flatten_inputs=flatten_inputs,
          flatten_weights=flatten_weights,
          flatten_feature_configs=self._flatten_feature_configs)
      _ = next(enqueue_inputs_iter)


if __name__ == '__main__':
  tf.test.main()
