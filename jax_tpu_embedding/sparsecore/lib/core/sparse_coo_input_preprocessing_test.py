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
import dataclasses
import functools

from absl.testing import absltest
from absl.testing import parameterized
from jax_tpu_embedding.sparsecore.lib.core import pybind_input_preprocessing
from jax_tpu_embedding.sparsecore.lib.core import test_utils
from jax_tpu_embedding.sparsecore.lib.core.pybind_input_preprocessing import ShardingStrategy
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
import numpy as np


class SparseCooInput:
  """Mimics tf.sparse.SparseTensor (or any other COO format)."""

  def __init__(self, indices, values, dense_shape):
    self.indices = indices
    self.values = values
    self.dense_shape = dense_shape


class SparseTensorInputPreprocessingTest(parameterized.TestCase):
  stacked_name = "one_table_to_rule_them_all"
  table_spec = embedding_spec.TableSpec(
      vocabulary_size=512,
      embedding_dim=8,
      initializer=lambda: np.zeros((512, 8), dtype=np.float32),
      optimizer=embedding_spec.SGDOptimizerSpec(
          learning_rate=0.001,
      ),
      combiner="sum",
      name="table_b",
      max_ids_per_partition=64,
      max_unique_ids_per_partition=64,
      _setting_in_stack=embedding_spec.TableSettingInStack(
          stack_name=stacked_name,
          padded_vocab_size=512,
          padded_embedding_dim=8,
          row_offset_in_shard=0,
          shard_rotation=0,
      ),
      _stacked_table_spec=embedding_spec.StackedTableSpec(
          stack_name=stacked_name,
          stack_vocab_size=512,
          stack_embedding_dim=8,
          optimizer=embedding_spec.SGDOptimizerSpec(
              learning_rate=0.001,
          ),
          combiner="sum",
          total_sample_count=16,
          max_ids_per_partition=64,
          max_unique_ids_per_partition=64,
      ),
  )
  feature_spec = embedding_spec.FeatureSpec(
      table_spec=table_spec,
      input_shape=[16, 16],
      output_shape=[
          16,
          table_spec.embedding_dim,
      ],
      name="feature_spec_b",
      _id_transformation=embedding_spec.FeatureIdTransformation(
          row_offset=0,
          col_offset=0,
          col_shift=0,
      ),
  )

  @parameterized.parameters(False, True)
  def test_sparse_tensor_input(self, has_leading_dimension):
    indices = []
    values = []
    for i in range(16):
      for j in range(16):
        indices.append((i, j))
    for i in range(16 * 16):
      values.append(i)

    sparse_tensor = SparseCooInput(
        indices=indices,
        values=values,
        dense_shape=[16, 512],
    )
    sparse_tensor_input_preprocessing = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulSparseCooInput
    )
    indices_tensor = [sparse_tensor.indices]
    values_tensor = [sparse_tensor.values]
    dense_shape_tensor = [sparse_tensor.dense_shape]
    batch_number = 42
    (
        row_pointers_sparse,
        embedding_ids_sparse,
        sample_ids_sparse,
        gains_sparse,
        *_,
    ) = sparse_tensor_input_preprocessing(
        indices_tensor,
        values_tensor,
        dense_shape_tensor,
        [self.feature_spec],
        local_device_count=4,
        global_device_count=4,
        num_sc_per_device=4,
        sharding_strategy=ShardingStrategy.Mod,
        has_leading_dimension=has_leading_dimension,
        allow_id_dropping=False,
        batch_number=batch_number,
    )

    numpy_features = np.zeros((16, 16), dtype=np.int32)
    numpy_weights = np.ones((16, 16), dtype=np.float32)
    for i in range(16):
      numpy_features[i] = np.array(
          [
              16 * i,
              16 * i + 1,
              16 * i + 2,
              16 * i + 3,
              16 * i + 4,
              16 * i + 5,
              16 * i + 6,
              16 * i + 7,
              16 * i + 8,
              16 * i + 9,
              16 * i + 10,
              16 * i + 11,
              16 * i + 12,
              16 * i + 13,
              16 * i + 14,
              16 * i + 15,
          ],
          dtype=np.int32,
      )

    batch_number = 42
    local_device_count = 4
    global_device_count = 4
    num_sc_per_device = 4
    (row_pointers_raw, embedding_ids_raw, sample_ids_raw, gains_raw, *_) = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
            [numpy_features],
            [numpy_weights],
            [self.feature_spec],
            local_device_count=local_device_count,
            global_device_count=global_device_count,
            num_sc_per_device=num_sc_per_device,
            sharding_strategy=ShardingStrategy.Mod,
            has_leading_dimension=has_leading_dimension,
            allow_id_dropping=False,
            batch_number=batch_number,
        )
    )
    np.testing.assert_equal(row_pointers_raw, row_pointers_sparse)
    assert_equal_coo_buffer = functools.partial(
        test_utils.assert_equal_coo_buffer,
        local_device_count,
        num_sc_per_device,
        row_pointers_raw[self.stacked_name],
    )
    assert_equal_coo_buffer(
        embedding_ids_sparse[self.stacked_name],
        embedding_ids_raw[self.stacked_name],
    )
    assert_equal_coo_buffer(
        sample_ids_sparse[self.stacked_name],
        sample_ids_raw[self.stacked_name],
    )
    assert_equal_coo_buffer(
        gains_sparse[self.stacked_name],
        gains_raw[self.stacked_name],
    )

  @parameterized.parameters(False, True)
  def test_sparse_tensor_input_with_empty_rows(self, has_leading_dimension):
    # Create a sparse tensor with 2400 rows and 1 column.
    # Only some rows have values, specifically rows 0, 150, 300, 450, ...
    indices = []
    values = []
    for i in range(0, 2251, 150):
      indices.append((i, 0))
      values.append(0)

    sparse_tensor = SparseCooInput(
        indices=indices,
        values=values,
        dense_shape=[2400, 1],
    )
    sparse_tensor_input_preprocessing = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulSparseCooInput
    )
    indices_tensor = [sparse_tensor.indices]
    values_tensor = [sparse_tensor.values]
    dense_shape_tensor = [sparse_tensor.dense_shape]
    self.feature_spec.table_spec.stacked_table_spec = dataclasses.replace(
        self.feature_spec.table_spec.stacked_table_spec,
        suggested_coo_buffer_size_per_device=1000,
        max_ids_per_partition=150,
    )
    batch_number = 42
    (
        row_pointers_sparse,
        embedding_ids_sparse,
        sample_ids_sparse,
        gains_sparse,
        *_,
    ) = sparse_tensor_input_preprocessing(
        indices_tensor,
        values_tensor,
        dense_shape_tensor,
        [self.feature_spec],
        local_device_count=4,
        global_device_count=4,
        num_sc_per_device=4,
        sharding_strategy=ShardingStrategy.Mod,
        has_leading_dimension=has_leading_dimension,
        allow_id_dropping=False,
        batch_number=batch_number,
    )

    numpy_features = []
    numpy_weights = []
    for i in range(2400):
      if i % 150 == 0:
        numpy_features.append(np.array([0], dtype=np.int32))
        numpy_weights.append(np.array([1.0], dtype=np.float32))
      else:
        numpy_features.append(np.array([], dtype=np.int32))
        numpy_weights.append(np.array([], dtype=np.float32))
    numpy_features = np.array(numpy_features, dtype=object)
    numpy_weights = np.array(numpy_weights, dtype=object)

    batch_number = 42
    local_device_count = 4
    global_device_count = 4
    num_sc_per_device = 4
    (row_pointers_raw, embedding_ids_raw, sample_ids_raw, gains_raw, *_) = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
            [numpy_features],
            [numpy_weights],
            [self.feature_spec],
            local_device_count=local_device_count,
            global_device_count=global_device_count,
            num_sc_per_device=num_sc_per_device,
            sharding_strategy=ShardingStrategy.Mod,
            has_leading_dimension=has_leading_dimension,
            allow_id_dropping=False,
            batch_number=batch_number,
        )
    )
    np.testing.assert_equal(
        row_pointers_raw[self.stacked_name],
        row_pointers_sparse[self.stacked_name],
    )
    assert_equal_coo_buffer = functools.partial(
        test_utils.assert_equal_coo_buffer,
        local_device_count,
        num_sc_per_device,
        row_pointers_raw[self.stacked_name],
    )
    assert_equal_coo_buffer(
        embedding_ids_sparse[self.stacked_name],
        embedding_ids_raw[self.stacked_name],
    )
    assert_equal_coo_buffer(
        sample_ids_sparse[self.stacked_name],
        sample_ids_raw[self.stacked_name],
    )
    assert_equal_coo_buffer(
        gains_sparse[self.stacked_name],
        gains_raw[self.stacked_name],
    )


class InputPreprocessingColumnTransformationTest(parameterized.TestCase):
  table_spec = embedding_spec.TableSpec(
      vocabulary_size=16,
      embedding_dim=8,
      initializer=lambda: np.zeros((16, 8), dtype=np.float32),
      optimizer=embedding_spec.SGDOptimizerSpec(
          learning_rate=0.001,
      ),
      combiner="sum",
      name="table_b",
      max_ids_per_partition=8,
      max_unique_ids_per_partition=8,
      _setting_in_stack=embedding_spec.TableSettingInStack(
          stack_name="one_table_to_rule_them_all",
          padded_vocab_size=48,
          padded_embedding_dim=8,
          row_offset_in_shard=0,
          shard_rotation=0,
      ),
      _stacked_table_spec=embedding_spec.StackedTableSpec(
          stack_name="one_table_to_rule_them_all",
          stack_vocab_size=48,
          stack_embedding_dim=8,
          optimizer=embedding_spec.SGDOptimizerSpec(
              learning_rate=0.001,
          ),
          combiner="sum",
          total_sample_count=8,
          max_ids_per_partition=16,
          max_unique_ids_per_partition=16,
      ),
  )
  feature_spec = embedding_spec.FeatureSpec(
      table_spec=table_spec,
      input_shape=[8, 4],
      output_shape=[
          8,
          table_spec.embedding_dim,
      ],
      name="feature_spec_b",
      _id_transformation=embedding_spec.FeatureIdTransformation(
          row_offset=0,
          col_offset=32,
          col_shift=4,
      ),
  )
  input_features_sparse = SparseCooInput(
      indices=[
          [0, 1],
          [0, 2],
          [0, 3],
          [0, 4],
          [1, 0],
          [1, 1],
          [1, 2],
          [1, 3],
          [2, 0],
          [2, 1],
          [2, 2],
          [2, 3],
          [3, 0],
          [3, 1],
          [3, 2],
          [3, 3],
          [4, 0],
          [4, 1],
          [4, 2],
          [4, 3],
          [5, 0],
          [5, 1],
          [5, 2],
          [5, 3],
          [6, 0],
          [6, 1],
          [6, 2],
          [6, 3],
          [7, 0],
          [7, 1],
          [7, 2],
          [7, 3],
      ],
      values=[
          2,
          4,
          6,
          8,
          10,
          12,
          14,
          14,
          1,
          3,
          5,
          7,
          9,
          11,
          13,
          15,
          3,
          4,
          5,
          6,
          7,
          8,
          9,
          10,
          4,
          5,
          6,
          7,
          8,
          9,
          10,
          11,
      ],
      dense_shape=[8, 16],
  )
  input_features = np.array(
      [
          [2, 4, 6, 8],
          [10, 12, 14, 14],
          [1, 3, 5, 7],
          [9, 11, 13, 15],
          [3, 4, 5, 6],
          [7, 8, 9, 10],
          [4, 5, 6, 7],
          [8, 9, 10, 11],
      ],
      dtype=np.int32,
  )
  input_weights = np.array(
      [
          [1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0],
      ],
      dtype=np.float32,
  )
  local_device_count = 1
  global_device_count = 1
  num_sc_per_device = 4

  @parameterized.parameters(False, True)
  def test_transformation_with_col_transformations(self, has_leading_dimension):
    batch_number = 42
    (row_pointers, embedding_ids, sample_ids, gains, *_) = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
            [self.input_features],
            [self.input_weights],
            [self.feature_spec],
            local_device_count=self.local_device_count,
            global_device_count=self.global_device_count,
            num_sc_per_device=self.num_sc_per_device,
            sharding_strategy=ShardingStrategy.Mod,
            has_leading_dimension=has_leading_dimension,
            allow_id_dropping=False,
            batch_number=batch_number,
        )
    )

    sparse_tensor_input_preprocessing = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulSparseCooInput
    )
    indices_tensor = [self.input_features_sparse.indices]
    values_tensor = [self.input_features_sparse.values]
    dense_shape_tensor = [self.input_features_sparse.dense_shape]
    batch_number = 42
    (
        row_pointers_sparse,
        embedding_ids_sparse,
        sample_ids_sparse,
        gains_sparse,
        *_,
    ) = sparse_tensor_input_preprocessing(
        indices_tensor,
        values_tensor,
        dense_shape_tensor,
        [self.feature_spec],
        local_device_count=self.local_device_count,
        global_device_count=self.global_device_count,
        num_sc_per_device=self.num_sc_per_device,
        sharding_strategy=ShardingStrategy.Mod,
        has_leading_dimension=has_leading_dimension,
        allow_id_dropping=False,
        batch_number=batch_number,
    )

    stack_name = "one_table_to_rule_them_all"
    np.testing.assert_equal(
        row_pointers[stack_name], row_pointers_sparse[stack_name]
    )
    assert_equal_coo_buffer = functools.partial(
        test_utils.assert_equal_coo_buffer,
        self.local_device_count,
        self.num_sc_per_device,
        row_pointers[stack_name],
    )
    assert_equal_coo_buffer(
        embedding_ids_sparse[stack_name],
        embedding_ids[stack_name],
    )
    assert_equal_coo_buffer(
        sample_ids_sparse[stack_name],
        sample_ids[stack_name],
    )
    assert_equal_coo_buffer(
        gains_sparse[stack_name],
        gains[stack_name],
    )


class InputPreprocessingTableStackingTest(parameterized.TestCase):
  table_spec_a = embedding_spec.TableSpec(
      vocabulary_size=32,
      embedding_dim=8,
      initializer=lambda: np.zeros((32, 8), dtype=np.float32),
      optimizer=embedding_spec.SGDOptimizerSpec(
          learning_rate=0.001,
      ),
      combiner="sum",
      name="table_a",
      max_ids_per_partition=8,
      max_unique_ids_per_partition=8,
      _setting_in_stack=embedding_spec.TableSettingInStack(
          stack_name="one_table_to_rule_them_all",
          padded_vocab_size=48,
          padded_embedding_dim=8,
          row_offset_in_shard=0,
          shard_rotation=0,
      ),
      _stacked_table_spec=embedding_spec.StackedTableSpec(
          stack_name="one_table_to_rule_them_all",
          stack_vocab_size=48,
          stack_embedding_dim=8,
          optimizer=embedding_spec.SGDOptimizerSpec(
              learning_rate=0.001,
          ),
          combiner="sum",
          total_sample_count=8,
          max_ids_per_partition=16,
          max_unique_ids_per_partition=16,
      ),
  )
  table_spec_b = embedding_spec.TableSpec(
      vocabulary_size=16,
      embedding_dim=8,
      initializer=lambda: np.zeros((16, 8), dtype=np.float32),
      optimizer=embedding_spec.SGDOptimizerSpec(
          learning_rate=0.001,
      ),
      combiner="sum",
      name="table_b",
      max_ids_per_partition=8,
      max_unique_ids_per_partition=8,
      _setting_in_stack=embedding_spec.TableSettingInStack(
          stack_name="one_table_to_rule_them_all",
          padded_vocab_size=48,
          padded_embedding_dim=8,
          row_offset_in_shard=0,
          shard_rotation=0,
      ),
      _stacked_table_spec=embedding_spec.StackedTableSpec(
          stack_name="one_table_to_rule_them_all",
          stack_vocab_size=48,
          stack_embedding_dim=8,
          optimizer=embedding_spec.SGDOptimizerSpec(
              learning_rate=0.001,
          ),
          combiner="sum",
          total_sample_count=8,
          max_ids_per_partition=16,
          max_unique_ids_per_partition=16,
      ),
  )
  feature_spec_a = embedding_spec.FeatureSpec(
      table_spec=table_spec_a,
      input_shape=[4, 4],
      output_shape=[
          4,
          table_spec_a.embedding_dim,
      ],
      name="feature_spec_a",
      _id_transformation=embedding_spec.FeatureIdTransformation(
          row_offset=0,
          col_offset=0,
          col_shift=0,
      ),
  )
  feature_spec_b = embedding_spec.FeatureSpec(
      table_spec=table_spec_b,
      input_shape=[8, 4],
      output_shape=[
          8,
          table_spec_b.embedding_dim,
      ],
      name="feature_spec_b",
      _id_transformation=embedding_spec.FeatureIdTransformation(
          row_offset=8,
          col_offset=32,
          col_shift=0,
      ),
  )
  input_features_sparse_a = SparseCooInput(
      indices=[
          [0, 0],
          [0, 1],
          [0, 2],
          [0, 3],
          [0, 4],
          [0, 5],
          [0, 6],
          [0, 7],
          [1, 0],
          [1, 1],
          [1, 2],
          [1, 3],
          [1, 4],
          [1, 5],
          [1, 6],
          [1, 7],
          [2, 0],
          [2, 1],
          [2, 2],
          [2, 3],
          [2, 4],
          [2, 5],
          [2, 6],
          [2, 7],
          [3, 0],
          [3, 1],
          [3, 2],
          [3, 3],
          [3, 4],
          [3, 5],
          [3, 6],
          [3, 7],
          [4, 0],
          [4, 1],
          [4, 2],
          [4, 3],
          [4, 4],
          [4, 5],
          [4, 6],
          [4, 7],
          [5, 0],
          [5, 1],
          [5, 2],
          [5, 3],
          [5, 4],
          [5, 5],
          [5, 6],
          [5, 7],
          [6, 0],
          [6, 1],
          [6, 2],
          [6, 3],
          [6, 4],
          [6, 5],
          [6, 6],
          [6, 7],
          [7, 0],
          [7, 1],
          [7, 2],
          [7, 3],
          [7, 4],
          [7, 5],
          [7, 6],
          [7, 7],
      ],
      values=[
          5,
          18,
          0,
          20,
          0,
          2,
          31,
          3,
          18,
          0,
          20,
          6,
          1,
          28,
          5,
          8,
          0,
          20,
          6,
          15,
          12,
          7,
          3,
          11,
          18,
          0,
          7,
          3,
          6,
          4,
          19,
          2,
          18,
          0,
          20,
          6,
          1,
          28,
          5,
          8,
          0,
          20,
          6,
          15,
          12,
          7,
          3,
          11,
          5,
          18,
          0,
          20,
          0,
          2,
          31,
          3,
          18,
          0,
          20,
          6,
          1,
          28,
          5,
          8,
      ],
      dense_shape=[8, 48],
  )
  input_features_a = np.array(
      [
          [5, 18, 0, 20, 0, 2, 31, 3],
          [18, 0, 20, 6, 1, 28, 5, 8],
          [0, 20, 6, 15, 12, 7, 3, 11],
          [18, 0, 7, 3, 6, 4, 19, 2],
          [18, 0, 20, 6, 1, 28, 5, 8],
          [0, 20, 6, 15, 12, 7, 3, 11],
          [5, 18, 0, 20, 0, 2, 31, 3],
          [18, 0, 20, 6, 1, 28, 5, 8],
      ],
      dtype=np.int32,
  )
  input_weights_a = np.array(
      [
          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      ],
      np.float32,
  )
  input_features_sparse_b = SparseCooInput(
      indices=[
          [0, 1],
          [0, 2],
          [0, 3],
          [0, 4],
          [1, 0],
          [1, 1],
          [1, 2],
          [1, 3],
          [2, 0],
          [2, 1],
          [2, 2],
          [2, 3],
          [3, 0],
          [3, 1],
          [3, 2],
          [3, 3],
          [4, 0],
          [4, 1],
          [4, 2],
          [4, 3],
          [5, 0],
          [5, 1],
          [5, 2],
          [5, 3],
          [6, 0],
          [6, 1],
          [6, 2],
          [6, 3],
          [7, 0],
          [7, 1],
          [7, 2],
          [7, 3],
      ],
      values=[
          2,
          4,
          6,
          8,
          10,
          12,
          14,
          14,
          1,
          3,
          5,
          7,
          9,
          11,
          13,
          15,
          3,
          4,
          5,
          6,
          7,
          8,
          9,
          10,
          4,
          5,
          6,
          7,
          8,
          9,
          10,
          11,
      ],
      dense_shape=[8, 16],
  )
  input_features_b = np.array(
      [
          [2, 4, 6, 8],
          [10, 12, 14, 14],
          [1, 3, 5, 7],
          [9, 11, 13, 15],
          [3, 4, 5, 6],
          [7, 8, 9, 10],
          [4, 5, 6, 7],
          [8, 9, 10, 11],
      ],
      dtype=np.int32,
  )
  input_weights_b = np.array(
      [
          [1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0],
      ],
      dtype=np.float32,
  )

  @parameterized.parameters(False, True)
  def test_feature_stacking_single_chip(self, has_leading_dimension):
    # Now create another feature spec that will be stacked with
    # `self.feature_spec_a`.
    feature_spec_a2 = embedding_spec.FeatureSpec(
        table_spec=self.table_spec_a,
        input_shape=[4, 4],
        output_shape=[
            4,
            self.table_spec_a.embedding_dim,
        ],
        name="feature_spec_a",
        _id_transformation=embedding_spec.FeatureIdTransformation(
            row_offset=8,
            col_offset=0,
            col_shift=0,
        ),
    )

    input_features_sparse_a2 = SparseCooInput(
        indices=[
            [0, 0],
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [0, 5],
            [0, 6],
            [0, 7],
            [1, 0],
            [1, 1],
            [1, 2],
            [1, 3],
            [1, 4],
            [1, 5],
            [1, 6],
            [1, 7],
            [2, 0],
            [2, 1],
            [2, 2],
            [2, 3],
            [2, 4],
            [2, 5],
            [2, 6],
            [2, 7],
            [3, 0],
            [3, 1],
            [3, 2],
            [3, 3],
            [3, 4],
            [3, 5],
            [3, 6],
            [3, 7],
        ],
        values=[
            18,
            0,
            20,
            6,
            1,
            28,
            5,
            8,
            0,
            20,
            6,
            15,
            12,
            7,
            3,
            11,
            5,
            18,
            0,
            20,
            0,
            2,
            31,
            3,
            18,
            0,
            20,
            6,
            1,
            28,
            5,
            8,
        ],
        dense_shape=[4, 32],
    )
    input_features_a2 = np.array(
        [
            [18, 0, 20, 6, 1, 28, 5, 8],
            [0, 20, 6, 15, 12, 7, 3, 11],
            [5, 18, 0, 20, 0, 2, 31, 3],
            [18, 0, 20, 6, 1, 28, 5, 8],
        ],
        dtype=np.int32,
    )
    input_weights_a2 = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        np.float32,
    )

    # Preprocess inputs for the stacked features.
    batch_number = 42
    local_device_count = 1
    global_device_count = 1
    num_sc_per_device = 4
    row_pointers, embedding_ids, sample_ids, gains, *_ = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
            [self.input_features_a, input_features_a2],
            [self.input_weights_a, input_weights_a2],
            [self.feature_spec_a, feature_spec_a2],
            local_device_count=local_device_count,
            global_device_count=global_device_count,
            num_sc_per_device=num_sc_per_device,
            sharding_strategy=ShardingStrategy.Mod,
            has_leading_dimension=has_leading_dimension,
            allow_id_dropping=False,
            batch_number=batch_number,
        )
    )
    sparse_tensor_input_preprocessing = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulSparseCooInput
    )
    indices = [
        self.input_features_sparse_a.indices,
        input_features_sparse_a2.indices,
    ]
    values = [
        self.input_features_sparse_a.values,
        input_features_sparse_a2.values,
    ]
    dense_shapes = [
        self.input_features_sparse_a.dense_shape,
        input_features_sparse_a2.dense_shape,
    ]
    batch_number += 1
    (
        row_pointers_sparse,
        embedding_ids_sparse,
        sample_ids_sparse,
        gains_sparse,
        *_,
    ) = sparse_tensor_input_preprocessing(
        indices,
        values,
        dense_shapes,
        [self.feature_spec_a, feature_spec_a2],
        local_device_count=local_device_count,
        global_device_count=global_device_count,
        num_sc_per_device=num_sc_per_device,
        sharding_strategy=ShardingStrategy.Mod,
        has_leading_dimension=has_leading_dimension,
        allow_id_dropping=False,
        batch_number=batch_number,
    )
    stack_name = "one_table_to_rule_them_all"
    np.testing.assert_equal(
        row_pointers[stack_name], row_pointers_sparse[stack_name]
    )
    assert_equal_coo_buffer = functools.partial(
        test_utils.assert_equal_coo_buffer,
        local_device_count,
        num_sc_per_device,
        row_pointers[stack_name],
    )
    assert_equal_coo_buffer(
        embedding_ids_sparse[stack_name],
        embedding_ids[stack_name],
    )
    assert_equal_coo_buffer(
        sample_ids_sparse[stack_name],
        sample_ids[stack_name],
    )
    assert_equal_coo_buffer(
        gains_sparse[stack_name],
        gains[stack_name],
    )

  @parameterized.parameters(False, True)
  def test_table_stacking_single_chip(self, has_leading_dimension):
    batch_number = 42
    local_device_count = 1
    global_device_count = 1
    num_sc_per_device = 4
    row_pointers, embedding_ids, sample_ids, gains, *_ = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
            [self.input_features_a, self.input_features_b],
            [self.input_weights_a, self.input_weights_b],
            [self.feature_spec_a, self.feature_spec_b],
            local_device_count=local_device_count,
            global_device_count=global_device_count,
            num_sc_per_device=num_sc_per_device,
            sharding_strategy=ShardingStrategy.Mod,
            has_leading_dimension=has_leading_dimension,
            allow_id_dropping=False,
            batch_number=batch_number,
        )
    )
    sparse_tensor_input_preprocessing = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulSparseCooInput
    )
    indices = [
        self.input_features_sparse_a.indices,
        self.input_features_sparse_b.indices,
    ]
    values = [
        self.input_features_sparse_a.values,
        self.input_features_sparse_b.values,
    ]
    dense_shapes = [
        self.input_features_sparse_a.dense_shape,
        self.input_features_sparse_b.dense_shape,
    ]
    batch_number += 1
    (
        row_pointers_sparse,
        embedding_ids_sparse,
        sample_ids_sparse,
        gains_sparse,
        *_,
    ) = sparse_tensor_input_preprocessing(
        indices,
        values,
        dense_shapes,
        [self.feature_spec_a, self.feature_spec_b],
        local_device_count=local_device_count,
        global_device_count=global_device_count,
        num_sc_per_device=num_sc_per_device,
        sharding_strategy=ShardingStrategy.Mod,
        has_leading_dimension=has_leading_dimension,
        allow_id_dropping=False,
        batch_number=batch_number,
    )

    stack_name = "one_table_to_rule_them_all"
    np.testing.assert_equal(
        row_pointers[stack_name], row_pointers_sparse[stack_name]
    )
    assert_equal_coo_buffer = functools.partial(
        test_utils.assert_equal_coo_buffer,
        local_device_count,
        num_sc_per_device,
        row_pointers[stack_name],
    )
    assert_equal_coo_buffer(
        embedding_ids_sparse[stack_name],
        embedding_ids[stack_name],
    )
    assert_equal_coo_buffer(
        sample_ids_sparse[stack_name],
        sample_ids[stack_name],
    )
    assert_equal_coo_buffer(
        gains_sparse[stack_name],
        gains[stack_name],
    )

  @parameterized.parameters(False, True)
  def test_multi_process_stacking(self, has_leading_dimension):
    feature_spec_1 = embedding_spec.FeatureSpec(
        table_spec=self.table_spec_a,
        input_shape=[16, 8],
        output_shape=[
            4,
            self.table_spec_a.embedding_dim,
        ],
        name="feature_spec_a",
        _id_transformation=embedding_spec.FeatureIdTransformation(
            row_offset=0,
            col_offset=0,
            col_shift=0,
        ),
    )
    feature_spec_2 = embedding_spec.FeatureSpec(
        table_spec=self.table_spec_b,
        input_shape=[16, 4],
        output_shape=[
            8,
            self.table_spec_b.embedding_dim,
        ],
        name="feature_spec_b",
        _id_transformation=embedding_spec.FeatureIdTransformation(
            row_offset=8,
            col_offset=32,
            col_shift=0,
        ),
    )
    batch_number = 42
    local_device_count = 1
    global_device_count = 2
    num_sc_per_device = 4
    (row_pointers, embedding_ids, sample_ids, gains, *_) = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
            [self.input_features_a, self.input_features_b],
            [self.input_weights_a, self.input_weights_b],
            [feature_spec_1, feature_spec_2],
            local_device_count=local_device_count,
            global_device_count=global_device_count,
            num_sc_per_device=num_sc_per_device,
            sharding_strategy=ShardingStrategy.Mod,
            has_leading_dimension=has_leading_dimension,
            allow_id_dropping=False,
            batch_number=batch_number,
        )
    )
    batch_number += 1
    sparse_tensor_input_preprocessing = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulSparseCooInput
    )
    indices = [
        self.input_features_sparse_a.indices,
        self.input_features_sparse_b.indices,
    ]
    values = [
        self.input_features_sparse_a.values,
        self.input_features_sparse_b.values,
    ]
    dense_shapes = [
        self.input_features_sparse_a.dense_shape,
        self.input_features_sparse_b.dense_shape,
    ]
    (
        row_pointers_sparse,
        embedding_ids_sparse,
        sample_ids_sparse,
        gains_sparse,
        *_,
    ) = sparse_tensor_input_preprocessing(
        indices,
        values,
        dense_shapes,
        [feature_spec_1, feature_spec_2],
        local_device_count=local_device_count,
        global_device_count=global_device_count,
        num_sc_per_device=num_sc_per_device,
        sharding_strategy=ShardingStrategy.Mod,
        has_leading_dimension=has_leading_dimension,
        allow_id_dropping=False,
        batch_number=batch_number,
    )
    stack_name = "one_table_to_rule_them_all"
    np.testing.assert_equal(
        row_pointers[stack_name], row_pointers_sparse[stack_name]
    )
    assert_equal_coo_buffer = functools.partial(
        test_utils.assert_equal_coo_buffer,
        local_device_count,
        num_sc_per_device,
        row_pointers[stack_name],
    )
    assert_equal_coo_buffer(
        embedding_ids_sparse[stack_name],
        embedding_ids[stack_name],
    )
    assert_equal_coo_buffer(
        sample_ids_sparse[stack_name],
        sample_ids[stack_name],
    )
    assert_equal_coo_buffer(
        gains_sparse[stack_name],
        gains[stack_name],
    )

  @parameterized.parameters(False, True)
  def test_table_stacking_multi_chip(self, has_leading_dimension):
    batch_number = 42
    local_device_count = 2
    global_device_count = 2
    num_sc_per_device = 4
    row_pointers, embedding_ids, sample_ids, gains, *_ = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
            [self.input_features_a, self.input_features_b],
            [self.input_weights_a, self.input_weights_b],
            [self.feature_spec_a, self.feature_spec_b],
            local_device_count=local_device_count,
            global_device_count=global_device_count,
            num_sc_per_device=num_sc_per_device,
            sharding_strategy=ShardingStrategy.Mod,
            has_leading_dimension=has_leading_dimension,
            allow_id_dropping=False,
            batch_number=batch_number,
        )
    )
    sparse_tensor_input_preprocessing = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulSparseCooInput
    )
    indices = [
        self.input_features_sparse_a.indices,
        self.input_features_sparse_b.indices,
    ]
    values = [
        self.input_features_sparse_a.values,
        self.input_features_sparse_b.values,
    ]
    dense_shapes = [
        self.input_features_sparse_a.dense_shape,
        self.input_features_sparse_b.dense_shape,
    ]
    batch_number += 1
    (
        row_pointers_sparse,
        embedding_ids_sparse,
        sample_ids_sparse,
        gains_sparse,
        *_,
    ) = sparse_tensor_input_preprocessing(
        indices,
        values,
        dense_shapes,
        [self.feature_spec_a, self.feature_spec_b],
        local_device_count=local_device_count,
        global_device_count=global_device_count,
        num_sc_per_device=num_sc_per_device,
        sharding_strategy=ShardingStrategy.Mod,
        has_leading_dimension=has_leading_dimension,
        allow_id_dropping=False,
        batch_number=batch_number,
    )
    stack_name = "one_table_to_rule_them_all"
    np.testing.assert_equal(
        row_pointers[stack_name], row_pointers_sparse[stack_name]
    )
    assert_equal_coo_buffer = functools.partial(
        test_utils.assert_equal_coo_buffer,
        local_device_count,
        num_sc_per_device,
        row_pointers[stack_name],
    )
    assert_equal_coo_buffer(
        embedding_ids_sparse[stack_name],
        embedding_ids[stack_name],
    )
    assert_equal_coo_buffer(
        sample_ids_sparse[stack_name],
        sample_ids[stack_name],
    )
    assert_equal_coo_buffer(
        gains_sparse[stack_name],
        gains[stack_name],
    )


class MeanCombinerTest(parameterized.TestCase):
  stacked_name = "one_table_to_rule_them_all"
  table_spec = embedding_spec.TableSpec(
      vocabulary_size=512,
      embedding_dim=8,
      initializer=lambda: np.zeros((512, 8), dtype=np.float32),
      optimizer=embedding_spec.SGDOptimizerSpec(
          learning_rate=0.001,
      ),
      combiner="mean",
      name="table_b",
      max_ids_per_partition=64,
      max_unique_ids_per_partition=64,
      _setting_in_stack=embedding_spec.TableSettingInStack(
          stack_name=stacked_name,
          padded_vocab_size=512,
          padded_embedding_dim=8,
          row_offset_in_shard=0,
          shard_rotation=0,
      ),
      _stacked_table_spec=embedding_spec.StackedTableSpec(
          stack_name=stacked_name,
          stack_vocab_size=512,
          stack_embedding_dim=8,
          optimizer=embedding_spec.SGDOptimizerSpec(
              learning_rate=0.001,
          ),
          combiner="mean",
          total_sample_count=16,
          max_ids_per_partition=64,
          max_unique_ids_per_partition=64,
      ),
  )
  feature_spec = embedding_spec.FeatureSpec(
      table_spec=table_spec,
      input_shape=[16, 16],
      output_shape=[
          16,
          table_spec.embedding_dim,
      ],
      name="feature_spec_b",
      _id_transformation=embedding_spec.FeatureIdTransformation(
          row_offset=0,
          col_offset=0,
          col_shift=0,
      ),
  )

  @parameterized.parameters(False, True)
  def test_sparse_tensor_input(self, has_leading_dimension):
    indices = []
    values = []
    for i in range(16):
      for j in range(16):
        indices.append((i, j))
    for i in range(16 * 16):
      values.append(i)

    sparse_tensor = SparseCooInput(
        indices=indices,
        values=values,
        dense_shape=[16, 512],
    )
    sparse_tensor_input_preprocessing = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulSparseCooInput
    )
    indices_tensor = [sparse_tensor.indices]
    values_tensor = [sparse_tensor.values]
    dense_shape_tensor = [sparse_tensor.dense_shape]
    batch_number = 42
    (
        row_pointers_sparse,
        embedding_ids_sparse,
        sample_ids_sparse,
        gains_sparse,
        *_,
    ) = sparse_tensor_input_preprocessing(
        indices_tensor,
        values_tensor,
        dense_shape_tensor,
        [self.feature_spec],
        local_device_count=4,
        global_device_count=4,
        num_sc_per_device=4,
        sharding_strategy=ShardingStrategy.Mod,
        has_leading_dimension=has_leading_dimension,
        allow_id_dropping=False,
        batch_number=batch_number,
    )

    numpy_features = np.zeros((16, 16), dtype=np.int32)
    numpy_weights = np.ones((16, 16), dtype=np.float32)
    for i in range(16):
      numpy_features[i] = np.array(
          [
              16 * i,
              16 * i + 1,
              16 * i + 2,
              16 * i + 3,
              16 * i + 4,
              16 * i + 5,
              16 * i + 6,
              16 * i + 7,
              16 * i + 8,
              16 * i + 9,
              16 * i + 10,
              16 * i + 11,
              16 * i + 12,
              16 * i + 13,
              16 * i + 14,
              16 * i + 15,
          ],
          dtype=np.int32,
      )
    batch_number = 42
    local_device_count = 4
    global_device_count = 4
    num_sc_per_device = 4
    (row_pointers_raw, embedding_ids_raw, sample_ids_raw, gains_raw, *_) = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
            [numpy_features],
            [numpy_weights],
            [self.feature_spec],
            local_device_count=local_device_count,
            global_device_count=global_device_count,
            num_sc_per_device=num_sc_per_device,
            sharding_strategy=ShardingStrategy.Mod,
            has_leading_dimension=has_leading_dimension,
            allow_id_dropping=False,
            batch_number=batch_number,
        )
    )
    np.testing.assert_equal(
        row_pointers_sparse[self.stacked_name],
        row_pointers_raw[self.stacked_name],
    )
    assert_equal_coo_buffer = functools.partial(
        test_utils.assert_equal_coo_buffer,
        local_device_count,
        num_sc_per_device,
        row_pointers_raw[self.stacked_name],
    )
    assert_equal_coo_buffer(
        embedding_ids_sparse[self.stacked_name],
        embedding_ids_raw[self.stacked_name],
    )
    assert_equal_coo_buffer(
        sample_ids_sparse[self.stacked_name],
        sample_ids_raw[self.stacked_name],
    )
    assert_equal_coo_buffer(
        gains_sparse[self.stacked_name],
        gains_raw[self.stacked_name],
    )


if __name__ == "__main__":
  absltest.main()
