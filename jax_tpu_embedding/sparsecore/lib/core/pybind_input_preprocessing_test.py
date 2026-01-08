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
import math

from absl.testing import absltest
from absl.testing import parameterized
from jax_tpu_embedding.sparsecore.lib.core import constants
from jax_tpu_embedding.sparsecore.lib.core import pybind_input_preprocessing
from jax_tpu_embedding.sparsecore.lib.core import test_utils
from jax_tpu_embedding.sparsecore.lib.core.pybind_input_preprocessing import ShardingStrategy
from jax_tpu_embedding.sparsecore.lib.fdo import file_fdo_client
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
import numpy as np
import portpicker


@dataclasses.dataclass(frozen=True)
class MockDevice:
  id: int


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
          total_sample_count=16,
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
    (row_pointers_raw, embedding_ids_raw, sample_ids_raw, gains_raw, *_) = (
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
    # Construct a feature spec without any col transformation.
    feature_spec_no_col_shift = embedding_spec.FeatureSpec(
        table_spec=self.table_spec,
        input_shape=[8, 4],
        output_shape=[
            8,
            self.table_spec.embedding_dim,
        ],
        name="feature_spec_no_transform",
        _id_transformation=embedding_spec.FeatureIdTransformation(
            row_offset=0,
            col_offset=0,
            col_shift=0,
        ),
    )
    # (col_ids + col_shift) % num_sc_shards +
    #    (col_ids // num_sc_shards * num_sc_shards) + col_offset
    num_sc_shards = int(
        math.log2(self.global_device_count * self.num_sc_per_device)
    )
    input_features_shifted = (
        (self.input_features + self.feature_spec.id_transformation.col_shift)
        % num_sc_shards
        + (self.input_features // num_sc_shards * num_sc_shards)
        + self.feature_spec.id_transformation.col_offset
    )
    batch_number = 42
    (row_pointers, embedding_ids, sample_ids, gains, *_) = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
            [input_features_shifted],
            [self.input_weights],
            [feature_spec_no_col_shift],
            local_device_count=self.local_device_count,
            global_device_count=self.global_device_count,
            num_sc_per_device=self.num_sc_per_device,
            sharding_strategy=ShardingStrategy.Mod,
            has_leading_dimension=has_leading_dimension,
            allow_id_dropping=False,
            batch_number=batch_number,
        )
    )

    np.testing.assert_equal(row_pointers, row_pointers_raw)
    stack_name = "one_table_to_rule_them_all"
    assert_equal_coo_buffer = functools.partial(
        test_utils.assert_equal_coo_buffer,
        self.local_device_count,
        self.num_sc_per_device,
        row_pointers[stack_name],
    )
    assert_equal_coo_buffer(
        embedding_ids[stack_name], embedding_ids_raw[stack_name]
    )
    assert_equal_coo_buffer(sample_ids[stack_name], sample_ids_raw[stack_name])
    assert_equal_coo_buffer(gains[stack_name], gains_raw[stack_name])


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
          total_sample_count=16,
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
          total_sample_count=16,
          max_ids_per_partition=16,
          max_unique_ids_per_partition=16,
      ),
  )
  feature_spec_a = embedding_spec.FeatureSpec(
      table_spec=table_spec_a,
      input_shape=[8, 8],
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
  def test_multi_process_fdo(self, has_leading_dimension):
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
    out_path = self.create_tempdir().full_path
    fdo_client = file_fdo_client.NPZFileFDOClient(out_path)
    batch_number = 42
    *_, stats = pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
        [self.input_features_a, self.input_features_b],
        [self.input_weights_a, self.input_weights_b],
        [feature_spec_1, feature_spec_2],
        local_device_count=1,
        global_device_count=2,
        num_sc_per_device=4,
        sharding_strategy=ShardingStrategy.Mod,
        has_leading_dimension=has_leading_dimension,
        allow_id_dropping=False,
        batch_number=batch_number,
    )
    stats = embedding.SparseDenseMatmulInputStats.from_cc(stats)
    fdo_client.record(stats)
    fdo_client.publish()
    # Duplicated ids on row 0 and 6 are combined.
    np.testing.assert_equal(
        stats.max_ids_per_partition["one_table_to_rule_them_all"],
        np.array([4, 2, 5, 6, 5, 3, 3, 5], dtype=np.int32),
    )
    np.testing.assert_equal(
        stats.max_unique_ids_per_partition["one_table_to_rule_them_all"],
        np.array([3, 2, 4, 5, 4, 2, 3, 4], dtype=np.int32),
    )


class InputPreprocessingTest(parameterized.TestCase):
  table_spec = embedding_spec.TableSpec(
      vocabulary_size=32,
      embedding_dim=8,
      initializer=lambda: np.zeros((32, 8), dtype=np.float32),
      optimizer=embedding_spec.SGDOptimizerSpec(
          learning_rate=0.001,
      ),
      combiner="sum",
      name="table",
      max_ids_per_partition=32,
      max_unique_ids_per_partition=32,
  )
  ragged_input_table_spec = embedding_spec.TableSpec(
      vocabulary_size=16,
      embedding_dim=8,
      initializer=lambda: np.zeros((16, 8), dtype=np.float32),
      optimizer=embedding_spec.SGDOptimizerSpec(
          learning_rate=0.001,
      ),
      combiner="sum",
      name="ragged_features_table",
      max_ids_per_partition=16,
      max_unique_ids_per_partition=16,
  )
  singleton_input_table_spec = embedding_spec.TableSpec(
      vocabulary_size=16,
      embedding_dim=8,
      initializer=lambda: np.zeros((16, 8), dtype=np.float32),
      optimizer=embedding_spec.SGDOptimizerSpec(
          learning_rate=0.001,
      ),
      combiner="sum",
      name="singleton_features_table",
      max_ids_per_partition=12,
      max_unique_ids_per_partition=12,
  )
  feature_spec = embedding_spec.FeatureSpec(
      table_spec=table_spec,
      input_shape=[4, 4],
      output_shape=[
          4,
          table_spec.embedding_dim,
      ],
      name="feature",
  )
  ragged_input_feature_spec = embedding_spec.FeatureSpec(
      table_spec=ragged_input_table_spec,
      input_shape=[8, 4],
      output_shape=[
          8,
          ragged_input_table_spec.embedding_dim,
      ],
      name="ragged_input_feature",
  )
  singleton_input_feature_spec = embedding_spec.FeatureSpec(
      table_spec=singleton_input_table_spec,
      input_shape=[16, 1],
      output_shape=[
          16,
          singleton_input_table_spec.embedding_dim,
      ],
      name="singleton_input_feature",
  )
  input_features = np.array(
      [
          [5, 18, 0, 20, 0, 2, 31, 3],
          [18, 0, 20, 6, 1, 28, 5, 8],
          [0, 20, 6, 15, 12, 7, 3, 11],
          [18, 0, 7, 3, 6, 4, 19, 2],
      ],
      dtype=np.int32,
  )
  ragged_input_features = np.array(
      [
          np.array([5, 18]),
          np.array([0, 2, 31]),
          np.array([18, 0, 20, 6]),
          np.array([1, 28, 5, 8]),
          np.array([0]),
          np.array([12, 7, 3, 11]),
          np.array([18, 0, 7, 3]),
          np.array([6, 4, 19, 2]),
      ],
      dtype=object,
  )
  singleton_input_features = np.array(
      [
          [5],
          [18],
          [0],
          [20],
          [6],
          [15],
          [12],
          [7],
          [3],
          [11],
          [8],
          [26],
          [0],
          [18],
          [7],
          [2],
      ],
      dtype=np.int32,
  )
  singleton_input_weights = np.array(
      [
          [1.0],
          [1.0],
          [1.0],
          [1.0],
          [1.0],
          [1.0],
          [1.0],
          [1.0],
          [1.0],
          [1.0],
          [1.0],
          [1.0],
          [1.0],
          [1.0],
          [1.0],
          [1.0],
      ],
      dtype=np.float32,
  )
  input_weights = np.array(
      [
          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      ],
      np.float32,
  )
  ragged_input_weights = np.array(
      [
          np.array([1.0, 1.0], dtype=np.float32),
          np.array([1.0, 1.0, 1.0], dtype=np.float32),
          np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
          np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
          np.array([1.0], dtype=np.float32),
          np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
          np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
          np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
      ],
      dtype=object,
  )

  def setUp(self):
    super().setUp()
    embedding.prepare_feature_specs_for_training(
        self.feature_spec,
        global_device_count=1,
        num_sc_per_device=4,
    )
    embedding.prepare_feature_specs_for_training(
        self.singleton_input_feature_spec,
        global_device_count=1,
        num_sc_per_device=4,
    )
    embedding.prepare_feature_specs_for_training(
        self.ragged_input_feature_spec,
        global_device_count=1,
        num_sc_per_device=4,
    )

  @parameterized.parameters(False, True)
  def test_correct_input_preprocessing_single_column(
      self, has_leading_dimension
  ):
    local_device_count = 1
    num_sc_per_device = 4
    batch_number = 42
    row_pointers, embedding_ids, sample_ids, gains, *_ = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
            [self.singleton_input_features],
            [self.singleton_input_weights],
            [self.singleton_input_feature_spec],
            local_device_count=local_device_count,
            global_device_count=1,
            num_sc_per_device=num_sc_per_device,
            sharding_strategy=ShardingStrategy.Mod,
            has_leading_dimension=False,
            allow_id_dropping=False,
            batch_number=batch_number,
        )
    )
    # Prepare expected outputs.
    expected_lhs_row_pointers = np.array(
        [
            2,
            9,
            17,
            24,
            24,
            24,
            24,
            24,
            1,
            8,
            9,
            18,
            24,
            24,
            24,
            24,
            1,
            8,
            9,
            18,
            24,
            24,
            24,
            24,
            1,
            8,
            10,
            17,
            24,
            24,
            24,
            24,
        ],
        dtype=np.int32,
    )

    expected_lhs_local_sample_ids = np.full(
        (16 * 4 * 4,),
        constants.PADDING_VALUE,
        dtype=np.int32,
    )
    expected_lhs_local_sample_ids[0] = 2
    expected_lhs_local_sample_ids[1] = 3
    expected_lhs_local_sample_ids[8] = 0
    expected_lhs_local_sample_ids[16] = 1
    expected_lhs_local_sample_ids[64] = 2
    expected_lhs_local_sample_ids[72] = 0
    expected_lhs_local_sample_ids[80] = 3
    expected_lhs_local_sample_ids[81] = 1
    expected_lhs_local_sample_ids[128] = 2
    expected_lhs_local_sample_ids[136] = 3
    expected_lhs_local_sample_ids[144] = 0
    expected_lhs_local_sample_ids[145] = 1
    expected_lhs_local_sample_ids[192] = 0
    expected_lhs_local_sample_ids[200] = 3
    expected_lhs_local_sample_ids[201] = 1
    expected_lhs_local_sample_ids[208] = 2

    expected_lhs_local_embedding_ids = np.full(
        (16 * 4 * 4,),
        constants.PADDING_VALUE,
        dtype=np.int32,
    )
    expected_lhs_local_embedding_ids[0] = 0
    expected_lhs_local_embedding_ids[1] = 5
    expected_lhs_local_embedding_ids[8] = 1
    expected_lhs_local_embedding_ids[16] = 4
    expected_lhs_local_embedding_ids[64] = 3
    expected_lhs_local_embedding_ids[72] = 1
    expected_lhs_local_embedding_ids[80] = 1
    expected_lhs_local_embedding_ids[81] = 3
    expected_lhs_local_embedding_ids[128] = 2
    expected_lhs_local_embedding_ids[136] = 6
    expected_lhs_local_embedding_ids[144] = 0
    expected_lhs_local_embedding_ids[145] = 2
    expected_lhs_local_embedding_ids[192] = 0
    expected_lhs_local_embedding_ids[200] = 0
    expected_lhs_local_embedding_ids[201] = 4
    expected_lhs_local_embedding_ids[208] = 1

    expected_lhs_gains = np.full(
        (16 * 4 * 4,),
        np.nan,
        dtype=np.float32,
    )
    expected_lhs_gains[0] = 1.0
    expected_lhs_gains[1] = 1.0
    expected_lhs_gains[8] = 1.0
    expected_lhs_gains[16] = 1.0
    expected_lhs_gains[64] = 1.0
    expected_lhs_gains[72] = 1.0
    expected_lhs_gains[80] = 1.0
    expected_lhs_gains[81] = 1.0
    expected_lhs_gains[128] = 1.0
    expected_lhs_gains[136] = 1.0
    expected_lhs_gains[144] = 1.0
    expected_lhs_gains[145] = 1.0
    expected_lhs_gains[192] = 1.0
    expected_lhs_gains[200] = 1.0
    expected_lhs_gains[201] = 1.0
    expected_lhs_gains[208] = 1.0

    # Compare the results.
    self.assertLen(row_pointers, 1)
    self.assertLen(embedding_ids, 1)
    self.assertLen(sample_ids, 1)
    self.assertLen(gains, 1)

    # If PMAP, then flatten result for numerical comparisons.
    if has_leading_dimension:
      row_pointers[self.singleton_input_table_spec.name] = row_pointers[
          self.singleton_input_table_spec.name
      ].reshape(-1)
      sample_ids[self.singleton_input_table_spec.name] = sample_ids[
          self.singleton_input_table_spec.name
      ].reshape(-1)
      gains[self.singleton_input_table_spec.name] = gains[
          self.singleton_input_table_spec.name
      ].reshape(-1)
      embedding_ids[self.singleton_input_table_spec.name] = embedding_ids[
          self.singleton_input_table_spec.name
      ].reshape(-1)
    np.testing.assert_equal(
        row_pointers[self.singleton_input_table_spec.name],
        expected_lhs_row_pointers,
    )
    assert_equal_coo_buffer = functools.partial(
        test_utils.assert_equal_coo_buffer,
        local_device_count,
        num_sc_per_device,
        row_pointers[self.singleton_input_table_spec.name],
    )
    assert_equal_coo_buffer(
        sample_ids[self.singleton_input_table_spec.name],
        expected_lhs_local_sample_ids,
    )
    assert_equal_coo_buffer(
        embedding_ids[self.singleton_input_table_spec.name],
        expected_lhs_local_embedding_ids,
    )
    assert_equal_coo_buffer(
        gains[self.singleton_input_table_spec.name],
        expected_lhs_gains,
    )

  @parameterized.parameters(False, True)
  def test_correct_input_preprocessing_multiple_columns(
      self, has_leading_dimension
  ):
    local_device_count = 1
    num_sc_per_device = 4
    batch_number = 42
    row_pointers, embedding_ids, sample_ids, gains, *_ = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
            [self.input_features],
            [self.input_weights],
            [self.feature_spec],
            local_device_count=local_device_count,
            global_device_count=1,
            num_sc_per_device=num_sc_per_device,
            sharding_strategy=ShardingStrategy.Mod,
            has_leading_dimension=has_leading_dimension,
            allow_id_dropping=False,
            batch_number=batch_number,
        )
    )
    with self.subTest(name="RowPointerEquality"):
      expected_lhs_row_pointers = np.array(
          [
              2,
              9,
              18,
              26,
              32,
              32,
              32,
              32,
              4,
              10,
              18,
              24,
              24,
              24,
              24,
              24,
              3,
              8,
              9,
              20,
              24,
              24,
              24,
              24,
              2,
              8,
              11,
              19,
              24,
              24,
              24,
              24,
          ],
          dtype=np.int32,
      )
      if has_leading_dimension:
        row_pointers[self.table_spec.name] = row_pointers[
            self.table_spec.name
        ].reshape(-1)
      np.testing.assert_equal(
          row_pointers[self.table_spec.name], expected_lhs_row_pointers
      )
    assert_equal_coo_buffer = functools.partial(
        test_utils.assert_equal_coo_buffer,
        local_device_count,
        num_sc_per_device,
        row_pointers[self.table_spec.name],
    )
    coo_buffer_size = 32 * 4 * 4
    with self.subTest(name="EmbeddingIdsEqaulity"):
      coo_buffer_size_per_sc = coo_buffer_size // 4

      expected_lhs_local_embedding_ids = np.full(
          (coo_buffer_size,),
          constants.PADDING_VALUE,
          dtype=np.int32,
      )
      expected_lhs_local_embedding_ids[0] = 0
      expected_lhs_local_embedding_ids[1] = 5
      expected_lhs_local_embedding_ids[8] = 1
      expected_lhs_local_embedding_ids[16] = 0
      expected_lhs_local_embedding_ids[17] = 4
      expected_lhs_local_embedding_ids[24] = 0
      expected_lhs_local_embedding_ids[25] = 7

      sc_1_start = coo_buffer_size_per_sc
      expected_lhs_local_embedding_ids[sc_1_start + 0] = 0
      expected_lhs_local_embedding_ids[sc_1_start + 1] = 2
      expected_lhs_local_embedding_ids[sc_1_start + 2] = 5
      expected_lhs_local_embedding_ids[sc_1_start + 3] = 7
      expected_lhs_local_embedding_ids[sc_1_start + 8] = 0
      expected_lhs_local_embedding_ids[sc_1_start + 9] = 1
      expected_lhs_local_embedding_ids[sc_1_start + 16] = 1
      expected_lhs_local_embedding_ids[sc_1_start + 17] = 4

      sc_2_start = coo_buffer_size_per_sc * 2
      expected_lhs_local_embedding_ids[sc_2_start + 0] = 0
      expected_lhs_local_embedding_ids[sc_2_start + 1] = 3
      expected_lhs_local_embedding_ids[sc_2_start + 2] = 5
      expected_lhs_local_embedding_ids[sc_2_start + 8] = 1
      expected_lhs_local_embedding_ids[sc_2_start + 16] = 0
      expected_lhs_local_embedding_ids[sc_2_start + 17] = 1
      expected_lhs_local_embedding_ids[sc_2_start + 18] = 2
      expected_lhs_local_embedding_ids[sc_2_start + 19] = 3

      sc_3_start = coo_buffer_size_per_sc * 3
      expected_lhs_local_embedding_ids[sc_3_start + 0] = 0
      expected_lhs_local_embedding_ids[sc_3_start + 1] = 1
      expected_lhs_local_embedding_ids[sc_3_start + 8] = 0
      expected_lhs_local_embedding_ids[sc_3_start + 9] = 1
      expected_lhs_local_embedding_ids[sc_3_start + 10] = 4
      expected_lhs_local_embedding_ids[sc_3_start + 16] = 0
      expected_lhs_local_embedding_ids[sc_3_start + 17] = 1
      expected_lhs_local_embedding_ids[sc_3_start + 18] = 4

      if has_leading_dimension:
        embedding_ids[self.table_spec.name] = embedding_ids[
            self.table_spec.name
        ].reshape(-1)
      assert_equal_coo_buffer(
          embedding_ids[self.table_spec.name],
          expected_lhs_local_embedding_ids,
      )

    with self.subTest(name="SampleIdsEqaulity"):
      expected_lhs_local_sample_ids = np.full(
          (coo_buffer_size,),
          constants.PADDING_VALUE,
          dtype=np.int32,
      )
      expected_lhs_local_sample_ids[0] = 0
      expected_lhs_local_sample_ids[1] = 0
      expected_lhs_local_sample_ids[8] = 0
      expected_lhs_local_sample_ids[16] = 0
      expected_lhs_local_sample_ids[17] = 0
      expected_lhs_local_sample_ids[24] = 0
      expected_lhs_local_sample_ids[25] = 0

      sc_1_start = coo_buffer_size_per_sc
      expected_lhs_local_sample_ids[sc_1_start + 0] = 0
      expected_lhs_local_sample_ids[sc_1_start + 1] = 0
      expected_lhs_local_sample_ids[sc_1_start + 2] = 0
      expected_lhs_local_sample_ids[sc_1_start + 3] = 0
      expected_lhs_local_sample_ids[sc_1_start + 8] = 0
      expected_lhs_local_sample_ids[sc_1_start + 9] = 0
      expected_lhs_local_sample_ids[sc_1_start + 16] = 0
      expected_lhs_local_sample_ids[sc_1_start + 17] = 0

      sc_2_start = coo_buffer_size_per_sc * 2
      expected_lhs_local_sample_ids[sc_2_start + 0] = 0
      expected_lhs_local_sample_ids[sc_2_start + 1] = 0
      expected_lhs_local_sample_ids[sc_2_start + 2] = 0
      expected_lhs_local_sample_ids[sc_2_start + 8] = 0
      expected_lhs_local_sample_ids[sc_2_start + 16] = 0
      expected_lhs_local_sample_ids[sc_2_start + 17] = 0
      expected_lhs_local_sample_ids[sc_2_start + 18] = 0
      expected_lhs_local_sample_ids[sc_2_start + 19] = 0

      sc_3_start = coo_buffer_size_per_sc * 3
      expected_lhs_local_sample_ids[sc_3_start + 0] = 0
      expected_lhs_local_sample_ids[sc_3_start + 1] = 0
      expected_lhs_local_sample_ids[sc_3_start + 8] = 0
      expected_lhs_local_sample_ids[sc_3_start + 9] = 0
      expected_lhs_local_sample_ids[sc_3_start + 10] = 0
      expected_lhs_local_sample_ids[sc_3_start + 16] = 0
      expected_lhs_local_sample_ids[sc_3_start + 17] = 0
      expected_lhs_local_sample_ids[sc_3_start + 18] = 0

      if has_leading_dimension:
        sample_ids[self.table_spec.name] = sample_ids[
            self.table_spec.name
        ].reshape(-1)
      assert_equal_coo_buffer(
          sample_ids[self.table_spec.name],
          expected_lhs_local_sample_ids,
      )

    with self.subTest(name="GainsEqualityTest"):
      coo_buffer_size_per_sc = coo_buffer_size // 4

      expected_lhs_gains = np.full(
          (coo_buffer_size,),
          np.nan,
          dtype=np.float32,
      )
      expected_lhs_gains[0] = 2.0
      expected_lhs_gains[1] = 1.0
      expected_lhs_gains[8] = 1.0
      expected_lhs_gains[16] = 1.0
      expected_lhs_gains[17] = 1.0
      expected_lhs_gains[24] = 1.0
      expected_lhs_gains[25] = 1.0

      sc_1_start = coo_buffer_size_per_sc
      expected_lhs_gains[sc_1_start + 0] = 1.0
      expected_lhs_gains[sc_1_start + 1] = 1.0
      expected_lhs_gains[sc_1_start + 2] = 1.0
      expected_lhs_gains[sc_1_start + 3] = 1.0
      expected_lhs_gains[sc_1_start + 8] = 1.0
      expected_lhs_gains[sc_1_start + 9] = 1.0
      expected_lhs_gains[sc_1_start + 16] = 1.0
      expected_lhs_gains[sc_1_start + 17] = 1.0

      sc_2_start = coo_buffer_size_per_sc * 2
      expected_lhs_gains[sc_2_start + 0] = 1.0
      expected_lhs_gains[sc_2_start + 1] = 1.0
      expected_lhs_gains[sc_2_start + 2] = 1.0
      expected_lhs_gains[sc_2_start + 8] = 1.0
      expected_lhs_gains[sc_2_start + 16] = 1.0
      expected_lhs_gains[sc_2_start + 17] = 1.0
      expected_lhs_gains[sc_2_start + 18] = 1.0
      expected_lhs_gains[sc_2_start + 19] = 1.0

      sc_3_start = coo_buffer_size_per_sc * 3
      expected_lhs_gains[sc_3_start + 0] = 1.0
      expected_lhs_gains[sc_3_start + 1] = 1.0
      expected_lhs_gains[sc_3_start + 8] = 1.0
      expected_lhs_gains[sc_3_start + 9] = 1.0
      expected_lhs_gains[sc_3_start + 10] = 1.0
      expected_lhs_gains[sc_3_start + 16] = 1.0
      expected_lhs_gains[sc_3_start + 17] = 1.0
      expected_lhs_gains[sc_3_start + 18] = 1.0

      if has_leading_dimension:
        gains[self.table_spec.name] = gains[self.table_spec.name].reshape(-1)
      assert_equal_coo_buffer(gains[self.table_spec.name], expected_lhs_gains)

  @parameterized.parameters(False, True)
  def test_mean_combiner(self, has_leading_dimension):
    table_spec = embedding_spec.TableSpec(
        vocabulary_size=16,
        embedding_dim=8,
        initializer=lambda: np.zeros((16, 8), dtype=np.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(
            learning_rate=0.001,
        ),
        combiner="mean",
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
            combiner="mean",
            total_sample_count=16,
            max_ids_per_partition=32,
            max_unique_ids_per_partition=32,
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
    batch_number = 42
    local_device_count = 1
    num_sc_per_device = 4
    row_pointers, _, _, gains, *_ = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
            [self.input_features],
            [self.input_weights],
            [feature_spec],
            local_device_count=local_device_count,
            global_device_count=1,
            num_sc_per_device=num_sc_per_device,
            sharding_strategy=ShardingStrategy.Mod,
            has_leading_dimension=has_leading_dimension,
            allow_id_dropping=False,
            batch_number=batch_number,
        )
    )

    coo_buffer_size = 32 * 4 * 4

    assert_equal_coo_buffer = functools.partial(
        test_utils.assert_equal_coo_buffer,
        local_device_count,
        num_sc_per_device,
        row_pointers["one_table_to_rule_them_all"],
    )

    with self.subTest(name="GainsEqualityTest"):
      coo_buffer_size_per_sc = coo_buffer_size // 4

      expected_lhs_gains = np.full(
          (coo_buffer_size,),
          np.nan,
          dtype=np.float32,
      )
      expected_lhs_gains[0] = 2.0 / 8
      expected_lhs_gains[1] = 1.0 / 8
      expected_lhs_gains[8] = 1.0 / 8
      expected_lhs_gains[16] = 1.0 / 8
      expected_lhs_gains[17] = 1.0 / 8
      expected_lhs_gains[24] = 1.0 / 8
      expected_lhs_gains[25] = 1.0 / 8

      sc_1_start = coo_buffer_size_per_sc
      expected_lhs_gains[sc_1_start + 0] = 1.0 / 8
      expected_lhs_gains[sc_1_start + 1] = 1.0 / 8
      expected_lhs_gains[sc_1_start + 2] = 1.0 / 8
      expected_lhs_gains[sc_1_start + 3] = 1.0 / 8
      expected_lhs_gains[sc_1_start + 8] = 1.0 / 8
      expected_lhs_gains[sc_1_start + 9] = 1.0 / 8
      expected_lhs_gains[sc_1_start + 16] = 1.0 / 8
      expected_lhs_gains[sc_1_start + 17] = 1.0 / 8

      sc_2_start = coo_buffer_size_per_sc * 2
      expected_lhs_gains[sc_2_start + 0] = 1.0 / 8
      expected_lhs_gains[sc_2_start + 1] = 1.0 / 8
      expected_lhs_gains[sc_2_start + 2] = 1.0 / 8
      expected_lhs_gains[sc_2_start + 8] = 1.0 / 8
      expected_lhs_gains[sc_2_start + 16] = 1.0 / 8
      expected_lhs_gains[sc_2_start + 17] = 1.0 / 8
      expected_lhs_gains[sc_2_start + 18] = 1.0 / 8
      expected_lhs_gains[sc_2_start + 19] = 1.0 / 8

      sc_3_start = coo_buffer_size_per_sc * 3
      expected_lhs_gains[sc_3_start + 0] = 1.0 / 8
      expected_lhs_gains[sc_3_start + 1] = 1.0 / 8
      expected_lhs_gains[sc_3_start + 8] = 1.0 / 8
      expected_lhs_gains[sc_3_start + 9] = 1.0 / 8
      expected_lhs_gains[sc_3_start + 10] = 1.0 / 8
      expected_lhs_gains[sc_3_start + 16] = 1.0 / 8
      expected_lhs_gains[sc_3_start + 17] = 1.0 / 8
      expected_lhs_gains[sc_3_start + 18] = 1.0 / 8

      if has_leading_dimension:
        gains["one_table_to_rule_them_all"] = gains[
            "one_table_to_rule_them_all"
        ].reshape(-1)
      assert_equal_coo_buffer(
          gains["one_table_to_rule_them_all"], expected_lhs_gains
      )

  def test_correct_input_preprocessing_multiple_features_two_local_four_global_devices(
      self,
  ):
    # Outputs with leading dimension (pmap)
    local_device_count = 2
    num_sc_per_device = 4
    batch_number = 42
    row_pointers, embedding_ids, sample_ids, gains, *_ = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
            [
                self.singleton_input_features,
                self.ragged_input_features,
            ],
            [
                self.singleton_input_weights,
                self.ragged_input_weights,
            ],
            [
                self.singleton_input_feature_spec,
                self.ragged_input_feature_spec,
            ],
            local_device_count=local_device_count,
            global_device_count=4,
            num_sc_per_device=num_sc_per_device,
            sharding_strategy=ShardingStrategy.Mod,
            has_leading_dimension=True,
            allow_id_dropping=False,
            batch_number=batch_number,
        )
    )
    # outputs without leading dimension (jit)
    batch_number = 42
    (
        row_pointers_flattened,
        embedding_ids_flattened,
        sample_ids_flattened,
        gains_flattened,
        *_,
    ) = pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
        [
            self.singleton_input_features,
            self.ragged_input_features,
        ],
        [
            self.singleton_input_weights,
            self.ragged_input_weights,
        ],
        [
            self.singleton_input_feature_spec,
            self.ragged_input_feature_spec,
        ],
        local_device_count=2,
        global_device_count=4,
        num_sc_per_device=4,
        sharding_strategy=ShardingStrategy.Mod,
        has_leading_dimension=False,
        allow_id_dropping=False,
        batch_number=batch_number,
    )
    with self.subTest(name="RowPointerEquality"):
      self.assertLen(row_pointers, 2)
      expected_singleton_table_row_pointers = np.array([
          [
              0,
              0,
              1,
              8,
              8,
              9,
              16,
              16,
              16,
              16,
              16,
              16,
              16,
              16,
              16,
              16,
              1,
              8,
              8,
              8,
              9,
              16,
              16,
              16,
              16,
              16,
              16,
              16,
              16,
              16,
              16,
              16,
              0,
              0,
              0,
              0,
              0,
              0,
              1,
              8,
              8,
              8,
              8,
              8,
              8,
              8,
              8,
              9,
              0,
              0,
              0,
              0,
              0,
              0,
              0,
              1,
              8,
              8,
              8,
              8,
              9,
              16,
              16,
              16,
          ],
          [
              0,
              0,
              0,
              1,
              8,
              8,
              8,
              8,
              8,
              8,
              8,
              9,
              16,
              16,
              16,
              16,
              0,
              0,
              0,
              0,
              0,
              0,
              0,
              0,
              1,
              8,
              9,
              16,
              16,
              16,
              16,
              16,
              1,
              8,
              9,
              16,
              16,
              16,
              16,
              16,
              16,
              16,
              16,
              16,
              16,
              16,
              16,
              16,
              0,
              0,
              1,
              8,
              8,
              8,
              8,
              9,
              16,
              16,
              16,
              16,
              16,
              16,
              16,
              16,
          ],
      ])
      expected_ragged_table_pow_pointers = np.array(
          [
              [
                  0,
                  0,
                  1,
                  8,
                  8,
                  9,
                  16,
                  16,
                  16,
                  16,
                  16,
                  16,
                  16,
                  16,
                  16,
                  16,
                  1,
                  8,
                  9,
                  16,
                  16,
                  16,
                  16,
                  16,
                  16,
                  16,
                  16,
                  16,
                  16,
                  16,
                  16,
                  17,
                  1,
                  8,
                  9,
                  16,
                  17,
                  24,
                  25,
                  32,
                  32,
                  32,
                  32,
                  32,
                  32,
                  32,
                  32,
                  32,
                  0,
                  1,
                  8,
                  8,
                  8,
                  9,
                  16,
                  16,
                  17,
                  24,
                  24,
                  24,
                  25,
                  32,
                  32,
                  32,
              ],
              [
                  1,
                  8,
                  8,
                  8,
                  8,
                  8,
                  8,
                  8,
                  8,
                  8,
                  8,
                  8,
                  8,
                  8,
                  8,
                  8,
                  0,
                  0,
                  0,
                  1,
                  8,
                  8,
                  8,
                  9,
                  16,
                  16,
                  16,
                  17,
                  25,
                  32,
                  32,
                  32,
                  1,
                  8,
                  9,
                  17,
                  24,
                  24,
                  24,
                  25,
                  32,
                  32,
                  32,
                  32,
                  32,
                  32,
                  32,
                  32,
                  0,
                  0,
                  1,
                  9,
                  17,
                  24,
                  25,
                  32,
                  32,
                  32,
                  32,
                  32,
                  32,
                  32,
                  32,
                  32,
              ],
          ],
          dtype=np.int32,
      )
      np.testing.assert_equal(
          row_pointers[self.singleton_input_table_spec.name],
          expected_singleton_table_row_pointers,
      )
      np.testing.assert_equal(
          row_pointers[self.ragged_input_table_spec.name],
          expected_ragged_table_pow_pointers,
      )
      np.testing.assert_equal(
          row_pointers_flattened[self.singleton_input_table_spec.name],
          np.ravel(expected_singleton_table_row_pointers),
      )
      np.testing.assert_equal(
          row_pointers_flattened[self.ragged_input_table_spec.name],
          np.ravel(expected_ragged_table_pow_pointers),
      )

    with self.subTest(name="EmbeddingIdsEqaulity"):
      expected_ragged_table_embedding_ids = np.full(
          (2, 1024),
          constants.PADDING_VALUE,
          dtype=np.int32,
      )
      expected_ragged_table_embedding_ids[0, 0] = 1
      expected_ragged_table_embedding_ids[0, 8] = 0
      expected_ragged_table_embedding_ids[0, 256] = 0
      expected_ragged_table_embedding_ids[0, 264] = 0
      expected_ragged_table_embedding_ids[0, 272] = 1
      expected_ragged_table_embedding_ids[0, 512] = 0
      expected_ragged_table_embedding_ids[0, 520] = 1
      expected_ragged_table_embedding_ids[0, 528] = 1
      expected_ragged_table_embedding_ids[0, 536] = 0
      expected_ragged_table_embedding_ids[0, 768] = 0
      expected_ragged_table_embedding_ids[0, 776] = 0
      expected_ragged_table_embedding_ids[0, 784] = 0
      expected_ragged_table_embedding_ids[0, 792] = 1

      expected_ragged_table_embedding_ids[1, 0] = 0
      expected_ragged_table_embedding_ids[1, 256] = 0
      expected_ragged_table_embedding_ids[1, 264] = 0
      expected_ragged_table_embedding_ids[1, 272] = 0
      expected_ragged_table_embedding_ids[1, 280] = 0
      expected_ragged_table_embedding_ids[1, 512] = 0
      expected_ragged_table_embedding_ids[1, 520] = 1
      expected_ragged_table_embedding_ids[1, 528] = 0
      expected_ragged_table_embedding_ids[1, 536] = 0
      expected_ragged_table_embedding_ids[1, 768] = 0
      expected_ragged_table_embedding_ids[1, 776] = 1
      expected_ragged_table_embedding_ids[1, 784] = 0
      expected_ragged_table_embedding_ids[1, 792] = 0

      expected_singleton_table_embedding_ids = np.full(
          (2, 1024),
          constants.PADDING_VALUE,
          dtype=np.int32,
      )
      expected_singleton_table_embedding_ids[0, 0] = 1
      expected_singleton_table_embedding_ids[0, 8] = 0
      expected_singleton_table_embedding_ids[0, 256] = 0
      expected_singleton_table_embedding_ids[0, 264] = 1
      expected_singleton_table_embedding_ids[0, 512] = 0
      expected_singleton_table_embedding_ids[0, 520] = 0
      expected_singleton_table_embedding_ids[0, 768] = 0
      expected_singleton_table_embedding_ids[0, 776] = 0

      expected_singleton_table_embedding_ids[1, 0] = 0
      expected_singleton_table_embedding_ids[1, 8] = 0
      expected_singleton_table_embedding_ids[1, 256] = 0
      expected_singleton_table_embedding_ids[1, 264] = 1
      expected_singleton_table_embedding_ids[1, 512] = 0
      expected_singleton_table_embedding_ids[1, 520] = 1
      expected_singleton_table_embedding_ids[1, 768] = 0
      expected_singleton_table_embedding_ids[1, 776] = 0

      assert_equal_coo_buffer = functools.partial(
          test_utils.assert_equal_coo_buffer,
          local_device_count,
          num_sc_per_device,
      )
      assert_equal_coo_buffer(
          row_pointers[self.singleton_input_table_spec.name],
          embedding_ids[self.singleton_input_table_spec.name],
          expected_singleton_table_embedding_ids,
      )
      assert_equal_coo_buffer(
          row_pointers[self.ragged_input_table_spec.name],
          embedding_ids[self.ragged_input_table_spec.name],
          expected_ragged_table_embedding_ids,
      )
      assert_equal_coo_buffer(
          row_pointers_flattened[self.singleton_input_table_spec.name],
          embedding_ids_flattened[self.singleton_input_table_spec.name],
          np.ravel(expected_singleton_table_embedding_ids),
      )
      assert_equal_coo_buffer(
          row_pointers_flattened[self.ragged_input_table_spec.name],
          embedding_ids_flattened[self.ragged_input_table_spec.name],
          np.ravel(expected_ragged_table_embedding_ids),
      )

    with self.subTest(name="SampleIdsEqaulity"):
      self.assertLen(sample_ids, 2)
      expected_ragged_table_sample_ids = np.full(
          (2, 1024),
          constants.PADDING_VALUE,
          dtype=np.int32,
      )
      expected_ragged_table_sample_ids[0, 0] = 0
      expected_ragged_table_sample_ids[0, 8] = 0
      expected_ragged_table_sample_ids[0, 256] = 0
      expected_ragged_table_sample_ids[0, 264] = 0
      expected_ragged_table_sample_ids[0, 272] = 0
      expected_ragged_table_sample_ids[0, 512] = 0
      expected_ragged_table_sample_ids[0, 520] = 0
      expected_ragged_table_sample_ids[0, 528] = 0
      expected_ragged_table_sample_ids[0, 536] = 0
      expected_ragged_table_sample_ids[0, 768] = 0
      expected_ragged_table_sample_ids[0, 776] = 0
      expected_ragged_table_sample_ids[0, 784] = 0
      expected_ragged_table_sample_ids[0, 792] = 0

      expected_ragged_table_sample_ids[1, 0] = 0
      expected_ragged_table_sample_ids[1, 256] = 0
      expected_ragged_table_sample_ids[1, 264] = 0
      expected_ragged_table_sample_ids[1, 272] = 0
      expected_ragged_table_sample_ids[1, 280] = 0
      expected_ragged_table_sample_ids[1, 512] = 0
      expected_ragged_table_sample_ids[1, 520] = 0
      expected_ragged_table_sample_ids[1, 528] = 0
      expected_ragged_table_sample_ids[1, 536] = 0
      expected_ragged_table_sample_ids[1, 768] = 0
      expected_ragged_table_sample_ids[1, 776] = 0
      expected_ragged_table_sample_ids[1, 784] = 0
      expected_ragged_table_sample_ids[1, 792] = 0

      expected_singleton_table_sample_ids = np.full(
          (2, 1024),
          constants.PADDING_VALUE,
          dtype=np.int32,
      )
      expected_singleton_table_sample_ids[0, 0] = 1
      expected_singleton_table_sample_ids[0, 8] = 0
      expected_singleton_table_sample_ids[0, 256] = 0
      expected_singleton_table_sample_ids[0, 264] = 1
      expected_singleton_table_sample_ids[0, 512] = 0
      expected_singleton_table_sample_ids[0, 520] = 1
      expected_singleton_table_sample_ids[0, 768] = 1
      expected_singleton_table_sample_ids[0, 776] = 0

      expected_singleton_table_sample_ids[1, 0] = 0
      expected_singleton_table_sample_ids[1, 8] = 1
      expected_singleton_table_sample_ids[1, 256] = 0
      expected_singleton_table_sample_ids[1, 264] = 1
      expected_singleton_table_sample_ids[1, 512] = 0
      expected_singleton_table_sample_ids[1, 520] = 1
      expected_singleton_table_sample_ids[1, 768] = 1
      expected_singleton_table_sample_ids[1, 776] = 0
      assert_equal_coo_buffer(
          row_pointers[self.singleton_input_table_spec.name],
          sample_ids[self.singleton_input_table_spec.name],
          expected_singleton_table_sample_ids,
      )
      assert_equal_coo_buffer(
          row_pointers[self.ragged_input_table_spec.name],
          sample_ids[self.ragged_input_table_spec.name],
          expected_ragged_table_sample_ids,
      )
      assert_equal_coo_buffer(
          row_pointers_flattened[self.singleton_input_table_spec.name],
          sample_ids_flattened[self.singleton_input_table_spec.name],
          np.ravel(expected_singleton_table_sample_ids),
      )
      assert_equal_coo_buffer(
          row_pointers_flattened[self.ragged_input_table_spec.name],
          sample_ids_flattened[self.ragged_input_table_spec.name],
          np.ravel(expected_ragged_table_sample_ids),
      )

    with self.subTest(name="GainsEqualityTest"):
      self.assertLen(gains, 2)
      expected_ragged_table_gains = np.full(
          (2, 1024),
          np.nan,
          dtype=np.float32,
      )
      expected_ragged_table_gains[0, 0] = 1.0
      expected_ragged_table_gains[0, 8] = 1.0
      expected_ragged_table_gains[0, 256] = 1.0
      expected_ragged_table_gains[0, 264] = 1.0
      expected_ragged_table_gains[0, 272] = 1.0
      expected_ragged_table_gains[0, 512] = 1.0
      expected_ragged_table_gains[0, 520] = 1.0
      expected_ragged_table_gains[0, 528] = 1.0
      expected_ragged_table_gains[0, 536] = 1.0
      expected_ragged_table_gains[0, 768] = 1.0
      expected_ragged_table_gains[0, 776] = 1.0
      expected_ragged_table_gains[0, 784] = 1.0
      expected_ragged_table_gains[0, 792] = 1.0

      expected_ragged_table_gains[1, 0] = 1.0
      expected_ragged_table_gains[1, 256] = 1.0
      expected_ragged_table_gains[1, 264] = 1.0
      expected_ragged_table_gains[1, 272] = 1.0
      expected_ragged_table_gains[1, 280] = 1.0
      expected_ragged_table_gains[1, 512] = 1.0
      expected_ragged_table_gains[1, 520] = 1.0
      expected_ragged_table_gains[1, 528] = 1.0
      expected_ragged_table_gains[1, 536] = 1.0
      expected_ragged_table_gains[1, 768] = 1.0
      expected_ragged_table_gains[1, 776] = 1.0
      expected_ragged_table_gains[1, 784] = 1.0
      expected_ragged_table_gains[1, 792] = 1.0

      expected_singleton_table_gains = np.full(
          (2, 1024),
          np.nan,
          dtype=np.float32,
      )
      expected_singleton_table_gains[0, 0] = 1.0
      expected_singleton_table_gains[0, 8] = 1.0
      expected_singleton_table_gains[0, 256] = 1.0
      expected_singleton_table_gains[0, 264] = 1.0
      expected_singleton_table_gains[0, 512] = 1.0
      expected_singleton_table_gains[0, 520] = 1.0
      expected_singleton_table_gains[0, 768] = 1.0
      expected_singleton_table_gains[0, 776] = 1.0

      expected_singleton_table_gains[1, 0] = 1.0
      expected_singleton_table_gains[1, 8] = 1.0
      expected_singleton_table_gains[1, 256] = 1.0
      expected_singleton_table_gains[1, 264] = 1.0
      expected_singleton_table_gains[1, 512] = 1.0
      expected_singleton_table_gains[1, 520] = 1.0
      expected_singleton_table_gains[1, 768] = 1.0
      expected_singleton_table_gains[1, 776] = 1.0

      assert_equal_coo_buffer(
          row_pointers[self.singleton_input_table_spec.name],
          gains[self.singleton_input_table_spec.name],
          expected_singleton_table_gains,
      )
      assert_equal_coo_buffer(
          row_pointers[self.ragged_input_table_spec.name],
          gains[self.ragged_input_table_spec.name],
          expected_ragged_table_gains,
      )
      assert_equal_coo_buffer(
          row_pointers_flattened[self.singleton_input_table_spec.name],
          gains_flattened[self.singleton_input_table_spec.name],
          np.ravel(expected_singleton_table_gains),
      )
      assert_equal_coo_buffer(
          row_pointers_flattened[self.ragged_input_table_spec.name],
          gains_flattened[self.ragged_input_table_spec.name],
          np.ravel(expected_ragged_table_gains),
      )

  def _compute_gains(self, weights, combiner: str):
    gains = []
    for w in weights:
      if combiner == "mean":
        w = w / np.sum(w)
      elif combiner == "sqrtn":
        w = w / np.sqrt(np.sum(np.square(w)))

      gains.append(w)

    if weights.ndim == 1:
      gains = np.array(gains, dtype=np.ndarray)
    else:
      gains = np.array(gains, dtype=np.float32)

    return gains

  @parameterized.product(combiner=["mean", "sqrtn"], ragged=[True, False])
  def test_combiner(
      self,
      combiner: str,
      ragged: bool,
  ):
    table_spec = embedding_spec.TableSpec(
        vocabulary_size=32,
        embedding_dim=8,
        initializer=lambda: np.zeros((32, 8), dtype=np.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(
            learning_rate=0.001,
        ),
        combiner=combiner,
        name="table",
        max_ids_per_partition=32,
        max_unique_ids_per_partition=32,
    )
    feature_spec = embedding_spec.FeatureSpec(
        table_spec=table_spec,
        input_shape=[4, None],  # Ragged or dense input.
        output_shape=[
            4,
            table_spec.embedding_dim,
        ],
        name="feature",
    )
    embedding.prepare_feature_specs_for_training(
        feature_spec,
        global_device_count=1,
        num_sc_per_device=4,
    )

    # Generate random samples.
    batch_size = feature_spec.input_shape[0]
    max_ids_per_row = table_spec.max_ids_per_partition // batch_size
    vocab_size = table_spec.vocabulary_size
    rng = np.random.default_rng(12345)
    input_features = []
    input_weights = []
    for _ in range(batch_size):
      n = max_ids_per_row
      if ragged:
        n = rng.integers(low=0, high=max_ids_per_row, size=1)

      input_features.append(
          rng.integers(low=0, high=vocab_size, size=n).astype(np.int32)
      )
      input_weights.append(
          rng.uniform(low=-1.0, high=1.0, size=n).astype(np.float32)
      )

    if ragged:
      input_features = np.array(input_features, dtype=np.ndarray)
      input_weights = np.array(input_weights, dtype=np.ndarray)
    else:
      input_features = np.array(input_features, dtype=np.int32)
      input_weights = np.array(input_weights, dtype=np.float32)

    batch_number = 42
    local_device_count = 1
    global_device_count = 1
    num_sc_per_device = 4
    row_pointers, embedding_ids, sample_ids, gains, *_ = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
            [input_features],
            [input_weights],
            [feature_spec],
            local_device_count=local_device_count,
            global_device_count=global_device_count,
            num_sc_per_device=num_sc_per_device,
            sharding_strategy=ShardingStrategy.Mod,
            has_leading_dimension=False,
            allow_id_dropping=False,
            batch_number=batch_number,
        )
    )

    # Compute expected by re-adjusting the weights using a "sum" combiner.
    table_spec_for_expectation = dataclasses.replace(table_spec, combiner="sum")
    feature_spec_for_expectation = dataclasses.replace(
        feature_spec, table_spec=table_spec_for_expectation
    )
    embedding.prepare_feature_specs_for_training(
        feature_spec_for_expectation,
        global_device_count=1,
        num_sc_per_device=4,
    )
    input_weights = self._compute_gains(input_weights, combiner)
    batch_number = 42
    (
        expected_row_pointers,
        expected_embedding_ids,
        expected_sample_ids,
        expected_gains,
        *_,
    ) = pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
        [input_features],
        [input_weights],
        [feature_spec_for_expectation],
        local_device_count=1,
        global_device_count=1,
        num_sc_per_device=4,
        sharding_strategy=ShardingStrategy.Mod,
        has_leading_dimension=False,
        allow_id_dropping=False,
        batch_number=batch_number,
    )

    np.testing.assert_array_equal(
        row_pointers["table"], expected_row_pointers["table"]
    )
    assert_equal_coo_buffer = functools.partial(
        test_utils.assert_equal_coo_buffer,
        local_device_count,
        num_sc_per_device,
        row_pointers["table"],
    )
    assert_equal_coo_buffer(
        embedding_ids["table"], expected_embedding_ids["table"]
    )
    assert_equal_coo_buffer(sample_ids["table"], expected_sample_ids["table"])
    assert_equal_coo_buffer(gains["table"], expected_gains["table"])

  @parameterized.product(combiner=["mean", "sqrtn", "sum"])
  def test_unity_weights_same_as_none(self, combiner):
    table_spec = dataclasses.replace(self.table_spec, combiner=combiner)
    feature_spec = dataclasses.replace(self.feature_spec, table_spec=table_spec)
    embedding.prepare_feature_specs_for_training(
        feature_spec,
        global_device_count=1,
        num_sc_per_device=4,
    )
    batch_number = 42
    local_device_count = 1
    num_sc_per_device = 4

    # With explicit unity weights
    row_pointers_w, embedding_ids_w, sample_ids_w, gains_w, *_ = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
            features=[self.input_features],
            feature_weights=[self.input_weights],
            feature_specs=[feature_spec],
            local_device_count=local_device_count,
            global_device_count=1,
            num_sc_per_device=num_sc_per_device,
            batch_number=batch_number,
        )
    )

    # With feature_weights=None
    row_pointers_n, embedding_ids_n, sample_ids_n, gains_n, *_ = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
            features=[self.input_features],
            feature_weights=None,
            feature_specs=[feature_spec],
            local_device_count=local_device_count,
            global_device_count=1,
            num_sc_per_device=num_sc_per_device,
            batch_number=batch_number,
        )
    )

    np.testing.assert_equal(row_pointers_w, row_pointers_n)
    stack_name = table_spec.name
    assert_equal_coo_buffer = functools.partial(
        test_utils.assert_equal_coo_buffer,
        local_device_count,
        num_sc_per_device,
        row_pointers_w[stack_name],
    )
    assert_equal_coo_buffer(
        embedding_ids_w[stack_name], embedding_ids_n[stack_name]
    )
    assert_equal_coo_buffer(sample_ids_w[stack_name], sample_ids_n[stack_name])
    assert_equal_coo_buffer(gains_w[stack_name], gains_n[stack_name])


class MinibatchingNodeTest(absltest.TestCase):

  def test_minibatching_node_creation(self):
    num_hosts = 8
    ports = [portpicker.pick_unused_port() for _ in range(num_hosts)]
    for host_id in range(num_hosts):
      minibatching_node = pybind_input_preprocessing.MinibatchingNode(
          host_id,
          num_hosts,
          [f"localhost:{port}" for i, port in enumerate(ports) if i != host_id],
          ports[host_id],
      )
      all_reduce_interface = minibatching_node.get_all_reduce_interface()
      self.assertIsNotNone(all_reduce_interface)


if __name__ == "__main__":
  absltest.main()
