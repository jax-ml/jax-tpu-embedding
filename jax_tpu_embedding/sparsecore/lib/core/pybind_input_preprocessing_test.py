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
import math

from absl.testing import absltest
from absl.testing import parameterized
from jax_tpu_embedding.sparsecore.lib.core import constants
from jax_tpu_embedding.sparsecore.lib.core import pybind_input_preprocessing
from jax_tpu_embedding.sparsecore.lib.core.pybind_input_preprocessing import ShardingStrategy
from jax_tpu_embedding.sparsecore.lib.fdo import file_fdo_client
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
import numpy as np


FeatureStackingStrategy = embedding.FeatureStackingStrategy


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
    (row_pointers_raw, embedding_ids_raw, sample_ids_raw, gains_raw, _) = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
            batch_number,
            [self.input_features],
            [self.input_weights],
            [self.feature_spec],
            local_device_count=self.local_device_count,
            global_device_count=self.global_device_count,
            num_sc_per_device=self.num_sc_per_device,
            sharding_strategy=ShardingStrategy.Mod,
            has_leading_dimension=has_leading_dimension,
            allow_id_dropping=False,
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
    (row_pointers, embedding_ids, sample_ids, gains, _) = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
            batch_number,
            [input_features_shifted],
            [self.input_weights],
            [feature_spec_no_col_shift],
            local_device_count=self.local_device_count,
            global_device_count=self.global_device_count,
            num_sc_per_device=self.num_sc_per_device,
            sharding_strategy=ShardingStrategy.Mod,
            has_leading_dimension=has_leading_dimension,
            allow_id_dropping=False,
        )
    )

    np.testing.assert_equal(row_pointers, row_pointers_raw)
    np.testing.assert_equal(embedding_ids, embedding_ids_raw)
    np.testing.assert_equal(sample_ids, sample_ids_raw)
    np.testing.assert_equal(gains, gains_raw)


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
    (row_pointers, embedding_ids, sample_ids, gains, _) = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
            batch_number,
            [self.input_features_a, self.input_features_b],
            [self.input_weights_a, self.input_weights_b],
            [feature_spec_1, feature_spec_2],
            local_device_count=1,
            global_device_count=2,
            num_sc_per_device=4,
            sharding_strategy=ShardingStrategy.Mod,
            has_leading_dimension=has_leading_dimension,
            allow_id_dropping=False,
        )
    )
    with self.subTest("row_pointers"):
      if has_leading_dimension:
        row_pointers["one_table_to_rule_them_all"] = row_pointers[
            "one_table_to_rule_them_all"
        ].reshape(-1)
      np.testing.assert_equal(
          row_pointers,
          {
              "one_table_to_rule_them_all": np.array(
                  [
                      5,
                      9,
                      21,
                      29,
                      38,
                      42,
                      51,
                      60,
                      7,
                      12,
                      22,
                      29,
                      41,
                      53,
                      61,
                      69,
                      2,
                      10,
                      18,
                      26,
                      34,
                      42,
                      50,
                      58,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                  ],
                  dtype=np.int32,
              )
          },
      )

    with self.subTest("embedding_ids"):
      expected_embedding_ids = np.full((512,), 2**31 - 1, dtype=np.int32)
      expected_embedding_ids[0] = 0
      expected_embedding_ids[1] = 0
      expected_embedding_ids[2] = 0
      expected_embedding_ids[3] = 0
      expected_embedding_ids[4] = 1
      expected_embedding_ids[8] = 0
      expected_embedding_ids[16] = 0
      expected_embedding_ids[17] = 0
      expected_embedding_ids[18] = 2
      expected_embedding_ids[19] = 2
      expected_embedding_ids[20] = 2
      expected_embedding_ids[24] = 0
      expected_embedding_ids[25] = 0
      expected_embedding_ids[26] = 0
      expected_embedding_ids[27] = 1
      expected_embedding_ids[28] = 2
      expected_embedding_ids[32] = 0
      expected_embedding_ids[33] = 1
      expected_embedding_ids[34] = 2
      expected_embedding_ids[35] = 2
      expected_embedding_ids[36] = 2
      expected_embedding_ids[37] = 3
      expected_embedding_ids[40] = 0
      expected_embedding_ids[41] = 0
      expected_embedding_ids[48] = 0
      expected_embedding_ids[49] = 0
      expected_embedding_ids[50] = 0
      expected_embedding_ids[56] = 0
      expected_embedding_ids[57] = 0
      expected_embedding_ids[58] = 1
      expected_embedding_ids[59] = 3
      expected_embedding_ids[128] = 0
      expected_embedding_ids[129] = 0
      expected_embedding_ids[130] = 0
      expected_embedding_ids[131] = 0
      expected_embedding_ids[132] = 1
      expected_embedding_ids[133] = 1
      expected_embedding_ids[134] = 5
      expected_embedding_ids[136] = 0
      expected_embedding_ids[137] = 0
      expected_embedding_ids[138] = 4
      expected_embedding_ids[139] = 5
      expected_embedding_ids[144] = 0
      expected_embedding_ids[145] = 2
      expected_embedding_ids[146] = 2
      expected_embedding_ids[147] = 2
      expected_embedding_ids[148] = 4
      expected_embedding_ids[149] = 5
      expected_embedding_ids[152] = 0
      expected_embedding_ids[153] = 0
      expected_embedding_ids[154] = 1
      expected_embedding_ids[155] = 4
      expected_embedding_ids[156] = 5
      expected_embedding_ids[160] = 1
      expected_embedding_ids[161] = 2
      expected_embedding_ids[162] = 2
      expected_embedding_ids[163] = 2
      expected_embedding_ids[164] = 2
      expected_embedding_ids[165] = 3
      expected_embedding_ids[166] = 3
      expected_embedding_ids[167] = 4
      expected_embedding_ids[168] = 5
      expected_embedding_ids[176] = 0
      expected_embedding_ids[177] = 0
      expected_embedding_ids[178] = 0
      expected_embedding_ids[179] = 4
      expected_embedding_ids[180] = 5
      expected_embedding_ids[184] = 0
      expected_embedding_ids[185] = 0
      expected_embedding_ids[186] = 0
      expected_embedding_ids[187] = 4
      expected_embedding_ids[188] = 5
      expected_embedding_ids[192] = 0
      expected_embedding_ids[193] = 1
      expected_embedding_ids[194] = 3
      expected_embedding_ids[195] = 4
      expected_embedding_ids[196] = 5
      expected_embedding_ids[256] = 5
      expected_embedding_ids[257] = 5
      expected_embedding_ids[264] = 5
      expected_embedding_ids[265] = 5
      expected_embedding_ids[272] = 5
      expected_embedding_ids[273] = 5
      expected_embedding_ids[280] = 4
      expected_embedding_ids[281] = 5
      expected_embedding_ids[288] = 4
      expected_embedding_ids[289] = 4
      expected_embedding_ids[296] = 4
      expected_embedding_ids[297] = 4
      expected_embedding_ids[304] = 4
      expected_embedding_ids[305] = 4
      expected_embedding_ids[312] = 4
      expected_embedding_ids[313] = 4

      if has_leading_dimension:
        embedding_ids["one_table_to_rule_them_all"] = embedding_ids[
            "one_table_to_rule_them_all"
        ].reshape(-1)
      np.testing.assert_equal(
          embedding_ids["one_table_to_rule_them_all"],
          expected_embedding_ids,
      )

    with self.subTest("sample_ids"):
      expected_sample_ids = np.full((512,), 2**31 - 1, dtype=np.int32)
      expected_sample_ids[0] = 0
      expected_sample_ids[1] = 1
      expected_sample_ids[2] = 2
      expected_sample_ids[3] = 3
      expected_sample_ids[4] = 1
      expected_sample_ids[8] = 1
      expected_sample_ids[16] = 0
      expected_sample_ids[17] = 3
      expected_sample_ids[18] = 0
      expected_sample_ids[19] = 1
      expected_sample_ids[20] = 3
      expected_sample_ids[24] = 0
      expected_sample_ids[25] = 2
      expected_sample_ids[26] = 3
      expected_sample_ids[27] = 2
      expected_sample_ids[28] = 3
      expected_sample_ids[32] = 3
      expected_sample_ids[33] = 2
      expected_sample_ids[34] = 0
      expected_sample_ids[35] = 1
      expected_sample_ids[36] = 2
      expected_sample_ids[37] = 1
      expected_sample_ids[40] = 0
      expected_sample_ids[41] = 1
      expected_sample_ids[48] = 1
      expected_sample_ids[49] = 2
      expected_sample_ids[50] = 3
      expected_sample_ids[56] = 2
      expected_sample_ids[57] = 3
      expected_sample_ids[58] = 2
      expected_sample_ids[59] = 0
      expected_sample_ids[128] = 0
      expected_sample_ids[129] = 1
      expected_sample_ids[130] = 2
      expected_sample_ids[131] = 3
      expected_sample_ids[132] = 0
      expected_sample_ids[133] = 3
      expected_sample_ids[134] = 0
      expected_sample_ids[136] = 0
      expected_sample_ids[137] = 3
      expected_sample_ids[138] = 2
      expected_sample_ids[139] = 3
      expected_sample_ids[144] = 2
      expected_sample_ids[145] = 0
      expected_sample_ids[146] = 2
      expected_sample_ids[147] = 3
      expected_sample_ids[148] = 0
      expected_sample_ids[149] = 1
      expected_sample_ids[152] = 1
      expected_sample_ids[153] = 2
      expected_sample_ids[154] = 1
      expected_sample_ids[155] = 2
      expected_sample_ids[156] = 3
      expected_sample_ids[160] = 1
      expected_sample_ids[161] = 0
      expected_sample_ids[162] = 1
      expected_sample_ids[163] = 2
      expected_sample_ids[164] = 3
      expected_sample_ids[165] = 0
      expected_sample_ids[166] = 3
      expected_sample_ids[167] = 0
      expected_sample_ids[168] = 1
      expected_sample_ids[176] = 0
      expected_sample_ids[177] = 2
      expected_sample_ids[178] = 3
      expected_sample_ids[179] = 2
      expected_sample_ids[180] = 3
      expected_sample_ids[184] = 0
      expected_sample_ids[185] = 1
      expected_sample_ids[186] = 3
      expected_sample_ids[187] = 0
      expected_sample_ids[188] = 1
      expected_sample_ids[192] = 1
      expected_sample_ids[193] = 1
      expected_sample_ids[194] = 2
      expected_sample_ids[195] = 2
      expected_sample_ids[196] = 3
      expected_sample_ids[256] = 1
      expected_sample_ids[257] = 3
      expected_sample_ids[264] = 1
      expected_sample_ids[265] = 3
      expected_sample_ids[272] = 1
      expected_sample_ids[273] = 3
      expected_sample_ids[280] = 0
      expected_sample_ids[281] = 3
      expected_sample_ids[288] = 0
      expected_sample_ids[289] = 2
      expected_sample_ids[296] = 0
      expected_sample_ids[297] = 2
      expected_sample_ids[304] = 0
      expected_sample_ids[305] = 2
      expected_sample_ids[312] = 1
      expected_sample_ids[313] = 2
      if has_leading_dimension:
        sample_ids["one_table_to_rule_them_all"] = sample_ids[
            "one_table_to_rule_them_all"
        ].reshape(-1)
      np.testing.assert_equal(
          sample_ids["one_table_to_rule_them_all"],
          expected_sample_ids,
      )

    with self.subTest("gains"):
      expected_gains = np.full((512,), np.nan, dtype=np.float32)
      expected_gains[0] = 2.0
      expected_gains[1] = 1.0
      expected_gains[2] = 1.0
      expected_gains[3] = 1.0
      expected_gains[4] = 1.0
      expected_gains[8] = 1.0
      expected_gains[16] = 1.0
      expected_gains[17] = 1.0
      expected_gains[18] = 1.0
      expected_gains[19] = 1.0
      expected_gains[20] = 1.0
      expected_gains[24] = 1.0
      expected_gains[25] = 1.0
      expected_gains[26] = 1.0
      expected_gains[27] = 1.0
      expected_gains[28] = 1.0
      expected_gains[32] = 1.0
      expected_gains[33] = 1.0
      expected_gains[34] = 1.0
      expected_gains[35] = 1.0
      expected_gains[36] = 1.0
      expected_gains[37] = 1.0
      expected_gains[40] = 1.0
      expected_gains[41] = 1.0
      expected_gains[48] = 1.0
      expected_gains[49] = 1.0
      expected_gains[50] = 1.0
      expected_gains[56] = 1.0
      expected_gains[57] = 1.0
      expected_gains[58] = 1.0
      expected_gains[59] = 1.0
      expected_gains[128] = 1.0
      expected_gains[129] = 1.0
      expected_gains[130] = 2.0
      expected_gains[131] = 1.0
      expected_gains[132] = 1.0
      expected_gains[133] = 1.0
      expected_gains[134] = 1.0
      expected_gains[136] = 1.0
      expected_gains[137] = 1.0
      expected_gains[138] = 1.0
      expected_gains[139] = 1.0
      expected_gains[144] = 1.0
      expected_gains[145] = 1.0
      expected_gains[146] = 1.0
      expected_gains[147] = 1.0
      expected_gains[148] = 1.0
      expected_gains[149] = 1.0
      expected_gains[152] = 1.0
      expected_gains[153] = 1.0
      expected_gains[154] = 1.0
      expected_gains[155] = 1.0
      expected_gains[156] = 1.0
      expected_gains[160] = 1.0
      expected_gains[161] = 1.0
      expected_gains[162] = 1.0
      expected_gains[163] = 1.0
      expected_gains[164] = 1.0
      expected_gains[165] = 1.0
      expected_gains[166] = 1.0
      expected_gains[167] = 1.0
      expected_gains[168] = 1.0
      expected_gains[176] = 1.0
      expected_gains[177] = 1.0
      expected_gains[178] = 1.0
      expected_gains[179] = 1.0
      expected_gains[180] = 1.0
      expected_gains[184] = 1.0
      expected_gains[185] = 1.0
      expected_gains[186] = 1.0
      expected_gains[187] = 1.0
      expected_gains[188] = 2.0
      expected_gains[192] = 1.0
      expected_gains[193] = 1.0
      expected_gains[194] = 1.0
      expected_gains[195] = 1.0
      expected_gains[196] = 1.0
      expected_gains[256] = 1.0
      expected_gains[257] = 1.0
      expected_gains[264] = 1.0
      expected_gains[265] = 1.0
      expected_gains[272] = 1.0
      expected_gains[273] = 1.0
      expected_gains[280] = 1.0
      expected_gains[281] = 1.0
      expected_gains[288] = 1.0
      expected_gains[289] = 1.0
      expected_gains[296] = 1.0
      expected_gains[297] = 1.0
      expected_gains[304] = 1.0
      expected_gains[305] = 1.0
      expected_gains[312] = 1.0
      expected_gains[313] = 1.0
      if has_leading_dimension:
        gains["one_table_to_rule_them_all"] = gains[
            "one_table_to_rule_them_all"
        ].reshape(-1)
      np.testing.assert_equal(
          gains["one_table_to_rule_them_all"],
          expected_gains,
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
    (_, _, _, _, stats) = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
            batch_number,
            [self.input_features_a, self.input_features_b],
            [self.input_weights_a, self.input_weights_b],
            [feature_spec_1, feature_spec_2],
            local_device_count=1,
            global_device_count=2,
            num_sc_per_device=4,
            sharding_strategy=ShardingStrategy.Mod,
            has_leading_dimension=has_leading_dimension,
            allow_id_dropping=False,
        )
    )
    stats = embedding.SparseDenseMatmulInputStats.from_cc(stats)
    fdo_client.record(stats)
    fdo_client.publish()
    # Duplicated ids on row 0 and 6 are combined.
    np.testing.assert_equal(
        stats.max_ids_per_partition["one_table_to_rule_them_all"],
        np.array([7, 4, 6, 5, 9, 5, 5, 5], dtype=np.int32),
    )
    np.testing.assert_equal(
        stats.max_unique_ids_per_partition["one_table_to_rule_them_all"],
        np.array([3, 3, 4, 4, 5, 3, 3, 5], dtype=np.int32),
    )

  @parameterized.parameters(False, True)
  def test_feature_stacking_single_chip(self, has_leading_dimension):
    # Tests feature stacking for a single chip. In the single chip case, the
    # result of the input preprocessing for the stacked features should be
    # identical to the result of just stacking the features and preprocessing
    # the inputs as if they were a single feature.

    # Set up the test. Copy over the table spec but increase the
    # max_ids_per_partition to account for feature stacking.
    table_for_feature_stacking_expectation = embedding_spec.TableSpec(
        vocabulary_size=32,
        embedding_dim=8,
        initializer=lambda: np.zeros((32, 8), dtype=np.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(
            learning_rate=0.001,
        ),
        combiner="sum",
        name="table_a",
        max_ids_per_partition=16,
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
    feature_spec_for_feature_stacking_expectation = embedding_spec.FeatureSpec(
        table_spec=table_for_feature_stacking_expectation,
        input_shape=[4, 4],
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
    # Prepare the expected output.
    feature_stacked_inputs = np.array(
        [
            [5, 18, 0, 20, 0, 2, 31, 3],
            [18, 0, 20, 6, 1, 28, 5, 8],
            [0, 20, 6, 15, 12, 7, 3, 11],
            [18, 0, 7, 3, 6, 4, 19, 2],
            [18, 0, 20, 6, 1, 28, 5, 8],
            [0, 20, 6, 15, 12, 7, 3, 11],
            [5, 18, 0, 20, 0, 2, 31, 3],
            [18, 0, 20, 6, 1, 28, 5, 8],
            [18, 0, 20, 6, 1, 28, 5, 8],
            [0, 20, 6, 15, 12, 7, 3, 11],
            [5, 18, 0, 20, 0, 2, 31, 3],
            [18, 0, 20, 6, 1, 28, 5, 8],
        ],
        dtype=np.int32,
    )
    feature_stacked_weights = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
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

    # Set up the expectation.
    batch_number = 42
    (
        stacked_row_pointers,
        stacked_embedding_ids,
        stacked_sample_ids,
        stacked_gains,
        _,
    ) = pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
        batch_number,
        [feature_stacked_inputs],
        [feature_stacked_weights],
        [feature_spec_for_feature_stacking_expectation],
        local_device_count=1,
        global_device_count=1,
        num_sc_per_device=4,
        sharding_strategy=ShardingStrategy.Mod,
        has_leading_dimension=has_leading_dimension,
        allow_id_dropping=False,
    )

    # Preprocess inputs for the stacked features.
    batch_number += 1
    row_pointers, embedding_ids, sample_ids, gains, _ = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
            batch_number,
            [self.input_features_a, input_features_a2],
            [self.input_weights_a, input_weights_a2],
            [self.feature_spec_a, feature_spec_a2],
            local_device_count=1,
            global_device_count=1,
            num_sc_per_device=4,
            sharding_strategy=ShardingStrategy.Mod,
            has_leading_dimension=has_leading_dimension,
            allow_id_dropping=False,
        )
    )
    np.testing.assert_equal(row_pointers, stacked_row_pointers)
    np.testing.assert_equal(embedding_ids, stacked_embedding_ids)
    np.testing.assert_equal(sample_ids, stacked_sample_ids)
    np.testing.assert_equal(gains, stacked_gains)

  @parameterized.parameters(False, True)
  def test_table_stacking_single_chip(self, has_leading_dimension):
    batch_number = 42
    row_pointers, embedding_ids, sample_ids, gains, _ = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
            batch_number,
            [self.input_features_a, self.input_features_b],
            [self.input_weights_a, self.input_weights_b],
            [self.feature_spec_a, self.feature_spec_b],
            local_device_count=1,
            global_device_count=1,
            num_sc_per_device=4,
            sharding_strategy=ShardingStrategy.Mod,
            has_leading_dimension=has_leading_dimension,
            allow_id_dropping=False,
        )
    )
    # Create a fake table/feature spec to test the assertion against the stacked
    # table.
    # In the case of a single chip, the result of the input preprocessing for
    # the above stacked tables should be identical to the result of just
    # stacking the tables and batches together and preprocessing the inputs as
    # if they were a single table and feature.
    expected_merged_table = embedding_spec.TableSpec(
        vocabulary_size=48,
        embedding_dim=16,
        initializer=lambda: np.zeros((48, 16), dtype=np.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(
            learning_rate=0.001,
        ),
        combiner="sum",
        name="one_table_to_rule_them_all",
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
    )
    expected_merged_feature_spec = embedding_spec.FeatureSpec(
        table_spec=expected_merged_table,
        input_shape=[16, 4],
        output_shape=[
            16,
            expected_merged_table.embedding_dim,
        ],
        name="stacked_feature",
    )
    embedding.prepare_feature_specs_for_training(
        expected_merged_feature_spec,
        global_device_count=1,
        num_sc_per_device=4,
    )
    all_samples = np.array(
        [
            np.array([5, 18, 0, 20, 0, 2, 31, 3], dtype=np.int32),
            np.array([18, 0, 20, 6, 1, 28, 5, 8], dtype=np.int32),
            np.array([0, 20, 6, 15, 12, 7, 3, 11], dtype=np.int32),
            np.array([18, 0, 7, 3, 6, 4, 19, 2], dtype=np.int32),
            np.array([18, 0, 20, 6, 1, 28, 5, 8], dtype=np.int32),
            np.array([0, 20, 6, 15, 12, 7, 3, 11], dtype=np.int32),
            np.array([5, 18, 0, 20, 0, 2, 31, 3], dtype=np.int32),
            np.array([18, 0, 20, 6, 1, 28, 5, 8], dtype=np.int32),
            np.array([2 + 32, 4 + 32, 6 + 32, 8 + 32], dtype=np.int32),
            np.array([10 + 32, 12 + 32, 14 + 32, 14 + 32], dtype=np.int32),
            np.array([1 + 32, 3 + 32, 5 + 32, 7 + 32], dtype=np.int32),
            np.array([9 + 32, 11 + 32, 13 + 32, 15 + 32], dtype=np.int32),
            np.array([3 + 32, 4 + 32, 5 + 32, 6 + 32], dtype=np.int32),
            np.array([7 + 32, 8 + 32, 9 + 32, 10 + 32], dtype=np.int32),
            np.array([4 + 32, 5 + 32, 6 + 32, 7 + 32], dtype=np.int32),
            np.array([8 + 32, 9 + 32, 10 + 32, 11 + 32], dtype=np.int32),
        ],
        dtype=object,
    )
    all_weights = np.array(
        [
            np.array(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32
            ),
            np.array(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32
            ),
            np.array(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32
            ),
            np.array(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32
            ),
            np.array(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32
            ),
            np.array(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32
            ),
            np.array(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32
            ),
            np.array(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32
            ),
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        ],
        dtype=object,
    )
    batch_number = 42
    s_row_pointers, s_embedding_ids, s_sample_ids, s_gains, _ = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
            batch_number,
            [all_samples],
            [all_weights],
            [expected_merged_feature_spec],
            local_device_count=1,
            global_device_count=1,
            num_sc_per_device=4,
            sharding_strategy=ShardingStrategy.Mod,
            has_leading_dimension=has_leading_dimension,
            allow_id_dropping=False,
        )
    )
    np.testing.assert_equal(row_pointers, s_row_pointers)
    np.testing.assert_equal(embedding_ids, s_embedding_ids)
    np.testing.assert_equal(sample_ids, s_sample_ids)
    np.testing.assert_equal(gains, s_gains)

  @parameterized.parameters(False, True)
  def test_table_stacking_multi_chip(self, has_leading_dimension):
    batch_number = 42
    row_pointers, embedding_ids, sample_ids, gains, _ = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
            batch_number,
            [self.input_features_a, self.input_features_b],
            [self.input_weights_a, self.input_weights_b],
            [self.feature_spec_a, self.feature_spec_b],
            local_device_count=2,
            global_device_count=2,
            num_sc_per_device=4,
            sharding_strategy=ShardingStrategy.Mod,
            has_leading_dimension=has_leading_dimension,
            allow_id_dropping=False,
        )
    )
    # For 2 chips, we expect the batches for each feature to be split into 2
    # and then concatenated together.
    # So if feature_a : [a1, a2, a3, a4, a5, a6, a7, a8]
    # and feature_b : [b1, b2, b3, b4, b5, b6, b7, b8]
    # then for chip 1 : [a1, a2, a3, a4, b1, b2, b3, b4]
    # and for chip 2 : [a5, a6, a7, a8, b5, b6, b7, b8]
    with self.subTest("row_pointers"):
      if has_leading_dimension:
        row_pointers["one_table_to_rule_them_all"] = row_pointers[
            "one_table_to_rule_them_all"
        ].reshape(-1)
      np.testing.assert_equal(
          row_pointers,
          {
              "one_table_to_rule_them_all": np.array(
                  [
                      3,
                      9,
                      19,
                      25,
                      35,
                      42,
                      49,
                      57,
                      2,
                      8,
                      10,
                      20,
                      27,
                      32,
                      34,
                      43,
                      1,
                      8,
                      10,
                      16,
                      18,
                      24,
                      26,
                      32,
                      0,
                      2,
                      8,
                      10,
                      16,
                      18,
                      24,
                      26,
                      3,
                      9,
                      17,
                      26,
                      36,
                      41,
                      50,
                      58,
                      3,
                      9,
                      19,
                      25,
                      35,
                      42,
                      49,
                      57,
                      1,
                      9,
                      17,
                      25,
                      33,
                      41,
                      49,
                      57,
                      1,
                      9,
                      17,
                      25,
                      33,
                      41,
                      49,
                      57,
                  ],
                  dtype=np.int32,
              )
          },
      )

    with self.subTest("embedding_ids"):
      expected_embedding_ids = np.full((1024,), 2**31 - 1, dtype=np.int32)
      expected_embedding_ids[0] = 0
      expected_embedding_ids[1] = 0
      expected_embedding_ids[2] = 1
      expected_embedding_ids[8] = 0
      expected_embedding_ids[16] = 0
      expected_embedding_ids[17] = 2
      expected_embedding_ids[18] = 2
      expected_embedding_ids[24] = 0
      expected_embedding_ids[32] = 2
      expected_embedding_ids[33] = 2
      expected_embedding_ids[34] = 3
      expected_embedding_ids[40] = 0
      expected_embedding_ids[41] = 0
      expected_embedding_ids[48] = 0
      expected_embedding_ids[56] = 3
      expected_embedding_ids[128] = 0
      expected_embedding_ids[129] = 0
      expected_embedding_ids[136] = 0
      expected_embedding_ids[137] = 2
      expected_embedding_ids[144] = 0
      expected_embedding_ids[145] = 0
      expected_embedding_ids[146] = 1
      expected_embedding_ids[147] = 2
      expected_embedding_ids[152] = 0
      expected_embedding_ids[153] = 1
      expected_embedding_ids[154] = 2
      expected_embedding_ids[160] = 0
      expected_embedding_ids[161] = 0
      expected_embedding_ids[168] = 0
      expected_embedding_ids[169] = 0
      expected_embedding_ids[170] = 1
      expected_embedding_ids[256] = 5
      expected_embedding_ids[264] = 4
      expected_embedding_ids[265] = 5
      expected_embedding_ids[272] = 4
      expected_embedding_ids[273] = 5
      expected_embedding_ids[280] = 4
      expected_embedding_ids[281] = 5
      expected_embedding_ids[384] = 4
      expected_embedding_ids[385] = 5
      expected_embedding_ids[392] = 4
      expected_embedding_ids[393] = 5
      expected_embedding_ids[400] = 4
      expected_embedding_ids[401] = 5
      expected_embedding_ids[408] = 4
      expected_embedding_ids[409] = 5
      expected_embedding_ids[512] = 0
      expected_embedding_ids[513] = 0
      expected_embedding_ids[514] = 1
      expected_embedding_ids[520] = 0
      expected_embedding_ids[528] = 2
      expected_embedding_ids[536] = 0
      expected_embedding_ids[537] = 1
      expected_embedding_ids[544] = 1
      expected_embedding_ids[545] = 2
      expected_embedding_ids[546] = 2
      expected_embedding_ids[547] = 3
      expected_embedding_ids[552] = 0
      expected_embedding_ids[560] = 0
      expected_embedding_ids[561] = 0
      expected_embedding_ids[568] = 0
      expected_embedding_ids[569] = 1
      expected_embedding_ids[640] = 0
      expected_embedding_ids[641] = 0
      expected_embedding_ids[642] = 1
      expected_embedding_ids[648] = 0
      expected_embedding_ids[656] = 0
      expected_embedding_ids[657] = 2
      expected_embedding_ids[658] = 2
      expected_embedding_ids[664] = 0
      expected_embedding_ids[672] = 2
      expected_embedding_ids[673] = 2
      expected_embedding_ids[674] = 3
      expected_embedding_ids[680] = 0
      expected_embedding_ids[681] = 0
      expected_embedding_ids[688] = 0
      expected_embedding_ids[696] = 3
      expected_embedding_ids[768] = 5
      expected_embedding_ids[776] = 5
      expected_embedding_ids[784] = 5
      expected_embedding_ids[792] = 4
      expected_embedding_ids[800] = 4
      expected_embedding_ids[808] = 4
      expected_embedding_ids[816] = 4
      expected_embedding_ids[824] = 4
      expected_embedding_ids[896] = 5
      expected_embedding_ids[904] = 5
      expected_embedding_ids[912] = 5
      expected_embedding_ids[920] = 5
      expected_embedding_ids[928] = 4
      expected_embedding_ids[936] = 4
      expected_embedding_ids[944] = 4
      expected_embedding_ids[952] = 4
      if has_leading_dimension:
        embedding_ids["one_table_to_rule_them_all"] = embedding_ids[
            "one_table_to_rule_them_all"
        ].reshape(-1)
      np.testing.assert_equal(
          embedding_ids["one_table_to_rule_them_all"],
          expected_embedding_ids,
      )

    with self.subTest("sample_ids"):
      expected_sample_ids = np.full((1024,), 2**31 - 1, dtype=np.int32)
      expected_sample_ids[0] = 0
      expected_sample_ids[1] = 1
      expected_sample_ids[2] = 1
      expected_sample_ids[8] = 1
      expected_sample_ids[16] = 0
      expected_sample_ids[17] = 0
      expected_sample_ids[18] = 1
      expected_sample_ids[24] = 0
      expected_sample_ids[32] = 0
      expected_sample_ids[33] = 1
      expected_sample_ids[34] = 1
      expected_sample_ids[40] = 0
      expected_sample_ids[41] = 1
      expected_sample_ids[48] = 1
      expected_sample_ids[56] = 0
      expected_sample_ids[128] = 0
      expected_sample_ids[129] = 1
      expected_sample_ids[136] = 1
      expected_sample_ids[137] = 1
      expected_sample_ids[144] = 0
      expected_sample_ids[145] = 1
      expected_sample_ids[146] = 0
      expected_sample_ids[147] = 1
      expected_sample_ids[152] = 1
      expected_sample_ids[153] = 0
      expected_sample_ids[154] = 0
      expected_sample_ids[160] = 0
      expected_sample_ids[161] = 1
      expected_sample_ids[168] = 0
      expected_sample_ids[169] = 1
      expected_sample_ids[170] = 0

      expected_sample_ids[256] = 0
      expected_sample_ids[264] = 0
      expected_sample_ids[265] = 1
      expected_sample_ids[272] = 0
      expected_sample_ids[273] = 1
      expected_sample_ids[280] = 0
      expected_sample_ids[281] = 1
      expected_sample_ids[384] = 0
      expected_sample_ids[385] = 1
      expected_sample_ids[392] = 0
      expected_sample_ids[393] = 1
      expected_sample_ids[400] = 0
      expected_sample_ids[401] = 1
      expected_sample_ids[408] = 0
      expected_sample_ids[409] = 1

      expected_sample_ids[512] = 0
      expected_sample_ids[513] = 1
      expected_sample_ids[514] = 0
      expected_sample_ids[520] = 0
      expected_sample_ids[528] = 0
      expected_sample_ids[536] = 1
      expected_sample_ids[537] = 1
      expected_sample_ids[544] = 1
      expected_sample_ids[545] = 0
      expected_sample_ids[546] = 1
      expected_sample_ids[547] = 0
      expected_sample_ids[552] = 0
      expected_sample_ids[560] = 0
      expected_sample_ids[561] = 1
      expected_sample_ids[568] = 1
      expected_sample_ids[569] = 1
      expected_sample_ids[640] = 0
      expected_sample_ids[641] = 1
      expected_sample_ids[642] = 1
      expected_sample_ids[648] = 1
      expected_sample_ids[656] = 0
      expected_sample_ids[657] = 0
      expected_sample_ids[658] = 1
      expected_sample_ids[664] = 0
      expected_sample_ids[672] = 0
      expected_sample_ids[673] = 1
      expected_sample_ids[674] = 1
      expected_sample_ids[680] = 0
      expected_sample_ids[681] = 1
      expected_sample_ids[688] = 1
      expected_sample_ids[696] = 0
      expected_sample_ids[768] = 1
      expected_sample_ids[776] = 1
      expected_sample_ids[784] = 1
      expected_sample_ids[792] = 0
      expected_sample_ids[800] = 0
      expected_sample_ids[808] = 0
      expected_sample_ids[816] = 0
      expected_sample_ids[824] = 1
      expected_sample_ids[896] = 1
      expected_sample_ids[904] = 1
      expected_sample_ids[912] = 1
      expected_sample_ids[920] = 1
      expected_sample_ids[928] = 0
      expected_sample_ids[936] = 0
      expected_sample_ids[944] = 0
      expected_sample_ids[952] = 0
      if has_leading_dimension:
        sample_ids["one_table_to_rule_them_all"] = sample_ids[
            "one_table_to_rule_them_all"
        ].reshape(-1)
      np.testing.assert_equal(
          sample_ids["one_table_to_rule_them_all"],
          expected_sample_ids,
      )
    with self.subTest("gains"):
      expected_gains = np.full((1024,), np.nan, dtype=np.float32)
      expected_gains[0] = 2.0
      expected_gains[1] = 1.0
      expected_gains[2] = 1.0
      expected_gains[8] = 1.0
      expected_gains[16] = 1.0
      expected_gains[17] = 1.0
      expected_gains[18] = 1.0
      expected_gains[24] = 1.0
      expected_gains[32] = 1.0
      expected_gains[33] = 1.0
      expected_gains[34] = 1.0
      expected_gains[40] = 1.0
      expected_gains[41] = 1.0
      expected_gains[48] = 1.0
      expected_gains[56] = 1.0
      expected_gains[128] = 1.0
      expected_gains[129] = 1.0
      expected_gains[136] = 1.0
      expected_gains[137] = 1.0
      expected_gains[144] = 1.0
      expected_gains[145] = 1.0
      expected_gains[146] = 1.0
      expected_gains[147] = 1.0
      expected_gains[152] = 1.0
      expected_gains[153] = 1.0
      expected_gains[154] = 1.0
      expected_gains[160] = 1.0
      expected_gains[161] = 1.0
      expected_gains[168] = 1.0
      expected_gains[169] = 1.0
      expected_gains[170] = 1.0
      expected_gains[256] = 1.0
      expected_gains[264] = 1.0
      expected_gains[265] = 1.0
      expected_gains[272] = 1.0
      expected_gains[273] = 1.0
      expected_gains[280] = 1.0
      expected_gains[281] = 2.0
      expected_gains[384] = 1.0
      expected_gains[385] = 1.0
      expected_gains[392] = 1.0
      expected_gains[393] = 1.0
      expected_gains[400] = 1.0
      expected_gains[401] = 1.0
      expected_gains[408] = 1.0
      expected_gains[409] = 1.0
      expected_gains[512] = 1.0
      expected_gains[513] = 1.0
      expected_gains[514] = 1.0
      expected_gains[520] = 1.0
      expected_gains[528] = 1.0
      expected_gains[536] = 1.0
      expected_gains[537] = 1.0
      expected_gains[544] = 1.0
      expected_gains[545] = 1.0
      expected_gains[546] = 1.0
      expected_gains[547] = 1.0
      expected_gains[552] = 1.0
      expected_gains[560] = 1.0
      expected_gains[561] = 1.0
      expected_gains[568] = 1.0
      expected_gains[569] = 1.0
      expected_gains[640] = 2.0
      expected_gains[641] = 1.0
      expected_gains[642] = 1.0
      expected_gains[648] = 1.0
      expected_gains[656] = 1.0
      expected_gains[657] = 1.0
      expected_gains[658] = 1.0
      expected_gains[664] = 1.0
      expected_gains[672] = 1.0
      expected_gains[673] = 1.0
      expected_gains[674] = 1.0
      expected_gains[680] = 1.0
      expected_gains[681] = 1.0
      expected_gains[688] = 1.0
      expected_gains[696] = 1.0
      expected_gains[768] = 1.0
      expected_gains[776] = 1.0
      expected_gains[784] = 1.0
      expected_gains[792] = 1.0
      expected_gains[800] = 1.0
      expected_gains[808] = 1.0
      expected_gains[816] = 1.0
      expected_gains[824] = 1.0
      expected_gains[896] = 1.0
      expected_gains[904] = 1.0
      expected_gains[912] = 1.0
      expected_gains[920] = 1.0
      expected_gains[928] = 1.0
      expected_gains[936] = 1.0
      expected_gains[944] = 1.0
      expected_gains[952] = 1.0
      if has_leading_dimension:
        gains["one_table_to_rule_them_all"] = gains[
            "one_table_to_rule_them_all"
        ].reshape(-1)
      np.testing.assert_equal(
          gains["one_table_to_rule_them_all"],
          expected_gains,
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
    batch_number = 42
    row_pointers, embedding_ids, sample_ids, gains, _ = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
            batch_number,
            [self.singleton_input_features],
            [self.singleton_input_weights],
            [self.singleton_input_feature_spec],
            local_device_count=1,
            global_device_count=1,
            num_sc_per_device=4,
            sharding_strategy=ShardingStrategy.Mod,
            has_leading_dimension=False,
            allow_id_dropping=False,
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
    np.testing.assert_equal(
        sample_ids[self.singleton_input_table_spec.name],
        expected_lhs_local_sample_ids,
    )
    np.testing.assert_equal(
        embedding_ids[self.singleton_input_table_spec.name],
        expected_lhs_local_embedding_ids,
    )
    np.testing.assert_equal(
        gains[self.singleton_input_table_spec.name], expected_lhs_gains
    )

  @parameterized.parameters(False, True)
  def test_correct_input_preprocessing_multiple_columns(
      self, has_leading_dimension
  ):
    batch_number = 42
    row_pointers, embedding_ids, sample_ids, gains, _ = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
            batch_number,
            [self.input_features],
            [self.input_weights],
            [self.feature_spec],
            local_device_count=1,
            global_device_count=1,
            num_sc_per_device=4,
            sharding_strategy=ShardingStrategy.Mod,
            has_leading_dimension=has_leading_dimension,
            allow_id_dropping=False,
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
      np.testing.assert_equal(
          embedding_ids[self.table_spec.name], expected_lhs_local_embedding_ids
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
      np.testing.assert_equal(
          sample_ids[self.table_spec.name], expected_lhs_local_sample_ids
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
      np.testing.assert_equal(gains[self.table_spec.name], expected_lhs_gains)

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
    _, _, _, gains, _ = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
            batch_number,
            [self.input_features],
            [self.input_weights],
            [feature_spec],
            local_device_count=1,
            global_device_count=1,
            num_sc_per_device=4,
            sharding_strategy=ShardingStrategy.Mod,
            has_leading_dimension=has_leading_dimension,
            allow_id_dropping=False,
        )
    )

    coo_buffer_size = 32 * 4 * 4

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
      np.testing.assert_equal(
          gains["one_table_to_rule_them_all"], expected_lhs_gains
      )

  def test_correct_input_preprocessing_multiple_features_two_local_four_global_devices(
      self,
  ):
    # Outputs with leading dimension (pmap)
    batch_number = 42
    row_pointers, embedding_ids, sample_ids, gains, _ = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
            batch_number,
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
            has_leading_dimension=True,
            allow_id_dropping=False,
        )
    )
    # outputs without leading dimension (jit)
    batch_number = 42
    (
        row_pointers_flattened,
        embedding_ids_flattened,
        sample_ids_flattened,
        gains_flattened,
        _,
    ) = pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
        batch_number,
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

      np.testing.assert_equal(
          embedding_ids[self.singleton_input_table_spec.name],
          expected_singleton_table_embedding_ids,
      )
      np.testing.assert_equal(
          embedding_ids[self.ragged_input_table_spec.name],
          expected_ragged_table_embedding_ids,
      )
      np.testing.assert_equal(
          embedding_ids_flattened[self.singleton_input_table_spec.name],
          np.ravel(expected_singleton_table_embedding_ids),
      )
      np.testing.assert_equal(
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
      np.testing.assert_equal(
          sample_ids[self.singleton_input_table_spec.name],
          expected_singleton_table_sample_ids,
      )
      np.testing.assert_equal(
          sample_ids[self.ragged_input_table_spec.name],
          expected_ragged_table_sample_ids,
      )
      np.testing.assert_equal(
          sample_ids_flattened[self.singleton_input_table_spec.name],
          np.ravel(expected_singleton_table_sample_ids),
      )
      np.testing.assert_equal(
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

      np.testing.assert_equal(
          gains[self.singleton_input_table_spec.name],
          expected_singleton_table_gains,
      )
      np.testing.assert_equal(
          gains[self.ragged_input_table_spec.name],
          expected_ragged_table_gains,
      )
      np.testing.assert_equal(
          gains_flattened[self.singleton_input_table_spec.name],
          np.ravel(expected_singleton_table_gains),
      )
      np.testing.assert_equal(
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
    row_pointers, embedding_ids, sample_ids, gains, _ = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
            batch_number,
            [input_features],
            [input_weights],
            [feature_spec],
            local_device_count=1,
            global_device_count=1,
            num_sc_per_device=4,
            sharding_strategy=ShardingStrategy.Mod,
            has_leading_dimension=False,
            allow_id_dropping=False,
        )
    )

    # Compute expected by re-adjusting the weights using a "sum" combiner.
    table_spec.combiner = "sum"
    table_spec.stacked_table_spec = None
    embedding.prepare_feature_specs_for_training(
        feature_spec,
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
        _,
    ) = pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
        batch_number,
        [input_features],
        [input_weights],
        [feature_spec],
        local_device_count=1,
        global_device_count=1,
        num_sc_per_device=4,
        sharding_strategy=ShardingStrategy.Mod,
        has_leading_dimension=False,
        allow_id_dropping=False,
    )

    np.testing.assert_array_equal(
        row_pointers["table"], expected_row_pointers["table"]
    )
    np.testing.assert_array_equal(
        embedding_ids["table"], expected_embedding_ids["table"]
    )
    np.testing.assert_array_equal(
        sample_ids["table"], expected_sample_ids["table"]
    )
    np.testing.assert_allclose(
        gains["table"], expected_gains["table"], rtol=1e-5
    )


if __name__ == "__main__":
  absltest.main()
