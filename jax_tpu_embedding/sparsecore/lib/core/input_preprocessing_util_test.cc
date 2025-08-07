// Copyright 2024 The JAX SC Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"

#include <array>
#include <bitset>
#include <climits>
#include <cmath>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"  // from @com_google_absl
#include "Eigen/Core"  // from @eigen_archive
#include "jax_tpu_embedding/sparsecore/lib/core/coo_format.h"
#include "jax_tpu_embedding/sparsecore/lib/core/partitioned_coo_tensors.h"

namespace jax_sc_embedding {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::IsNan;
using ::testing::NanSensitiveFloatEq;
using ::testing::Pair;
using ::testing::Pointwise;

TEST(InputPreprocessingUtilTest, ColIds) {
  // int GetColId(int col_id, int col_shift, int col_offset, int num_scs_mod,
  //          int num_scs_mod_inv);
  EXPECT_EQ(
      CooFormat::GetColId(/*col_id=*/2, /*col_shift=*/4, /*col_offset=*/32,
                          /*num_scs_mod=*/3),
      34);
  EXPECT_EQ(
      CooFormat::GetColId(/*col_id=*/38, /*col_shift=*/0, /*col_offset=*/0,
                          /*num_scs_mod=*/3),
      38);
  EXPECT_EQ(
      CooFormat::GetColId(/*col_id=*/10, /*col_shift=*/0, /*col_offset=*/32,
                          /*num_scs_mod=*/7),
      42);
  EXPECT_EQ(
      CooFormat::GetColId(/*col_id=*/26, /*col_shift=*/0, /*col_offset=*/0,
                          /*num_scs_mod=*/3),
      26);
}

TEST(InputPreprocessingUtilTest, CeilOfRatio) {
  EXPECT_EQ(CeilOfRatio(/*numerator=*/1, /*denominator=*/1), 1);
  EXPECT_EQ(CeilOfRatio(/*numerator=*/4, /*denominator=*/2), 2);
  EXPECT_EQ(CeilOfRatio(/*numerator=*/3, /*denominator=*/2), 2);
  EXPECT_EQ(CeilOfRatio(/*numerator=*/10, /*denominator=*/4), 3);
  EXPECT_EQ(CeilOfRatio(/*numerator=*/0, /*denominator=*/3), 0);
}

TEST(InputPreprocessingUtilTest, MaxIdsPerPartitionForStackedTables) {
  std::vector<StackedTableMetadata> stacked_table_metadata;
  stacked_table_metadata.push_back(StackedTableMetadata(
      /*name=*/"table_0",
      /*feature_index=*/0, /*max_ids_per_partition=*/16,
      /*max_unique_ids_per_partition=*/16, /*row_offset=*/0,
      /*col_offset=*/0, /*col_shift=*/0, /*batch_size=*/8));
  stacked_table_metadata.push_back(StackedTableMetadata(
      /*name=*/"table_1",
      /*feature_index=*/1, /*max_ids_per_partition=*/16,
      /*max_unique_ids_per_partition=*/16, /*row_offset=*/0,
      /*col_offset=*/0, /*col_shift=*/0, /*batch_size=*/8));
  EXPECT_EQ(MaxIdsPerPartitionForStackedTables(stacked_table_metadata), 16);
}

TEST(InputPreprocessingUtilTest, ComputeCooBufferSize) {
  // Default case.
  std::vector<StackedTableMetadata> stacked_table_metadata;
  stacked_table_metadata.push_back(StackedTableMetadata(
      /*name=*/"table_0",
      /*feature_index=*/0, /*max_ids_per_partition=*/16,
      /*max_unique_ids_per_partition=*/16, /*row_offset=*/0,
      /*col_offset=*/0, /*col_shift=*/0, /*batch_size=*/8));
  stacked_table_metadata.push_back(StackedTableMetadata(
      /*name=*/"table_1",
      /*feature_index=*/1, /*max_ids_per_partition=*/16,
      /*max_unique_ids_per_partition=*/16, /*row_offset=*/0,
      /*col_offset=*/0, /*col_shift=*/0, /*batch_size=*/16));
  stacked_table_metadata.push_back(StackedTableMetadata(
      /*name=*/"table_2",
      /*feature_index=*/2, /*max_ids_per_partition=*/16,
      /*max_unique_ids_per_partition=*/16, /*row_offset=*/0,
      /*col_offset=*/0, /*col_shift=*/0, /*batch_size=*/24));
  EXPECT_EQ(ComputeCooBufferSizePerDevice(/*num_scs=*/4,
                                          /*num_scs_per_device=*/4,
                                          stacked_table_metadata),
            16 * 4 * 4);
  stacked_table_metadata[0].suggested_coo_buffer_size_per_device = 48;
  EXPECT_EQ(ComputeCooBufferSizePerDevice(/*num_scs=*/4,
                                          /*num_scs_per_device=*/4,
                                          stacked_table_metadata),
            64);

  stacked_table_metadata[0].suggested_coo_buffer_size_per_device = 96;
  EXPECT_EQ(ComputeCooBufferSizePerDevice(/*num_scs=*/4,
                                          /*num_scs_per_device=*/4,
                                          stacked_table_metadata),
            96);
}

TEST(InputPreprocessingUtilTest, IncrementScId) {
  std::pair<int, int> sc_id = {0, 0};
  IncrementScId(sc_id, /*num_scs=*/4, /*num_scs_per_device=*/2);
  EXPECT_THAT(sc_id, Pair(0, 1));
  IncrementScId(sc_id, /*num_scs=*/4, /*num_scs_per_device=*/2);
  EXPECT_THAT(sc_id, Pair(0, 2));
  IncrementScId(sc_id, /*num_scs=*/4, /*num_scs_per_device=*/2);
  EXPECT_THAT(sc_id, Pair(0, 3));
  IncrementScId(sc_id, /*num_scs=*/4, /*num_scs_per_device=*/2);
  EXPECT_THAT(sc_id, Pair(1, 0));
}

TEST(InputPreprocessingUtilTest, SortAndGroup) {
  std::vector<CooFormat> coo_formats;

  for (int row = 0; row < 8; ++row) {
    coo_formats.push_back(CooFormat(row, 0, 1.0));
    coo_formats.push_back(CooFormat(row, 1, 1.0));
    coo_formats.push_back(CooFormat(row, 2, 1.0));
    coo_formats.push_back(CooFormat(row, 3, 1.0));
  }
  Eigen::VectorXi max_id_per_sc{{0, 0, 0, 0}};
  Eigen::VectorXi max_unique_id_per_sc{{0, 0, 0, 0}};
  Eigen::VectorXi required_buffer_sizes_per_sc{{0, 0, 0, 0}};
  ExtractedCooTensors extracted_coo_tensors(4, 8);
  extracted_coo_tensors.coo_tensors = coo_formats;
  StackedTableMetadata stacked_table_metadata(
      "stacked_table", /*feature_index=*/0, /*max_ids_per_partition=*/32,
      /*max_unique_ids_per_partition=*/32, /*row_offset=*/0, /*col_offset=*/0,
      /*col_shift=*/0, /*batch_size=*/0);
  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 4,
      .global_device_count = 1,
      .num_sc_per_device = 4,
      .allow_id_dropping = false,
  };
  MinibatchingSplit minibatching_split = 0;
  PartitionedCooTensors coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          extracted_coo_tensors, stacked_table_metadata, options, max_id_per_sc,
          max_unique_id_per_sc, required_buffer_sizes_per_sc,
          minibatching_split);

  std::vector<CooFormat> expected_sc_0;
  expected_sc_0.push_back(CooFormat(0, 0, 1.0));
  expected_sc_0.push_back(CooFormat(1, 0, 1.0));
  expected_sc_0.push_back(CooFormat(0, 1, 1.0));
  expected_sc_0.push_back(CooFormat(1, 1, 1.0));
  expected_sc_0.push_back(CooFormat(0, 2, 1.0));
  expected_sc_0.push_back(CooFormat(1, 2, 1.0));
  expected_sc_0.push_back(CooFormat(0, 3, 1.0));
  expected_sc_0.push_back(CooFormat(1, 3, 1.0));
  expected_sc_0.push_back(CooFormat(2, 0, 0.0));

  std::vector<CooFormat> expected_sc_1;
  expected_sc_1.push_back(CooFormat(2, 0, 1.0));
  expected_sc_1.push_back(CooFormat(3, 0, 1.0));
  expected_sc_1.push_back(CooFormat(2, 1, 1.0));
  expected_sc_1.push_back(CooFormat(3, 1, 1.0));
  expected_sc_1.push_back(CooFormat(2, 2, 1.0));
  expected_sc_1.push_back(CooFormat(3, 2, 1.0));
  expected_sc_1.push_back(CooFormat(2, 3, 1.0));
  expected_sc_1.push_back(CooFormat(3, 3, 1.0));
  expected_sc_1.push_back(CooFormat(4, 0, 0.0));

  std::vector<CooFormat> expected_sc_2;
  expected_sc_2.push_back(CooFormat(4, 0, 1.0));
  expected_sc_2.push_back(CooFormat(5, 0, 1.0));
  expected_sc_2.push_back(CooFormat(4, 1, 1.0));
  expected_sc_2.push_back(CooFormat(5, 1, 1.0));
  expected_sc_2.push_back(CooFormat(4, 2, 1.0));
  expected_sc_2.push_back(CooFormat(5, 2, 1.0));
  expected_sc_2.push_back(CooFormat(4, 3, 1.0));
  expected_sc_2.push_back(CooFormat(5, 3, 1.0));
  expected_sc_2.push_back(CooFormat(6, 0, 0.0));

  std::vector<CooFormat> expected_sc_3;
  expected_sc_3.push_back(CooFormat(6, 0, 1.0));
  expected_sc_3.push_back(CooFormat(7, 0, 1.0));
  expected_sc_3.push_back(CooFormat(6, 1, 1.0));
  expected_sc_3.push_back(CooFormat(7, 1, 1.0));
  expected_sc_3.push_back(CooFormat(6, 2, 1.0));
  expected_sc_3.push_back(CooFormat(7, 2, 1.0));
  expected_sc_3.push_back(CooFormat(6, 3, 1.0));
  expected_sc_3.push_back(CooFormat(7, 3, 1.0));
  expected_sc_3.push_back(CooFormat(8, 0, 0.0));

  EXPECT_THAT(coo_tensors_by_id(/*local_sc_id=*/0, /*bucket_id=*/0),
              ElementsAreArray(expected_sc_0));
  EXPECT_THAT(coo_tensors_by_id(/*local_sc_id=*/1, /*bucket_id=*/0),
              ElementsAreArray(expected_sc_1));
  EXPECT_THAT(coo_tensors_by_id(/*local_sc_id=*/2, /*bucket_id=*/0),
              ElementsAreArray(expected_sc_2));
  EXPECT_THAT(coo_tensors_by_id(/*local_sc_id=*/3, /*bucket_id=*/0),
              ElementsAreArray(expected_sc_3));
  EXPECT_THAT(max_id_per_sc, ElementsAreArray({2, 2, 2, 2}));
  EXPECT_THAT(max_unique_id_per_sc, ElementsAreArray({1, 1, 1, 1}));
  EXPECT_THAT(required_buffer_sizes_per_sc, ElementsAreArray({33, 33, 33, 33}));
}

TEST(InputPreprocessingUtilTest, SortAndGroup_TwoScs) {
  std::vector<CooFormat> coo_formats;

  for (int row = 0; row < 8; ++row) {
    coo_formats.push_back(CooFormat(row, 0, 1.0));
    coo_formats.push_back(CooFormat(row, 1, 1.0));
    coo_formats.push_back(CooFormat(row, 2, 1.0));
    coo_formats.push_back(CooFormat(row, 3, 1.0));
  }
  Eigen::VectorXi max_id_per_sc{{0, 0}};
  Eigen::VectorXi max_unique_id_per_sc{{0, 0}};
  Eigen::VectorXi required_buffer_sizes_per_sc{{0, 0}};
  ExtractedCooTensors extracted_coo_tensors(2, 8);
  extracted_coo_tensors.coo_tensors = coo_formats;
  StackedTableMetadata stacked_table_metadata(
      "stacked_table", /*feature_index=*/0, /*max_ids_per_partition=*/32,
      /*max_unique_ids_per_partition=*/32, /*row_offset=*/0, /*col_offset=*/0,
      /*col_shift=*/0, /*batch_size=*/0);
  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 2,
      .global_device_count = 1,
      .num_sc_per_device = 2,
      .allow_id_dropping = false,
  };
  MinibatchingSplit minibatching_split = 0;
  PartitionedCooTensors coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          extracted_coo_tensors, stacked_table_metadata, options, max_id_per_sc,
          max_unique_id_per_sc, required_buffer_sizes_per_sc,
          minibatching_split);

  EXPECT_THAT(
      coo_tensors_by_id(/*local_sc_id=*/0, /*bucket_id=*/0),
      ElementsAre(
          CooFormat(0, 0, 1.0), CooFormat(1, 0, 1.0), CooFormat(2, 0, 1.0),
          CooFormat(3, 0, 1.0), CooFormat(0, 2, 1.0), CooFormat(1, 2, 1.0),
          CooFormat(2, 2, 1.0), CooFormat(3, 2, 1.0), CooFormat(0, 1, 1.0),
          CooFormat(1, 1, 1.0), CooFormat(2, 1, 1.0), CooFormat(3, 1, 1.0),
          CooFormat(0, 3, 1.0), CooFormat(1, 3, 1.0), CooFormat(2, 3, 1.0),
          CooFormat(3, 3, 1.0), CooFormat(4, 0, 0.0)));
  EXPECT_THAT(
      coo_tensors_by_id(/*local_sc_id=*/1, /*bucket_id=*/0),
      ElementsAre(
          CooFormat(4, 0, 1.0), CooFormat(5, 0, 1.0), CooFormat(6, 0, 1.0),
          CooFormat(7, 0, 1.0), CooFormat(4, 2, 1.0), CooFormat(5, 2, 1.0),
          CooFormat(6, 2, 1.0), CooFormat(7, 2, 1.0), CooFormat(4, 1, 1.0),
          CooFormat(5, 1, 1.0), CooFormat(6, 1, 1.0), CooFormat(7, 1, 1.0),
          CooFormat(4, 3, 1.0), CooFormat(5, 3, 1.0), CooFormat(6, 3, 1.0),
          CooFormat(7, 3, 1.0), CooFormat(8, 0, 0.0)));
  EXPECT_THAT(max_id_per_sc, ElementsAreArray({8, 8}));
  EXPECT_THAT(max_unique_id_per_sc, ElementsAreArray({2, 2}));
  EXPECT_THAT(required_buffer_sizes_per_sc, ElementsAreArray({17, 17}));
}

TEST(InputPreprocessingUtilTest, SortAndGroup_VerifyIdLimitations1) {
  std::vector<CooFormat> coo_formats;

  // With 8 samples, each sample has 4 ids [0, 1, 2, 3]
  // Each sparsecore serves 1 row of data.
  // Each sparsecore looks up for 2 samples. For each sample, requesting
  // one row of data from one sparsecore.
  // [max_ids_per_partition == 2]
  // For each sparsecore, it receives at most "2" rows of data from any one
  // peer sparsecore.
  // [max_unique_ids_per_partition == 1]
  // For each sparsecore, it receives the data of at most "1" row ID from each
  // sparsecore.
  for (int row = 0; row < 8; ++row) {
    coo_formats.push_back(CooFormat(row, 0, 1.0));
    coo_formats.push_back(CooFormat(row, 1, 1.0));
    coo_formats.push_back(CooFormat(row, 2, 1.0));
    coo_formats.push_back(CooFormat(row, 3, 1.0));
  }
  Eigen::VectorXi max_id_per_sc{{0, 0, 0, 0}};
  Eigen::VectorXi max_unique_id_per_sc{{0, 0, 0, 0}};
  Eigen::VectorXi required_buffer_sizes_per_sc{{0, 0, 0, 0}};
  ExtractedCooTensors extracted_coo_tensors(4, 8);
  extracted_coo_tensors.coo_tensors = coo_formats;
  StackedTableMetadata stacked_table_metadata(
      "stacked_table", /*feature_index=*/0, /*max_ids_per_partition=*/2,
      /*max_unique_ids_per_partition=*/1, /*row_offset=*/0, /*col_offset=*/0,
      /*col_shift=*/0, /*batch_size=*/0);
  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 4,
      .global_device_count = 1,
      .num_sc_per_device = 4,
      .allow_id_dropping = false,
  };
  MinibatchingSplit minibatching_split = 0;
  PartitionedCooTensors coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          extracted_coo_tensors, stacked_table_metadata, options, max_id_per_sc,
          max_unique_id_per_sc, required_buffer_sizes_per_sc,
          minibatching_split);
  EXPECT_THAT(max_id_per_sc, ElementsAreArray({2, 2, 2, 2}));
  EXPECT_THAT(max_unique_id_per_sc, ElementsAreArray({1, 1, 1, 1}));
  EXPECT_THAT(required_buffer_sizes_per_sc, ElementsAreArray({33, 33, 33, 33}));
}

TEST(InputPreprocessingUtilTest, SortAndGroup_VerifyIdLimitations2) {
  std::vector<CooFormat> coo_formats;

  // With 16 samples, each sample has 4 ids [0, 1, 2, 3]
  // Each sparsecore serves 1 row of data.
  // Each sparsecore looks up for 4 samples. For each sample, requesting
  // one row of data from one sparsecore.
  // [max_ids_per_partition == 4]
  // For each sparsecore, it receives at most "4" rows of data from any one
  // peer sparsecore.
  // [max_unique_ids_per_partition == 1]
  // For each sparsecore, it receives the data of at most "1" row ID from each
  // sparsecore.
  for (int row = 0; row < 16; ++row) {
    coo_formats.push_back(CooFormat(row, 0, 1.0));
    coo_formats.push_back(CooFormat(row, 1, 1.0));
    coo_formats.push_back(CooFormat(row, 2, 1.0));
    coo_formats.push_back(CooFormat(row, 3, 1.0));
  }
  Eigen::VectorXi max_id_per_sc{{0, 0, 0, 0}};
  Eigen::VectorXi max_unique_id_per_sc{{0, 0, 0, 0}};
  Eigen::VectorXi required_buffer_sizes_per_sc{{0, 0, 0, 0}};
  ExtractedCooTensors extracted_coo_tensors(4, 16);
  extracted_coo_tensors.coo_tensors = coo_formats;
  StackedTableMetadata stacked_table_metadata(
      "stacked_table", /*feature_index=*/0, /*max_ids_per_partition=*/4,
      /*max_unique_ids_per_partition=*/1, /*row_offset=*/0, /*col_offset=*/0,
      /*col_shift=*/0, /*batch_size=*/0);
  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 4,
      .global_device_count = 1,
      .num_sc_per_device = 4,
      .allow_id_dropping = false,
  };
  MinibatchingSplit minibatching_split = 0;
  PartitionedCooTensors coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          extracted_coo_tensors, stacked_table_metadata, options, max_id_per_sc,
          max_unique_id_per_sc, required_buffer_sizes_per_sc,
          minibatching_split);
  EXPECT_THAT(max_id_per_sc, ElementsAreArray({4, 4, 4, 4}));
  EXPECT_THAT(max_unique_id_per_sc, ElementsAreArray({1, 1, 1, 1}));
  EXPECT_THAT(required_buffer_sizes_per_sc, ElementsAreArray({33, 33, 33, 33}));
}

TEST(InputPreprocessingUtilTest, SortAndGroup_VerifyIdLimitations3) {
  std::vector<CooFormat> coo_formats;

  // With 16 samples, each sample has 8 ids [0, 1, 2, 3, 4, 5, 6, 7]
  // Each sparsecore serves 2 rows of data [0, 4], [1, 5], [2, 6], [3, 7]
  // Each sparsecore looks up for 4 samples. For each sample, requesting
  // two rows of data from one sparsecore. [0, 4] from sparsecore 0, [1, 5]
  // from sparsecore 1, [2, 6] from sparsecore 2, [3, 7] from sparsecore 3.
  // [max_ids_per_partition == 8]
  // For each sparsecore, it receives at most "8" rows of data from any one
  // peer sparsecore.
  // [max_unique_ids_per_partition == 2]
  // For each sparsecore, it receives the data of at most "2" row IDs from each
  // sparsecore.
  for (int row = 0; row < 16; ++row) {
    coo_formats.push_back(CooFormat(row, 0, 1.0));
    coo_formats.push_back(CooFormat(row, 1, 1.0));
    coo_formats.push_back(CooFormat(row, 2, 1.0));
    coo_formats.push_back(CooFormat(row, 3, 1.0));
    coo_formats.push_back(CooFormat(row, 4, 1.0));
    coo_formats.push_back(CooFormat(row, 5, 1.0));
    coo_formats.push_back(CooFormat(row, 6, 1.0));
    coo_formats.push_back(CooFormat(row, 7, 1.0));
  }
  Eigen::VectorXi max_id_per_sc{{0, 0, 0, 0}};
  Eigen::VectorXi max_unique_id_per_sc{{0, 0, 0, 0}};
  Eigen::VectorXi required_buffer_sizes_per_sc{{0, 0, 0, 0}};
  ExtractedCooTensors extracted_coo_tensors(4, 16);
  extracted_coo_tensors.coo_tensors = coo_formats;
  StackedTableMetadata stacked_table_metadata(
      "stacked_table", /*feature_index=*/0, /*max_ids_per_partition=*/8,
      /*max_unique_ids_per_partition=*/2, /*row_offset=*/0, /*col_offset=*/0,
      /*col_shift=*/0, /*batch_size=*/0);
  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 4,
      .global_device_count = 1,
      .num_sc_per_device = 4,
      .allow_id_dropping = false,
  };
  MinibatchingSplit minibatching_split = 0;
  PartitionedCooTensors coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          extracted_coo_tensors, stacked_table_metadata, options, max_id_per_sc,
          max_unique_id_per_sc, required_buffer_sizes_per_sc,
          minibatching_split);
  EXPECT_THAT(max_id_per_sc, ElementsAreArray({8, 8, 8, 8}));
  EXPECT_THAT(max_unique_id_per_sc, ElementsAreArray({2, 2, 2, 2}));
  // 4 partitions of size 8 with 2 elements each
  EXPECT_THAT(required_buffer_sizes_per_sc, ElementsAreArray({33, 33, 33, 33}));
}

TEST(InputPreprocessingUtilTest, SortAndGroup_VerifyIdLimitations4) {
  std::vector<CooFormat> coo_formats;

  // With 128 samples, each sample has 8 ids [0, 1, 2, 3, 4, 5, 6, 7]
  // Each sparsecore serves 2 rows of data [0, 4], [1, 5], [2, 6], [3, 7]
  // Each sparsecore looks up for 32 samples. For each sample, requesting
  // two rows of data from one sparsecore. [0, 4] from sparsecore 0, [1, 5]
  // from sparsecore 1, [2, 6] from sparsecore 2, [3, 7] from sparsecore 3.
  // [max_ids_per_partition == 64]
  // For each sparsecore, it receives at most "64" rows of data from any one
  // peer sparsecore.
  // [max_unique_ids_per_partition == 2]
  // For each sparsecore, it receives the data of at most "2" row IDs from each
  // sparsecore.
  for (int row = 0; row < 128; ++row) {
    coo_formats.push_back(CooFormat(row, 0, 1.0));
    coo_formats.push_back(CooFormat(row, 1, 1.0));
    coo_formats.push_back(CooFormat(row, 2, 1.0));
    coo_formats.push_back(CooFormat(row, 3, 1.0));
    coo_formats.push_back(CooFormat(row, 4, 1.0));
    coo_formats.push_back(CooFormat(row, 5, 1.0));
    coo_formats.push_back(CooFormat(row, 6, 1.0));
    coo_formats.push_back(CooFormat(row, 7, 1.0));
  }
  Eigen::VectorXi max_id_per_sc{{0, 0, 0, 0}};
  Eigen::VectorXi max_unique_id_per_sc{{0, 0, 0, 0}};
  Eigen::VectorXi required_buffer_sizes_per_sc{{0, 0, 0, 0}};
  ExtractedCooTensors extracted_coo_tensors(4, 128);
  extracted_coo_tensors.coo_tensors = coo_formats;
  StackedTableMetadata stacked_table_metadata(
      "stacked_table", /*feature_index=*/0, /*max_ids_per_partition=*/64,
      /*max_unique_ids_per_partition=*/2, /*row_offset=*/0, /*col_offset=*/0,
      /*col_shift=*/0, /*batch_size=*/0);
  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 4,
      .global_device_count = 1,
      .num_sc_per_device = 4,
      .allow_id_dropping = false,
  };
  MinibatchingSplit minibatching_split = 0;
  PartitionedCooTensors coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          extracted_coo_tensors, stacked_table_metadata, options, max_id_per_sc,
          max_unique_id_per_sc, required_buffer_sizes_per_sc,
          minibatching_split);
  EXPECT_THAT(max_id_per_sc, ElementsAreArray({64, 64, 64, 64}));
  EXPECT_THAT(max_unique_id_per_sc, ElementsAreArray({2, 2, 2, 2}));
  // 8 partitions of size 256 with 32 elements each
  EXPECT_THAT(required_buffer_sizes_per_sc,
              ElementsAreArray({257, 257, 257, 257}));
}

TEST(InputPreprocessingUtilTest, SortAndGroup_VerifyIdLimitations5) {
  std::vector<CooFormat> coo_formats;

  // With 128 samples, each sample has 8 ids [0, 4, 8, 16]
  // SparseCore 0 alone serves all 4 rows of data [0, 4, 8, 16]
  // Each sparsecore looks up for 32 samples. For each sample, requesting
  // all 4 rows of data from sparsecore 0.
  // [max_ids_per_partition == 128]
  // For each sparsecore, it receives at most "128" rows of data from any one
  // peer sparsecore. (In this case, only sparsecore 0 has the data)
  // [max_unique_ids_per_partition == 4]
  // For each sparsecore, it receives the data of at most "4" row IDs from each
  // sparsecore.
  for (int row = 0; row < 128; ++row) {
    coo_formats.push_back(CooFormat(row, 0, 1.0));
    coo_formats.push_back(CooFormat(row, 4, 1.0));
    coo_formats.push_back(CooFormat(row, 8, 1.0));
    coo_formats.push_back(CooFormat(row, 16, 1.0));
  }
  Eigen::VectorXi max_id_per_sc{{0, 0, 0, 0}};
  Eigen::VectorXi max_unique_id_per_sc{{0, 0, 0, 0}};
  Eigen::VectorXi required_buffer_sizes_per_sc{{0, 0, 0, 0}};
  ExtractedCooTensors extracted_coo_tensors(4, 128);
  extracted_coo_tensors.coo_tensors = coo_formats;
  StackedTableMetadata stacked_table_metadata(
      "stacked_table", /*feature_index=*/0, /*max_ids_per_partition=*/128,
      /*max_unique_ids_per_partition=*/4, /*row_offset=*/0, /*col_offset=*/0,
      /*col_shift=*/0, /*batch_size=*/0);
  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 4,
      .global_device_count = 1,
      .num_sc_per_device = 4,
      .allow_id_dropping = false,
  };
  MinibatchingSplit minibatching_split = 0;
  PartitionedCooTensors coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          extracted_coo_tensors, stacked_table_metadata, options, max_id_per_sc,
          max_unique_id_per_sc, required_buffer_sizes_per_sc,
          minibatching_split);
  EXPECT_THAT(max_id_per_sc, ElementsAreArray({128, 0, 0, 0}));
  EXPECT_THAT(max_unique_id_per_sc, ElementsAreArray({4, 0, 0, 0}));
  // 1 partition of size 128 with 128 elements
  EXPECT_THAT(required_buffer_sizes_per_sc,
              ElementsAreArray({129, 129, 129, 129}));
}

TEST(InputPreprocessingUtilTest, SortAndGroup_VerifyIdLimitations6) {
  std::vector<CooFormat> coo_formats;

  // This is one of the worst case scenarios.
  // Every ID is unique, and all IDs come from the same sparsecore.
  //
  // With 128 samples, each sample has 1 id [row * 4]
  // SparseCore 0 alone serves all 128 rows of data [0, 4, 8, ...]
  // Each sparsecore looks up for 32 samples. For each sample, requesting
  // the single row of data from sparsecore 0.
  // [max_ids_per_partition == 32]
  // For each sparsecore, it receives at most "32" rows of data from any one
  // peer sparsecore. (In this case, only sparsecore 0 has the data)
  // [max_unique_ids_per_partition == 32]
  // For each sparsecore, it receives the data of at most "32" row IDs from each
  // sparsecore.
  for (int row = 0; row < 128; ++row) {
    coo_formats.push_back(CooFormat(row, row * 4, 1.0));
  }
  Eigen::VectorXi max_id_per_sc{{0, 0, 0, 0}};
  Eigen::VectorXi max_unique_id_per_sc{{0, 0, 0, 0}};
  Eigen::VectorXi required_buffer_sizes_per_sc{{0, 0, 0, 0}};
  ExtractedCooTensors extracted_coo_tensors(4, 128);
  extracted_coo_tensors.coo_tensors = coo_formats;
  StackedTableMetadata stacked_table_metadata(
      "stacked_table", /*feature_index=*/0, /*max_ids_per_partition=*/32,
      /*max_unique_ids_per_partition=*/32, /*row_offset=*/0, /*col_offset=*/0,
      /*col_shift=*/0, /*batch_size=*/0);
  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 4,
      .global_device_count = 1,
      .num_sc_per_device = 4,
      .allow_id_dropping = false,
  };
  MinibatchingSplit minibatching_split = 0;
  PartitionedCooTensors coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          extracted_coo_tensors, stacked_table_metadata, options, max_id_per_sc,
          max_unique_id_per_sc, required_buffer_sizes_per_sc,
          minibatching_split);
  EXPECT_THAT(max_id_per_sc, ElementsAreArray({32, 0, 0, 0}));
  EXPECT_THAT(max_unique_id_per_sc, ElementsAreArray({32, 0, 0, 0}));
  // 1 partition of size 32 with 32 elements
  EXPECT_THAT(required_buffer_sizes_per_sc, ElementsAreArray({33, 33, 33, 33}));
}

TEST(InputPreprocessingUtilTest, SortAndGroup_IdDropping) {
  std::vector<CooFormat> coo_formats;

  // With 16 samples, each sample has 4 ids [0, 1, 2, 3]
  // Each sparsecore serves 1 row of data.
  // Each sparsecore looks up for 4 samples. For each sample, requesting
  // one row of data from one sparsecore.
  // [max_ids_per_partition == 4]
  // For each sparsecore, it receives at most "4" rows of data from any one
  // peer sparsecore.
  // [max_unique_ids_per_partition == 1]
  // For each sparsecore, it receives the data of at most "1" row ID from each
  // sparsecore.
  for (int row = 0; row < 16; ++row) {
    coo_formats.push_back(CooFormat(row, 0, 1.0));
    coo_formats.push_back(CooFormat(row, 1, 1.0));
    coo_formats.push_back(CooFormat(row, 2, 1.0));
    coo_formats.push_back(CooFormat(row, 3, 1.0));
  }
  // Force dropping of IDs here with max_ids_per_partition == 2
  // The later 2 samples for each sparsecore will be dropped.
  Eigen::VectorXi max_id_per_sc{{0, 0, 0, 0}};
  Eigen::VectorXi max_unique_id_per_sc{{0, 0, 0, 0}};
  Eigen::VectorXi required_buffer_sizes_per_sc{{0, 0, 0, 0}};
  ExtractedCooTensors extracted_coo_tensors(4, 16);
  extracted_coo_tensors.coo_tensors = coo_formats;
  StackedTableMetadata stacked_table_metadata(
      "stacked_table", /*feature_index=*/0, /*max_ids_per_partition=*/2,
      /*max_unique_ids_per_partition=*/1, /*row_offset=*/0, /*col_offset=*/0,
      /*col_shift=*/0, /*batch_size=*/0);
  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 4,
      .global_device_count = 1,
      .num_sc_per_device = 4,
      .allow_id_dropping = true,
  };
  MinibatchingSplit minibatching_split = 0;
  PartitionedCooTensors coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          extracted_coo_tensors, stacked_table_metadata, options, max_id_per_sc,
          max_unique_id_per_sc, required_buffer_sizes_per_sc,
          minibatching_split);
  EXPECT_THAT(max_id_per_sc, ElementsAreArray({4, 4, 4, 4}));
  EXPECT_THAT(max_unique_id_per_sc, ElementsAreArray({1, 1, 1, 1}));
  // 4 partition of size 8 with 4 element each
  EXPECT_THAT(required_buffer_sizes_per_sc, ElementsAreArray({33, 33, 33, 33}));

  // Note that sample 2, 3, 6, 7, 10, 11, 14, 15 are dropped.
  // It's unclear how embedding activations will be constructed without these
  // samples at this unit test level.
  std::vector<CooFormat> expected_sc_0;
  expected_sc_0.push_back(CooFormat(0, 0, 1.0));
  expected_sc_0.push_back(CooFormat(1, 0, 1.0));
  expected_sc_0.push_back(CooFormat(0, 1, 1.0));
  expected_sc_0.push_back(CooFormat(1, 1, 1.0));
  expected_sc_0.push_back(CooFormat(0, 2, 1.0));
  expected_sc_0.push_back(CooFormat(1, 2, 1.0));
  expected_sc_0.push_back(CooFormat(0, 3, 1.0));
  expected_sc_0.push_back(CooFormat(1, 3, 1.0));
  expected_sc_0.push_back(CooFormat(4, 0, 0.0));

  std::vector<CooFormat> expected_sc_1;
  expected_sc_1.push_back(CooFormat(4, 0, 1.0));
  expected_sc_1.push_back(CooFormat(5, 0, 1.0));
  expected_sc_1.push_back(CooFormat(4, 1, 1.0));
  expected_sc_1.push_back(CooFormat(5, 1, 1.0));
  expected_sc_1.push_back(CooFormat(4, 2, 1.0));
  expected_sc_1.push_back(CooFormat(5, 2, 1.0));
  expected_sc_1.push_back(CooFormat(4, 3, 1.0));
  expected_sc_1.push_back(CooFormat(5, 3, 1.0));
  expected_sc_1.push_back(CooFormat(8, 0, 0.0));

  std::vector<CooFormat> expected_sc_2;
  expected_sc_2.push_back(CooFormat(8, 0, 1.0));
  expected_sc_2.push_back(CooFormat(9, 0, 1.0));
  expected_sc_2.push_back(CooFormat(8, 1, 1.0));
  expected_sc_2.push_back(CooFormat(9, 1, 1.0));
  expected_sc_2.push_back(CooFormat(8, 2, 1.0));
  expected_sc_2.push_back(CooFormat(9, 2, 1.0));
  expected_sc_2.push_back(CooFormat(8, 3, 1.0));
  expected_sc_2.push_back(CooFormat(9, 3, 1.0));
  expected_sc_2.push_back(CooFormat(12, 0, 0.0));

  std::vector<CooFormat> expected_sc_3;
  expected_sc_3.push_back(CooFormat(12, 0, 1.0));
  expected_sc_3.push_back(CooFormat(13, 0, 1.0));
  expected_sc_3.push_back(CooFormat(12, 1, 1.0));
  expected_sc_3.push_back(CooFormat(13, 1, 1.0));
  expected_sc_3.push_back(CooFormat(12, 2, 1.0));
  expected_sc_3.push_back(CooFormat(13, 2, 1.0));
  expected_sc_3.push_back(CooFormat(12, 3, 1.0));
  expected_sc_3.push_back(CooFormat(13, 3, 1.0));
  expected_sc_3.push_back(CooFormat(16, 0, 0.0));

  EXPECT_THAT(coo_tensors_by_id(/*local_sc_id=*/0, /*bucket_id=*/0),
              ElementsAreArray(expected_sc_0));
  EXPECT_THAT(coo_tensors_by_id(/*local_sc_id=*/1, /*bucket_id=*/0),
              ElementsAreArray(expected_sc_1));
  EXPECT_THAT(coo_tensors_by_id(/*local_sc_id=*/2, /*bucket_id=*/0),
              ElementsAreArray(expected_sc_2));
  EXPECT_THAT(coo_tensors_by_id(/*local_sc_id=*/3, /*bucket_id=*/0),
              ElementsAreArray(expected_sc_3));
}

TEST(InputPreprocessingUtilTest, FillBuffer) {
  std::vector<CooFormat> coo_formats;

  for (int row = 0; row < 8; ++row) {
    coo_formats.push_back(CooFormat(row, 0, 1.0));
    coo_formats.push_back(CooFormat(row, 1, 1.0));
    coo_formats.push_back(CooFormat(row, 2, 1.0));
    coo_formats.push_back(CooFormat(row, 3, 1.0));
  }
  Eigen::VectorXi max_id_per_sc{{0, 0, 0, 0}};
  Eigen::VectorXi max_unique_id_per_sc{{0, 0, 0, 0}};
  Eigen::VectorXi required_buffer_sizes_per_sc{{0, 0, 0, 0}};
  ExtractedCooTensors extracted_coo_tensors(4, 8);
  extracted_coo_tensors.coo_tensors = coo_formats;
  StackedTableMetadata stacked_table_metadata(
      "stacked_table", /*feature_index=*/0, /*max_ids_per_partition=*/32,
      /*max_unique_ids_per_partition=*/32, /*row_offset=*/0, /*col_offset=*/0,
      /*col_shift=*/0, /*batch_size=*/0);
  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 4,
      .global_device_count = 1,
      .num_sc_per_device = 4,
      .allow_id_dropping = false,
  };
  MinibatchingSplit minibatching_split = 0;
  PartitionedCooTensors coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          extracted_coo_tensors, stacked_table_metadata, options, max_id_per_sc,
          max_unique_id_per_sc, required_buffer_sizes_per_sc,
          minibatching_split);

  Eigen::VectorXi row_pointers(8 * 4);
  Eigen::VectorXi embedding_ids(40 * 4);
  Eigen::VectorXi sample_ids(40 * 4);
  Eigen::VectorXf gains(40 * 4);
  FillLocalDeviceBuffer(coo_tensors_by_id,
                        /*row_pointers_size_per_sc=*/8,
                        /*coo_buffer_size_per_sc=*/40,
                        /*batch_size_per_sc=*/2, options, row_pointers,
                        embedding_ids, sample_ids, gains);

  std::array<int, 32> expected_row_pointers = {
      2, 10, 18, 26, 32, 32, 32, 32,  //
      2, 10, 18, 26, 32, 32, 32, 32,  //
      2, 10, 18, 26, 32, 32, 32, 32,  //
      2, 10, 18, 26, 32, 32, 32, 32,  //
  };
  EXPECT_THAT(row_pointers, ElementsAreArray(expected_row_pointers));

  std::array<int, 4 * 40> expected_embedding_ids = {
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
  };

  EXPECT_THAT(embedding_ids, ElementsAreArray(expected_embedding_ids));

  std::array<int, 4 * 40> expected_sample_ids = {
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
  };
  EXPECT_THAT(sample_ids, ElementsAreArray(expected_sample_ids));

  auto expected_gains = ElementsAre(
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),
      IsNan(),                                                     //
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),
      IsNan(),                                                     //
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),
      IsNan(),                                                     //
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),
      IsNan()  //
  );
  EXPECT_THAT(gains, expected_gains);
}

TEST(InputPreprocessingUtilTest, FillBufferMinibatchingSingleMinibatch) {
  std::vector<CooFormat> coo_formats;

  for (int row = 0; row < 8; ++row) {
    coo_formats.push_back(CooFormat(row, 0, 1.0));
    coo_formats.push_back(CooFormat(row, 1, 1.0));
    coo_formats.push_back(CooFormat(row, 2, 1.0));
    coo_formats.push_back(CooFormat(row, 3, 1.0));
  }
  Eigen::VectorXi max_id_per_sc{{0, 0, 0, 0}};
  Eigen::VectorXi max_unique_id_per_sc{{0, 0, 0, 0}};
  Eigen::VectorXi required_buffer_sizes_per_sc{{0, 0, 0, 0}};
  ExtractedCooTensors extracted_coo_tensors(4, 8);
  extracted_coo_tensors.coo_tensors = coo_formats;
  StackedTableMetadata stacked_table_metadata(
      "stacked_table", /*feature_index=*/0, /*max_ids_per_partition=*/32,
      /*max_unique_ids_per_partition=*/32, /*row_offset=*/0, /*col_offset=*/0,
      /*col_shift=*/0, /*batch_size=*/0);
  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 1,
      .global_device_count = 1,
      .num_sc_per_device = 4,
      .allow_id_dropping = false,
      .enable_minibatching = true,
  };
  MinibatchingSplit minibatching_split = 0;
  PartitionedCooTensors coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          extracted_coo_tensors, stacked_table_metadata, options, max_id_per_sc,
          max_unique_id_per_sc, required_buffer_sizes_per_sc,
          minibatching_split);

  coo_tensors_by_id.MergeAll();

  Eigen::VectorXi row_pointers(8 * 4);
  Eigen::VectorXi embedding_ids(40 * 4);
  Eigen::VectorXi sample_ids(40 * 4);
  Eigen::VectorXf gains(40 * 4);
  FillLocalDeviceBuffer(coo_tensors_by_id,
                        /*row_pointers_size_per_sc=*/8,
                        /*coo_buffer_size_per_sc=*/40,
                        /*batch_size_per_sc=*/2, options, row_pointers,
                        embedding_ids, sample_ids, gains);

  std::array<int, 32> expected_row_pointers = {
      2, 10, 18, 26, 32, 32, 32, 32,  //
      2, 10, 18, 26, 32, 32, 32, 32,  //
      2, 10, 18, 26, 32, 32, 32, 32,  //
      2, 10, 18, 26, 32, 32, 32, 32,  //
  };
  EXPECT_THAT(row_pointers, ElementsAreArray(expected_row_pointers));

  std::array<int, 4 * 40> expected_embedding_ids = {
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       0,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
  };

  EXPECT_THAT(embedding_ids, ElementsAreArray(expected_embedding_ids));

  std::array<int, 4 * 40> expected_sample_ids = {
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0,       1,       INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
  };
  EXPECT_THAT(sample_ids, ElementsAreArray(expected_sample_ids));

  auto expected_gains = ElementsAre(
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
      IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),
      IsNan(),  //
      IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),
      IsNan(),  //
      IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),
      IsNan(),  //
      IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),
      IsNan()  //
  );
  EXPECT_THAT(gains, expected_gains);
}

TEST(InputPreprocessingUtilTest, FillBufferMinibatchingFourMinibatches) {
  std::vector<CooFormat> coo_formats;

  for (int row = 0; row < 8; ++row) {
    for (int col = 0; col < 64; ++col) {
      coo_formats.push_back(CooFormat(row, col, 1.0));
    }
  }
  Eigen::VectorXi max_id_per_sc{{0, 0, 0, 0}};
  Eigen::VectorXi max_unique_id_per_sc{{0, 0, 0, 0}};
  Eigen::VectorXi required_buffer_sizes_per_sc{{0, 0, 0, 0}};
  ExtractedCooTensors extracted_coo_tensors(4, 8);
  extracted_coo_tensors.coo_tensors = coo_formats;
  StackedTableMetadata stacked_table_metadata(
      "stacked_table", /*feature_index=*/0, /*max_ids_per_partition=*/33,
      /*max_unique_ids_per_partition=*/33, /*row_offset=*/0, /*col_offset=*/0,
      /*col_shift=*/0, /*batch_size=*/0);
  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 1,
      .global_device_count = 1,
      .num_sc_per_device = 4,
      .allow_id_dropping = false,
      .enable_minibatching = true,
  };
  MinibatchingSplit minibatching_split = 0;
  PartitionedCooTensors coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          extracted_coo_tensors, stacked_table_metadata, options, max_id_per_sc,
          max_unique_id_per_sc, required_buffer_sizes_per_sc,
          minibatching_split);

  // 4 Minibatches of bucket sizes [16,8,24,16]
  // 62:                  32
  // 60, 61:          16      48
  // 56, 57, 58, 59: 8  24  40  56
  // Overwrite split.
  minibatching_split = 0;
  minibatching_split.set(61);
  minibatching_split.set(60);
  minibatching_split.set(57);
  coo_tensors_by_id.Merge(minibatching_split);
  EXPECT_EQ(coo_tensors_by_id.GetNumMinibatches(), 4);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < coo_tensors_by_id.GetNumMinibatches(); j++) {
      auto coo_tensors = coo_tensors_by_id(i, j);
      if (j == coo_tensors_by_id.GetNumMinibatches() - 1)
        coo_tensors.remove_suffix(1);  // Ignore the sentinel node.
      EXPECT_TRUE(absl::c_is_sorted(
          coo_tensors, [&](const CooFormat& coo1, const CooFormat& coo2) {
            // Sorted by global SC ID.
            return (coo1.col_id % 4) < (coo2.col_id % 4);
          }));
    }
  }

  // Each SC has ids 2 rows with ids [0,1,..63] each. Last partition will also
  // have the sentinel node.
  // 4 Minibatches of bucket sizes [16,8,24,16]
  EXPECT_EQ(coo_tensors_by_id(0, 0).size(), 16 * 2);
  EXPECT_EQ(coo_tensors_by_id(0, 1).size(), 8 * 2);
  EXPECT_EQ(coo_tensors_by_id(0, 2).size(), 24 * 2);
  EXPECT_EQ(coo_tensors_by_id(0, 3).size(), 16 * 2 + 1);

  const int coo_buffer_size_per_sc = 168;

  Eigen::VectorXi row_pointers(32 * 4);
  Eigen::VectorXi embedding_ids(coo_buffer_size_per_sc * 4);
  Eigen::VectorXi sample_ids(coo_buffer_size_per_sc * 4);
  Eigen::VectorXf gains(coo_buffer_size_per_sc * 4);

  FillLocalDeviceBuffer(coo_tensors_by_id,
                        /*row_pointers_size_per_bucket=*/8,
                        coo_buffer_size_per_sc,
                        /*batch_size_per_sc=*/2, options, row_pointers,
                        embedding_ids, sample_ids, gains);

  std::array<int, 128> expected_row_pointers = {
      8,  16, 24, 32, 32, 32, 32, 32,  // SC0 MB0
      4,  12, 20, 28, 28, 28, 28, 28,  // SC0 MB1
      12, 28, 44, 60, 60, 60, 60, 60,  // SC0 MB2
      8,  16, 24, 32, 32, 32, 32, 32,  // SC0 MB3
      8,  16, 24, 32, 32, 32, 32, 32,  // SC1 MB0
      4,  12, 20, 28, 28, 28, 28, 28,  // SC1 MB1
      12, 28, 44, 60, 60, 60, 60, 60,  // SC1 MB2
      8,  16, 24, 32, 32, 32, 32, 32,  // SC1 MB3
      8,  16, 24, 32, 32, 32, 32, 32,  // SC2 MB0
      4,  12, 20, 28, 28, 28, 28, 28,  // SC2 MB1
      12, 28, 44, 60, 60, 60, 60, 60,  // SC2 MB2
      8,  16, 24, 32, 32, 32, 32, 32,  // SC2 MB3
      8,  16, 24, 32, 32, 32, 32, 32,  // SC3 MB0
      4,  12, 20, 28, 28, 28, 28, 28,  // SC3 MB1
      12, 28, 44, 60, 60, 60, 60, 60,  // SC3 MB2
      8,  16, 24, 32, 32, 32, 32, 32,  // SC3 MB3
  };
  EXPECT_THAT(row_pointers, ElementsAreArray(expected_row_pointers));

  RowVectorXi expected_embedding_ids =
      RowVectorXi::Constant(coo_buffer_size_per_sc * 4, INT_MAX);
  RowVectorXi expected_sample_ids =
      RowVectorXi::Constant(coo_buffer_size_per_sc * 4, INT_MAX);
  RowVectorXf expected_gains =
      RowVectorXf::Constant(coo_buffer_size_per_sc * 4, std::nanf(""));
  // Sum of padded minibatch buffer sizes per SC: 32 (MB0) + 32 (MB1) + 64 (MB2)
  // + 32 (MB3) = 160.
  const int minibatches_buffer_size_per_sc = 160;
  for (int sc = 0; sc < 4; ++sc) {
    // MB0
    expected_embedding_ids.segment(sc * minibatches_buffer_size_per_sc, 32)
        << (RowVectorXi(8) << 0, 0, 1, 1, 2, 2, 3, 3)
               .finished()
               .replicate(1, 4);
    expected_sample_ids.segment(sc * minibatches_buffer_size_per_sc, 32)
        << (RowVectorXi(2) << 0, 1).finished().replicate(1, 16);
    expected_gains.segment(sc * minibatches_buffer_size_per_sc, 32).setOnes();
    // MB1
    expected_embedding_ids.segment(sc * minibatches_buffer_size_per_sc + 32, 32)
        << (RowVectorXi(8) << 4, 4, 5, 5, INT_MAX, INT_MAX, INT_MAX, INT_MAX)
               .finished()
               .replicate(1, 4);
    expected_sample_ids.segment(sc * minibatches_buffer_size_per_sc + 32, 32)
        << (RowVectorXi(8) << 0, 1, 0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX)
               .finished()
               .replicate(1, 4);
    expected_gains.segment(sc * minibatches_buffer_size_per_sc + 32, 32)
        << (RowVectorXf(8) << 1, 1, 1, 1, std::nanf(""), std::nanf(""),
            std::nanf(""), std::nanf(""))
               .finished()
               .replicate(1, 4);
    // MB2
    expected_embedding_ids.segment(sc * minibatches_buffer_size_per_sc + 64, 64)
        << (RowVectorXi(16) << 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, INT_MAX,
            INT_MAX, INT_MAX, INT_MAX)
               .finished()
               .replicate(1, 4);
    expected_sample_ids.segment(sc * minibatches_buffer_size_per_sc + 64, 64)
        << (RowVectorXi(16) << 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, INT_MAX,
            INT_MAX, INT_MAX, INT_MAX)
               .finished()
               .replicate(1, 4);
    expected_gains.segment(sc * minibatches_buffer_size_per_sc + 64, 64)
        << (RowVectorXf(16) << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            std::nanf(""), std::nanf(""), std::nanf(""), std::nanf(""))
               .finished()
               .replicate(1, 4);
    // MB3
    expected_embedding_ids.segment(sc * minibatches_buffer_size_per_sc + 128,
                                   32)
        << (RowVectorXi(8) << 12, 12, 13, 13, 14, 14, 15, 15)
               .finished()
               .replicate(1, 4);
    expected_sample_ids.segment(sc * minibatches_buffer_size_per_sc + 128, 32)
        << (RowVectorXi(2) << 0, 1).finished().replicate(1, 16);
    expected_gains.segment(sc * minibatches_buffer_size_per_sc + 128, 32)
        .setOnes();
  }

  EXPECT_THAT(embedding_ids, ElementsAreArray(expected_embedding_ids));
  EXPECT_THAT(sample_ids, ElementsAreArray(expected_sample_ids));
  EXPECT_THAT(gains, Pointwise(NanSensitiveFloatEq(), expected_gains));
}

}  // namespace
}  // namespace jax_sc_embedding
