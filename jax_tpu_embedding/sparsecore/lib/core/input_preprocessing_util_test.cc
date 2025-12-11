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
#include <functional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "fuzztest/fuzztest.h"
#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "Eigen/Core"  // from @eigen_archive
#include "jax_tpu_embedding/sparsecore/lib/core/coo_format.h"
#include "jax_tpu_embedding/sparsecore/lib/core/partitioned_coo_tensors.h"
#include "jax_tpu_embedding/sparsecore/lib/core/sort_and_group_coo_tensors_impl.h"

namespace jax_sc_embedding {
namespace {

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::IsNan;
using ::testing::NanSensitiveFloatEq;
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

void GetColIdIsCorrect(int embedding_id, int col_shift,
                       int col_offset_per_shard, int num_scs_bit) {
  const int num_scs = 1 << num_scs_bit;
  const int num_scs_mod = num_scs - 1;
  const int col_offset = col_offset_per_shard * num_scs;
  int col_id =
      CooFormat::GetColId(embedding_id, col_shift, col_offset, num_scs_mod);
  // Partition ID is shifted by col_shift.
  EXPECT_EQ(col_id % num_scs, (embedding_id + col_shift) % num_scs);
  // Local embedding ID is shifted by col_offset.
  EXPECT_EQ(col_id / num_scs, (embedding_id + col_offset) / num_scs);
}

FUZZ_TEST(InputPreprocessingUtilTest, GetColIdIsCorrect)
    .WithDomains(/*embedding_id=*/fuzztest::InRange(0, 100000000),
                 /*col_shift=*/fuzztest::InRange(0, 64),
                 /*col_offset_per_shard=*/fuzztest::InRange(0, 600000),
                 /*num_scs_bit=*/fuzztest::InRange(0, 9));

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
  stacked_table_metadata[0].suggested_coo_buffer_size_per_device = 1024;
  // The theoretical max is 16 * 4 * 4 = 256. This is less than the suggestion.
  EXPECT_DEATH(ComputeCooBufferSizePerDevice(/*num_scs=*/4,
                                             /*num_scs_per_device=*/4,
                                             stacked_table_metadata),
               ".*Check failed: suggested_value <= theoretical_max.*");
}

TEST(SortAndGroupTest, Base) {
  std::vector<CooFormat> coo_formats;

  for (int row = 0; row < 8; ++row) {
    coo_formats.push_back(CooFormat(row, 0, 1.0));
    coo_formats.push_back(CooFormat(row, 1, 1.0));
    coo_formats.push_back(CooFormat(row, 2, 1.0));
    coo_formats.push_back(CooFormat(row, 3, 1.0));
  }
  ExtractedCooTensors extracted_coo_tensors(4, 8, coo_formats);
  StackedTableMetadata stacked_table_metadata(
      "stacked_table", /*feature_index=*/0, /*max_ids_per_partition=*/32,
      /*max_unique_ids_per_partition=*/32, /*row_offset=*/0, /*col_offset=*/0,
      /*col_shift=*/0, /*batch_size=*/0);
  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 4,
      .global_device_count = 1,
      .num_sc_per_device = 4,
      .allow_id_dropping = false,
      .feature_stacking_strategy = FeatureStackingStrategy::kStackThenSplit,
  };
  MinibatchingSplit minibatching_split = 0;
  StatsPerHost stats_per_host(/*local_device_count=*/1, /*num_partitions=*/4,
                              /*num_sc_per_device=*/4);
  internal::StatsPerDevice stats_per_device =
      stats_per_host.GetStatsPerDevice(0);
  PartitionedCooTensors coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          extracted_coo_tensors, stacked_table_metadata, options,
          stats_per_device, minibatching_split);

  std::vector<CooFormat> expected_sc_0;
  expected_sc_0.push_back(CooFormat(0, 0, 1.0));
  expected_sc_0.push_back(CooFormat(1, 0, 1.0));
  expected_sc_0.push_back(CooFormat(0, 1, 1.0));
  expected_sc_0.push_back(CooFormat(1, 1, 1.0));
  expected_sc_0.push_back(CooFormat(0, 2, 1.0));
  expected_sc_0.push_back(CooFormat(1, 2, 1.0));
  expected_sc_0.push_back(CooFormat(0, 3, 1.0));
  expected_sc_0.push_back(CooFormat(1, 3, 1.0));

  std::vector<CooFormat> expected_sc_1;
  expected_sc_1.push_back(CooFormat(2, 0, 1.0));
  expected_sc_1.push_back(CooFormat(3, 0, 1.0));
  expected_sc_1.push_back(CooFormat(2, 1, 1.0));
  expected_sc_1.push_back(CooFormat(3, 1, 1.0));
  expected_sc_1.push_back(CooFormat(2, 2, 1.0));
  expected_sc_1.push_back(CooFormat(3, 2, 1.0));
  expected_sc_1.push_back(CooFormat(2, 3, 1.0));
  expected_sc_1.push_back(CooFormat(3, 3, 1.0));

  std::vector<CooFormat> expected_sc_2;
  expected_sc_2.push_back(CooFormat(4, 0, 1.0));
  expected_sc_2.push_back(CooFormat(5, 0, 1.0));
  expected_sc_2.push_back(CooFormat(4, 1, 1.0));
  expected_sc_2.push_back(CooFormat(5, 1, 1.0));
  expected_sc_2.push_back(CooFormat(4, 2, 1.0));
  expected_sc_2.push_back(CooFormat(5, 2, 1.0));
  expected_sc_2.push_back(CooFormat(4, 3, 1.0));
  expected_sc_2.push_back(CooFormat(5, 3, 1.0));

  std::vector<CooFormat> expected_sc_3;
  expected_sc_3.push_back(CooFormat(6, 0, 1.0));
  expected_sc_3.push_back(CooFormat(7, 0, 1.0));
  expected_sc_3.push_back(CooFormat(6, 1, 1.0));
  expected_sc_3.push_back(CooFormat(7, 1, 1.0));
  expected_sc_3.push_back(CooFormat(6, 2, 1.0));
  expected_sc_3.push_back(CooFormat(7, 2, 1.0));
  expected_sc_3.push_back(CooFormat(6, 3, 1.0));
  expected_sc_3.push_back(CooFormat(7, 3, 1.0));

  EXPECT_THAT(coo_tensors_by_id(/*local_sc_id=*/0, /*bucket_id=*/0),
              ElementsAreArray(expected_sc_0));
  EXPECT_THAT(coo_tensors_by_id(/*local_sc_id=*/1, /*bucket_id=*/0),
              ElementsAreArray(expected_sc_1));
  EXPECT_THAT(coo_tensors_by_id(/*local_sc_id=*/2, /*bucket_id=*/0),
              ElementsAreArray(expected_sc_2));
  EXPECT_THAT(coo_tensors_by_id(/*local_sc_id=*/3, /*bucket_id=*/0),
              ElementsAreArray(expected_sc_3));
  EXPECT_EQ(stats_per_device.dropped_id_count, 0);
  EXPECT_THAT(stats_per_device.max_ids_per_partition,
              ElementsAreArray({2, 2, 2, 2}));
  EXPECT_THAT(stats_per_device.max_unique_ids_per_partition,
              ElementsAreArray({1, 1, 1, 1}));
  EXPECT_THAT(stats_per_device.required_buffer_size,
              ElementsAreArray({32, 32, 32, 32}));
}

TEST(SortAndGroupTest, TwoScs) {
  std::vector<CooFormat> coo_formats;

  for (int row = 0; row < 8; ++row) {
    coo_formats.push_back(CooFormat(row, 0, 1.0));
    coo_formats.push_back(CooFormat(row, 1, 1.0));
    coo_formats.push_back(CooFormat(row, 2, 1.0));
    coo_formats.push_back(CooFormat(row, 3, 1.0));
  }
  ExtractedCooTensors extracted_coo_tensors(2, 8, coo_formats);
  StackedTableMetadata stacked_table_metadata(
      "stacked_table", /*feature_index=*/0, /*max_ids_per_partition=*/32,
      /*max_unique_ids_per_partition=*/32, /*row_offset=*/0, /*col_offset=*/0,
      /*col_shift=*/0, /*batch_size=*/0);
  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 2,
      .global_device_count = 1,
      .num_sc_per_device = 2,
      .allow_id_dropping = false,
      .feature_stacking_strategy = FeatureStackingStrategy::kStackThenSplit,
  };
  MinibatchingSplit minibatching_split = 0;
  StatsPerHost stats_per_host(/*local_device_count=*/1, /*num_partitions=*/2,
                              /*num_sc_per_device=*/2);
  internal::StatsPerDevice stats_per_device =
      stats_per_host.GetStatsPerDevice(0);
  PartitionedCooTensors coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          extracted_coo_tensors, stacked_table_metadata, options,
          stats_per_device, minibatching_split);

  EXPECT_EQ(minibatching_split, 0);

  EXPECT_THAT(coo_tensors_by_id(/*local_sc_id=*/0, /*bucket_id=*/0),
              ElementsAre(CooFormat(0, 0, 1.0), CooFormat(1, 0, 1.0),
                          CooFormat(2, 0, 1.0), CooFormat(3, 0, 1.0),
                          CooFormat(0, 2, 1.0), CooFormat(1, 2, 1.0),
                          CooFormat(2, 2, 1.0), CooFormat(3, 2, 1.0),
                          CooFormat(0, 1, 1.0), CooFormat(1, 1, 1.0),
                          CooFormat(2, 1, 1.0), CooFormat(3, 1, 1.0),
                          CooFormat(0, 3, 1.0), CooFormat(1, 3, 1.0),
                          CooFormat(2, 3, 1.0), CooFormat(3, 3, 1.0)));
  EXPECT_THAT(coo_tensors_by_id(/*local_sc_id=*/1, /*bucket_id=*/0),
              ElementsAre(CooFormat(4, 0, 1.0), CooFormat(5, 0, 1.0),
                          CooFormat(6, 0, 1.0), CooFormat(7, 0, 1.0),
                          CooFormat(4, 2, 1.0), CooFormat(5, 2, 1.0),
                          CooFormat(6, 2, 1.0), CooFormat(7, 2, 1.0),
                          CooFormat(4, 1, 1.0), CooFormat(5, 1, 1.0),
                          CooFormat(6, 1, 1.0), CooFormat(7, 1, 1.0),
                          CooFormat(4, 3, 1.0), CooFormat(5, 3, 1.0),
                          CooFormat(6, 3, 1.0), CooFormat(7, 3, 1.0)));
  EXPECT_EQ(stats_per_device.dropped_id_count, 0);
  EXPECT_THAT(stats_per_device.max_ids_per_partition, ElementsAreArray({8, 8}));
  EXPECT_THAT(stats_per_device.max_unique_ids_per_partition,
              ElementsAreArray({2, 2}));
  EXPECT_THAT(stats_per_device.required_buffer_size,
              ElementsAreArray({16, 16}));
}

TEST(SortAndGroupTest, VerifyIdLimitations1) {
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
  ExtractedCooTensors extracted_coo_tensors(4, 8, coo_formats);
  StackedTableMetadata stacked_table_metadata(
      "stacked_table", /*feature_index=*/0, /*max_ids_per_partition=*/2,
      /*max_unique_ids_per_partition=*/1, /*row_offset=*/0, /*col_offset=*/0,
      /*col_shift=*/0, /*batch_size=*/0);
  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 4,
      .global_device_count = 1,
      .num_sc_per_device = 4,
      .allow_id_dropping = false,
      .feature_stacking_strategy = FeatureStackingStrategy::kStackThenSplit,
  };
  MinibatchingSplit minibatching_split = 0;
  StatsPerHost stats_per_host(/*local_device_count=*/1, /*num_partitions=*/4,
                              /*num_sc_per_device=*/4);
  internal::StatsPerDevice stats_per_device =
      stats_per_host.GetStatsPerDevice(0);
  PartitionedCooTensors coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          extracted_coo_tensors, stacked_table_metadata, options,
          stats_per_device, minibatching_split);

  EXPECT_EQ(minibatching_split, 0);
  EXPECT_THAT(stats_per_device.dropped_id_count, 0);
  EXPECT_THAT(stats_per_device.max_ids_per_partition,
              ElementsAreArray({2, 2, 2, 2}));
  EXPECT_THAT(stats_per_device.max_unique_ids_per_partition,
              ElementsAreArray({1, 1, 1, 1}));
  EXPECT_THAT(stats_per_device.required_buffer_size,
              ElementsAreArray({32, 32, 32, 32}));
}

TEST(SortAndGroupTest, VerifyIdLimitations2) {
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
  ExtractedCooTensors extracted_coo_tensors(4, 16, coo_formats);
  StackedTableMetadata stacked_table_metadata(
      "stacked_table", /*feature_index=*/0, /*max_ids_per_partition=*/4,
      /*max_unique_ids_per_partition=*/1, /*row_offset=*/0, /*col_offset=*/0,
      /*col_shift=*/0, /*batch_size=*/0);
  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 4,
      .global_device_count = 1,
      .num_sc_per_device = 4,
      .allow_id_dropping = false,
      .feature_stacking_strategy = FeatureStackingStrategy::kStackThenSplit,
  };
  MinibatchingSplit minibatching_split = 0;
  StatsPerHost stats_per_host(/*local_device_count=*/1, /*num_partitions=*/4,
                              /*num_sc_per_device=*/4);
  internal::StatsPerDevice stats_per_device =
      stats_per_host.GetStatsPerDevice(0);
  PartitionedCooTensors coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          extracted_coo_tensors, stacked_table_metadata, options,
          stats_per_device, minibatching_split);

  EXPECT_EQ(minibatching_split, 0);
  EXPECT_THAT(stats_per_device.dropped_id_count, 0);
  EXPECT_THAT(stats_per_device.max_ids_per_partition,
              ElementsAreArray({4, 4, 4, 4}));
  EXPECT_THAT(stats_per_device.max_unique_ids_per_partition,
              ElementsAreArray({1, 1, 1, 1}));
  EXPECT_THAT(stats_per_device.required_buffer_size,
              ElementsAreArray({32, 32, 32, 32}));
}

TEST(SortAndGroupTest, VerifyIdLimitations3) {
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
  ExtractedCooTensors extracted_coo_tensors(4, 16, coo_formats);
  StackedTableMetadata stacked_table_metadata(
      "stacked_table", /*feature_index=*/0, /*max_ids_per_partition=*/8,
      /*max_unique_ids_per_partition=*/2, /*row_offset=*/0, /*col_offset=*/0,
      /*col_shift=*/0, /*batch_size=*/0);
  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 4,
      .global_device_count = 1,
      .num_sc_per_device = 4,
      .allow_id_dropping = false,
      .feature_stacking_strategy = FeatureStackingStrategy::kStackThenSplit,
  };
  MinibatchingSplit minibatching_split = 0;
  StatsPerHost stats_per_host(/*local_device_count=*/1, /*num_partitions=*/4,
                              /*num_sc_per_device=*/4);
  internal::StatsPerDevice stats_per_device =
      stats_per_host.GetStatsPerDevice(0);
  PartitionedCooTensors coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          extracted_coo_tensors, stacked_table_metadata, options,
          stats_per_device, minibatching_split);

  EXPECT_EQ(minibatching_split, 0);
  EXPECT_THAT(stats_per_device.dropped_id_count, 0);
  EXPECT_THAT(stats_per_device.max_ids_per_partition,
              ElementsAreArray({8, 8, 8, 8}));
  EXPECT_THAT(stats_per_device.max_unique_ids_per_partition,
              ElementsAreArray({2, 2, 2, 2}));
  // 4 partitions of size 8 with 2 elements each
  EXPECT_THAT(stats_per_device.required_buffer_size,
              ElementsAreArray({32, 32, 32, 32}));
}

TEST(SortAndGroupTest, VerifyIdLimitations4) {
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
  ExtractedCooTensors extracted_coo_tensors(4, 128, coo_formats);
  StackedTableMetadata stacked_table_metadata(
      "stacked_table", /*feature_index=*/0, /*max_ids_per_partition=*/64,
      /*max_unique_ids_per_partition=*/2, /*row_offset=*/0, /*col_offset=*/0,
      /*col_shift=*/0, /*batch_size=*/0);
  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 4,
      .global_device_count = 1,
      .num_sc_per_device = 4,
      .allow_id_dropping = false,
      .feature_stacking_strategy = FeatureStackingStrategy::kStackThenSplit,
  };
  MinibatchingSplit minibatching_split = 0;
  StatsPerHost stats_per_host(/*local_device_count=*/1, /*num_partitions=*/4,
                              /*num_sc_per_device=*/4);
  internal::StatsPerDevice stats_per_device =
      stats_per_host.GetStatsPerDevice(0);
  PartitionedCooTensors coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          extracted_coo_tensors, stacked_table_metadata, options,
          stats_per_device, minibatching_split);

  EXPECT_EQ(minibatching_split, 0);
  EXPECT_THAT(stats_per_device.dropped_id_count, 0);
  EXPECT_THAT(stats_per_device.max_ids_per_partition,
              ElementsAreArray({64, 64, 64, 64}));
  EXPECT_THAT(stats_per_device.max_unique_ids_per_partition,
              ElementsAreArray({2, 2, 2, 2}));
  // 8 partitions of size 256 with 32 elements each
  EXPECT_THAT(stats_per_device.required_buffer_size,
              ElementsAreArray({256, 256, 256, 256}));
}

TEST(SortAndGroupTest, VerifyIdLimitations5) {
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
  ExtractedCooTensors extracted_coo_tensors(4, 128, coo_formats);
  StackedTableMetadata stacked_table_metadata(
      "stacked_table", /*feature_index=*/0, /*max_ids_per_partition=*/128,
      /*max_unique_ids_per_partition=*/4, /*row_offset=*/0, /*col_offset=*/0,
      /*col_shift=*/0, /*batch_size=*/0);
  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 4,
      .global_device_count = 1,
      .num_sc_per_device = 4,
      .allow_id_dropping = false,
      .feature_stacking_strategy = FeatureStackingStrategy::kStackThenSplit,
  };
  MinibatchingSplit minibatching_split = 0;
  StatsPerHost stats_per_host(/*local_device_count=*/1, /*num_partitions=*/4,
                              /*num_sc_per_device=*/4);
  internal::StatsPerDevice stats_per_device =
      stats_per_host.GetStatsPerDevice(0);
  PartitionedCooTensors coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          extracted_coo_tensors, stacked_table_metadata, options,
          stats_per_device, minibatching_split);

  EXPECT_EQ(minibatching_split, 0);
  EXPECT_THAT(stats_per_device.dropped_id_count, 0);
  EXPECT_THAT(stats_per_device.max_ids_per_partition,
              ElementsAreArray({128, 0, 0, 0}));
  EXPECT_THAT(stats_per_device.max_unique_ids_per_partition,
              ElementsAreArray({4, 0, 0, 0}));
  // 1 partition of size 128 with 128 elements
  EXPECT_THAT(stats_per_device.required_buffer_size,
              ElementsAreArray({128, 128, 128, 128}));
}

TEST(SortAndGroupTest, VerifyIdLimitations6) {
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
  ExtractedCooTensors extracted_coo_tensors(4, 128, coo_formats);
  StackedTableMetadata stacked_table_metadata(
      "stacked_table", /*feature_index=*/0, /*max_ids_per_partition=*/32,
      /*max_unique_ids_per_partition=*/32, /*row_offset=*/0, /*col_offset=*/0,
      /*col_shift=*/0, /*batch_size=*/0);
  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 4,
      .global_device_count = 1,
      .num_sc_per_device = 4,
      .allow_id_dropping = false,
      .feature_stacking_strategy = FeatureStackingStrategy::kStackThenSplit,
  };
  MinibatchingSplit minibatching_split = 0;
  StatsPerHost stats_per_host(/*local_device_count=*/1, /*num_partitions=*/4,
                              /*num_sc_per_device=*/4);
  auto stats_per_device = stats_per_host.GetStatsPerDevice(0);
  PartitionedCooTensors coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          extracted_coo_tensors, stacked_table_metadata, options,
          stats_per_device, minibatching_split);

  EXPECT_EQ(minibatching_split, 0);
  EXPECT_THAT(stats_per_device.dropped_id_count, 0);
  EXPECT_THAT(stats_per_device.max_ids_per_partition,
              ElementsAreArray({32, 0, 0, 0}));
  EXPECT_THAT(stats_per_device.max_unique_ids_per_partition,
              ElementsAreArray({32, 0, 0, 0}));
  // 1 partition of size 32 with 32 elements
  EXPECT_THAT(stats_per_device.required_buffer_size,
              ElementsAreArray({32, 32, 32, 32}));
}

TEST(SortAndGroupTest, IdDropping) {
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
  ExtractedCooTensors extracted_coo_tensors(4, 16, coo_formats);
  StackedTableMetadata stacked_table_metadata(
      "stacked_table", /*feature_index=*/0, /*max_ids_per_partition=*/2,
      /*max_unique_ids_per_partition=*/1, /*row_offset=*/0, /*col_offset=*/0,
      /*col_shift=*/0, /*batch_size=*/0);
  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 4,
      .global_device_count = 1,
      .num_sc_per_device = 4,
      .allow_id_dropping = true,
      .feature_stacking_strategy = FeatureStackingStrategy::kStackThenSplit,
  };
  bool minibatching_split = 0;
  StatsPerHost stats_per_host(/*local_device_count=*/1, /*num_partitions=*/4,
                              /*num_sc_per_device=*/4);
  auto stats_per_device = stats_per_host.GetStatsPerDevice(0);
  PartitionedCooTensors coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          extracted_coo_tensors, stacked_table_metadata, options,
          stats_per_device, minibatching_split);
  EXPECT_THAT(stats_per_device.dropped_id_count, 32);
  EXPECT_THAT(stats_per_device.max_ids_per_partition,
              ElementsAreArray({4, 4, 4, 4}));
  EXPECT_THAT(stats_per_device.max_unique_ids_per_partition,
              ElementsAreArray({1, 1, 1, 1}));
  // 4 partition of size 8 with 4 element each
  EXPECT_THAT(stats_per_device.required_buffer_size,
              ElementsAreArray({32, 32, 32, 32}));

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

  std::vector<CooFormat> expected_sc_1;
  expected_sc_1.push_back(CooFormat(4, 0, 1.0));
  expected_sc_1.push_back(CooFormat(5, 0, 1.0));
  expected_sc_1.push_back(CooFormat(4, 1, 1.0));
  expected_sc_1.push_back(CooFormat(5, 1, 1.0));
  expected_sc_1.push_back(CooFormat(4, 2, 1.0));
  expected_sc_1.push_back(CooFormat(5, 2, 1.0));
  expected_sc_1.push_back(CooFormat(4, 3, 1.0));
  expected_sc_1.push_back(CooFormat(5, 3, 1.0));

  std::vector<CooFormat> expected_sc_2;
  expected_sc_2.push_back(CooFormat(8, 0, 1.0));
  expected_sc_2.push_back(CooFormat(9, 0, 1.0));
  expected_sc_2.push_back(CooFormat(8, 1, 1.0));
  expected_sc_2.push_back(CooFormat(9, 1, 1.0));
  expected_sc_2.push_back(CooFormat(8, 2, 1.0));
  expected_sc_2.push_back(CooFormat(9, 2, 1.0));
  expected_sc_2.push_back(CooFormat(8, 3, 1.0));
  expected_sc_2.push_back(CooFormat(9, 3, 1.0));

  std::vector<CooFormat> expected_sc_3;
  expected_sc_3.push_back(CooFormat(12, 0, 1.0));
  expected_sc_3.push_back(CooFormat(13, 0, 1.0));
  expected_sc_3.push_back(CooFormat(12, 1, 1.0));
  expected_sc_3.push_back(CooFormat(13, 1, 1.0));
  expected_sc_3.push_back(CooFormat(12, 2, 1.0));
  expected_sc_3.push_back(CooFormat(13, 2, 1.0));
  expected_sc_3.push_back(CooFormat(12, 3, 1.0));
  expected_sc_3.push_back(CooFormat(13, 3, 1.0));

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
  ExtractedCooTensors extracted_coo_tensors(4, 8, coo_formats);
  StackedTableMetadata stacked_table_metadata(
      "stacked_table", /*feature_index=*/0, /*max_ids_per_partition=*/32,
      /*max_unique_ids_per_partition=*/32, /*row_offset=*/0, /*col_offset=*/0,
      /*col_shift=*/0, /*batch_size=*/0);
  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 4,
      .global_device_count = 1,
      .num_sc_per_device = 4,
      .allow_id_dropping = false,
      .feature_stacking_strategy = FeatureStackingStrategy::kStackThenSplit,
  };
  MinibatchingSplit minibatching_split = 0;
  StatsPerHost stats_per_host(/*local_device_count=*/1, /*num_partitions=*/4,
                              /*num_sc_per_device=*/4);
  internal::StatsPerDevice stats_per_device =
      stats_per_host.GetStatsPerDevice(0);
  PartitionedCooTensors coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          extracted_coo_tensors, stacked_table_metadata, options,
          stats_per_device, minibatching_split);

  EXPECT_EQ(minibatching_split, 0);

  MatrixXi row_pointers(1, 8 * 4);
  MatrixXi embedding_ids(1, 40 * 4);
  MatrixXi sample_ids(1, 40 * 4);
  MatrixXf gains(1, 40 * 4);

  CsrArraysPerHost csr_arrays_per_host(row_pointers, embedding_ids, sample_ids,
                                       gains);

  internal::CsrArraysPerDevice csr_array =
      csr_arrays_per_host.GetCsrArraysPerDevice(0);
  int dropped_static_bound = 0;
  FillLocalDeviceBuffer(coo_tensors_by_id,
                        /*row_pointers_size_per_sc=*/8,
                        /*coo_buffer_size_per_sc=*/40,
                        /*batch_size_per_sc=*/2, options, csr_array,
                        dropped_static_bound);

  std::array<int, 32> expected_row_pointers = {
      2, 10, 18, 26, 32, 32, 32, 32,  //
      2, 10, 18, 26, 32, 32, 32, 32,  //
      2, 10, 18, 26, 32, 32, 32, 32,  //
      2, 10, 18, 26, 32, 32, 32, 32,  //
  };
  EXPECT_THAT(csr_array.row_pointers, ElementsAreArray(expected_row_pointers));

  EXPECT_THAT(
      csr_array.embedding_ids,
      ElementsAre(
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          _, _, _, _, _, _, _, _,                                      //
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          _, _, _, _, _, _, _, _,                                      //
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          _, _, _, _, _, _, _, _,                                      //
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          _, _, _, _, _, _, _, _));
  EXPECT_THAT(
      csr_array.sample_ids,
      ElementsAre(
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          _, _, _, _, _, _, _, _,                                      //
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          _, _, _, _, _, _, _, _,                                      //
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          _, _, _, _, _, _, _, _,                                      //
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          _, _, _, _, _, _, _, _));
  EXPECT_THAT(
      csr_array.gains,
      ElementsAre(
          1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
          1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
          1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
          1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
          _, _, _, _, _, _, _, _,                                      //
          1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
          1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
          1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
          1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
          _, _, _, _, _, _, _, _,                                      //
          1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
          1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
          1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
          1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
          _, _, _, _, _, _, _, _,                                      //
          1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
          1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
          1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
          1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),  //
          _, _, _, _, _, _, _, _));

  EXPECT_EQ(dropped_static_bound, 0);
}

TEST(InputPreprocessingUtilTest, FillBufferMinibatchingSingleMinibatch) {
  std::vector<CooFormat> coo_formats;

  for (int row = 0; row < 8; ++row) {
    coo_formats.push_back(CooFormat(row, 0, 1.0));
    coo_formats.push_back(CooFormat(row, 1, 1.0));
    coo_formats.push_back(CooFormat(row, 2, 1.0));
    coo_formats.push_back(CooFormat(row, 3, 1.0));
  }
  ExtractedCooTensors extracted_coo_tensors(4, 8, coo_formats);
  StackedTableMetadata stacked_table_metadata(
      "stacked_table", /*feature_index=*/0, /*max_ids_per_partition=*/32,
      /*max_unique_ids_per_partition=*/32, /*row_offset=*/0, /*col_offset=*/0,
      /*col_shift=*/0, /*batch_size=*/0);
  auto hash_fn = std::identity();  // No hashing for simplicity.
  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 1,
      .global_device_count = 1,
      .num_sc_per_device = 4,
      .allow_id_dropping = false,
      .feature_stacking_strategy = FeatureStackingStrategy::kStackThenSplit,
      .enable_minibatching = true,
      .minibatching_bucketing_hash_fn = hash_fn};
  MinibatchingSplit minibatching_split = 0;
  StatsPerHost stats_per_host(/*local_device_count=*/1, /*num_partitions=*/4,
                              /*num_sc_per_device=*/4);
  internal::StatsPerDevice stats_per_device =
      stats_per_host.GetStatsPerDevice(0);
  PartitionedCooTensors coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          extracted_coo_tensors, stacked_table_metadata, options,
          stats_per_device, minibatching_split);

  coo_tensors_by_id.MergeAll();

  MatrixXi row_pointers(1, 8 * 4);
  MatrixXi embedding_ids(1, 40 * 4);
  MatrixXi sample_ids(1, 40 * 4);
  MatrixXf gains(1, 40 * 4);

  CsrArraysPerHost csr_arrays_per_host(row_pointers, embedding_ids, sample_ids,
                                       gains);

  internal::CsrArraysPerDevice csr_array =
      csr_arrays_per_host.GetCsrArraysPerDevice(0);
  FillLocalDeviceBuffer(coo_tensors_by_id,
                        /*row_pointers_size_per_sc=*/8,
                        /*coo_buffer_size_per_sc=*/40,
                        /*batch_size_per_sc=*/2, options, csr_array,
                        stats_per_device.dropped_id_count);

  std::array<int, 32> expected_row_pointers = {
      2,  10,  18,  26,  32,  32,  32,  32,  // MB0
      34, 42,  50,  58,  64,  64,  64,  64,  // MB1
      66, 74,  82,  90,  96,  96,  96,  96,  // MB2
      98, 106, 114, 122, 128, 128, 128, 128  // MB3
  };
  EXPECT_THAT(csr_array.row_pointers, ElementsAreArray(expected_row_pointers));

  EXPECT_THAT(
      csr_array.embedding_ids,
      ElementsAre(
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          _, _, _, _, _, _, _, _,                                      //
          _, _, _, _, _, _, _, _,                                      //
          _, _, _, _, _, _, _, _,                                      //
          _, _, _, _, _, _, _, _));
  EXPECT_THAT(
      csr_array.sample_ids,
      ElementsAre(
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,  //
          _, _, _, _, _, _, _, _,                                      //
          _, _, _, _, _, _, _, _,                                      //
          _, _, _, _, _, _, _, _,                                      //
          _, _, _, _, _, _, _, _));
  EXPECT_THAT(
      csr_array.gains,
      ElementsAre(
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
          _, _, _, _, _, _, _, _,                                      //
          _, _, _, _, _, _, _, _,                                      //
          _, _, _, _, _, _, _, _,                                      //
          _, _, _, _, _, _, _, _));
  EXPECT_EQ(stats_per_device.dropped_id_count, 0);
}

TEST(InputPreprocessingUtilTest, FillBufferMinibatchingFourMinibatches) {
  std::vector<CooFormat> coo_formats;

  for (int row = 0; row < 8; ++row) {
    for (int col = 0; col < 64; ++col) {
      coo_formats.push_back(CooFormat(row, col, 1.0));
    }
  }
  ExtractedCooTensors extracted_coo_tensors(4, 8, coo_formats);
  StackedTableMetadata stacked_table_metadata(
      "stacked_table", /*feature_index=*/0, /*max_ids_per_partition=*/32,
      /*max_unique_ids_per_partition=*/32, /*row_offset=*/0, /*col_offset=*/0,
      /*col_shift=*/0, /*batch_size=*/0);
  auto hash_fn = std::identity();  // No hashing for simplicity.
  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 1,
      .global_device_count = 1,
      .num_sc_per_device = 4,
      .allow_id_dropping = false,
      .feature_stacking_strategy = FeatureStackingStrategy::kStackThenSplit,
      .enable_minibatching = true,
      .minibatching_bucketing_hash_fn = hash_fn};
  MinibatchingSplit minibatching_split = 0;
  StatsPerHost stats_per_host(/*local_device_count=*/1, /*num_partitions=*/4,
                              /*num_sc_per_device=*/4);
  internal::StatsPerDevice stats_per_device =
      stats_per_host.GetStatsPerDevice(0);
  PartitionedCooTensors coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          extracted_coo_tensors, stacked_table_metadata, options,
          stats_per_device, minibatching_split);
  EXPECT_EQ(stats_per_device.dropped_id_count, 0);

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
  EXPECT_EQ(coo_tensors_by_id(0, 3).size(), 16 * 2);

  const int coo_buffer_size_per_sc = 168;
  const int row_pointers_size = 128;
  const int num_devices = 1;
  MatrixXi row_pointers(num_devices, row_pointers_size);
  MatrixXi embedding_ids(num_devices, coo_buffer_size_per_sc * 4);
  MatrixXi sample_ids(num_devices, coo_buffer_size_per_sc * 4);
  MatrixXf gains(num_devices, coo_buffer_size_per_sc * 4);
  CsrArraysPerHost csr_arrays_per_host(row_pointers, embedding_ids, sample_ids,
                                       gains);

  internal::CsrArraysPerDevice csr_array =
      csr_arrays_per_host.GetCsrArraysPerDevice(0);

  FillLocalDeviceBuffer(coo_tensors_by_id,
                        /*row_pointers_size_per_bucket=*/8,
                        coo_buffer_size_per_sc,
                        /*batch_size_per_sc=*/2, options, csr_array,
                        stats_per_device.dropped_id_count);

  std::array<int, row_pointers_size> expected_row_pointers = {
      // SC0 (Base: 0)
      8,  16,  24,  32,  32,  32,  32,  32,   // MB0
      36,  44,  52,  60,  64,  64,  64,  64,   // MB1
      76,  92, 108, 124, 128, 128, 128, 128,  // MB2
      136, 144, 152, 160, 160, 160, 160, 160,  // MB3
      // SC1 (Base: 160)
      168, 176, 184, 192, 192, 192, 192, 192,  // MB0
      196, 204, 212, 220, 224, 224, 224, 224,  // MB1
      236, 252, 268, 284, 288, 288, 288, 288,  // MB2
      296, 304, 312, 320, 320, 320, 320, 320,  // MB3
      // SC2 (Base: 320)
      328, 336, 344, 352, 352, 352, 352, 352,  // MB0
      356, 364, 372, 380, 384, 384, 384, 384,  // MB1
      396, 412, 428, 444, 448, 448, 448, 448,  // MB2
      456, 464, 472, 480, 480, 480, 480, 480,  // MB3
      // SC3 (Base: 480)
      488, 496, 504, 512, 512, 512, 512, 512,  // MB0
      516, 524, 532, 540, 544, 544, 544, 544,  // MB1
      556, 572, 588, 604, 608, 608, 608, 608,  // MB2
      616, 624, 632, 640, 640, 640, 640, 640,  // MB3
  };
  EXPECT_THAT(csr_array.row_pointers, ElementsAreArray(expected_row_pointers));

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
        << (RowVectorXi(16) << 0, 1, 0, 1, 0, 1
            , 0, 1, 0, 1, 0, 1, INT_MAX,
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

  EXPECT_THAT(absl::MakeSpan(csr_array.embedding_ids).subspan(0, 640),
              Pointwise(Eq(), expected_embedding_ids.segment(0, 640)));
  EXPECT_THAT(absl::MakeSpan(csr_array.sample_ids).subspan(0, 640),
              Pointwise(Eq(), expected_sample_ids.segment(0, 640)));
  EXPECT_THAT(absl::MakeSpan(csr_array.gains).subspan(0, 640),
              Pointwise(NanSensitiveFloatEq(), expected_gains.segment(0, 640)));
  EXPECT_EQ(stats_per_device.dropped_id_count, 0);
}

TEST(InputPreprocessingUtilTest,
     FillBufferStaticBoundCountsOneDropNoMinibatching) {
  std::vector<CooFormat> coo_formats;
  coo_formats.emplace_back(/*row=*/0, /*col=*/0, /*gain=*/1.0);
  coo_formats.emplace_back(/*row=*/0, /*col=*/1, /*gain=*/1.0);
  coo_formats.emplace_back(/*row=*/1, /*col=*/0, /*gain=*/1.0);
  coo_formats.emplace_back(/*row=*/1, /*col=*/1, /*gain=*/1.0);

  ExtractedCooTensors extracted(/*num_sc_per_device=*/1,
                                /*batch_size_for_device=*/4, coo_formats);

  StackedTableMetadata meta("stacked_table", /*feature_index=*/0,
                            /*max_ids_per_partition=*/32,
                            /*max_unique_ids_per_partition=*/32,
                            /*row_offset=*/0, /*col_offset=*/0, /*col_shift=*/0,
                            /*batch_size=*/0);

  PreprocessSparseDenseMatmulInputOptions opts{
      .local_device_count = 1,
      .global_device_count = 1,
      .num_sc_per_device = 1,
      .allow_id_dropping = false,
      .feature_stacking_strategy = FeatureStackingStrategy::kStackThenSplit,
  };

  bool minibatching_required = false;
  StatsPerHost stats_per_host(/*local_device_count=*/1, /*num_partitions=*/1,
                              /*num_sc_per_device=*/1);
  internal::StatsPerDevice stats_per_device =
      stats_per_host.GetStatsPerDevice(0);
  PartitionedCooTensors grouped = SortAndGroupCooTensorsPerLocalDevice(
      extracted, meta, opts, stats_per_device, minibatching_required);
  int dropped_sort = stats_per_device.dropped_id_count;

  const int row_ptrs_size_per_bucket = 4;
  const int coo_buffer_size_per_sc = 3;
  const int batch_size_per_sc = 4;

  MatrixXi row_pointers(1, row_ptrs_size_per_bucket);
  MatrixXi embedding_ids(1, coo_buffer_size_per_sc);
  MatrixXi sample_ids(1, coo_buffer_size_per_sc);
  MatrixXf gains(1, coo_buffer_size_per_sc);
  CsrArraysPerHost csr_arrays_per_host(row_pointers, embedding_ids, sample_ids,
                                       gains);

  internal::CsrArraysPerDevice csr_arrays =
      csr_arrays_per_host.GetCsrArraysPerDevice(0);

  int dropped_static = 0;
  FillLocalDeviceBuffer(grouped, row_ptrs_size_per_bucket,
                        coo_buffer_size_per_sc, batch_size_per_sc, opts,
                        csr_arrays,
                        /*dropped_id_count_static_bound=*/dropped_static);

  EXPECT_EQ(dropped_static, 1);
  EXPECT_EQ(dropped_sort, 0);
}

TEST(InputPreprocessingUtilTest,
     FillBufferStaticBoundCountsDropsWithMinibatching) {
  std::vector<CooFormat> coo_formats;
  // 4 samples, 2 ids each. Total 8 ids.
  // col=0 will be in minibatch 0, col=1 in minibatch 1.
  coo_formats.emplace_back(/*row=*/0, /*col=*/0, /*gain=*/1.0);
  coo_formats.emplace_back(/*row=*/0, /*col=*/1, /*gain=*/1.0);
  coo_formats.emplace_back(/*row=*/1, /*col=*/0, /*gain=*/1.0);
  coo_formats.emplace_back(/*row=*/1, /*col=*/1, /*gain=*/1.0);
  coo_formats.emplace_back(/*row=*/2, /*col=*/0, /*gain=*/1.0);
  coo_formats.emplace_back(/*row=*/2, /*col=*/1, /*gain=*/1.0);
  coo_formats.emplace_back(/*row=*/3, /*col=*/0, /*gain=*/1.0);
  coo_formats.emplace_back(/*row=*/3, /*col=*/1, /*gain=*/1.0);

  ExtractedCooTensors extracted(/*num_sc_per_device=*/1,
                                /*batch_size_for_device=*/4, coo_formats);

  StackedTableMetadata meta("stacked_table", /*feature_index=*/0,
                            /*max_ids_per_partition=*/32,
                            /*max_unique_ids_per_partition=*/32,
                            /*row_offset=*/0, /*col_offset=*/0, /*col_shift=*/0,
                            /*batch_size=*/0);

  auto hash_fn = std::identity();  // No hashing for simplicity.
  PreprocessSparseDenseMatmulInputOptions opts{
      .local_device_count = 1,
      .global_device_count = 1,
      .num_sc_per_device = 1,
      .allow_id_dropping = false,
      .feature_stacking_strategy = FeatureStackingStrategy::kStackThenSplit,
      .enable_minibatching = true,
      .minibatching_bucketing_hash_fn = hash_fn,
  };

  MinibatchingSplit minibatching_split = 0;
  StatsPerHost stats_per_host(/*local_device_count=*/1, /*num_partitions=*/1,
                              /*num_sc_per_device=*/1);
  internal::StatsPerDevice stats_per_device =
      stats_per_host.GetStatsPerDevice(0);
  PartitionedCooTensors grouped = SortAndGroupCooTensorsPerLocalDevice(
      extracted, meta, opts, stats_per_device, minibatching_split);

  // Create 2 minibatches by splitting based on bucket ID.
  minibatching_split.set(0);
  grouped.Merge(minibatching_split);
  ASSERT_EQ(grouped.GetNumMinibatches(), 2);

  static constexpr int row_ptrs_size_per_bucket = 4;
  // Buffer size is 6, but there are 8 total IDs (4 in each minibatch).
  static constexpr int coo_buffer_size_per_sc = 6;
  static constexpr int batch_size_per_sc = 4;

  MatrixXi row_pointers(1,
                        row_ptrs_size_per_bucket * grouped.GetNumMinibatches());
  MatrixXi embedding_ids(1, coo_buffer_size_per_sc);
  MatrixXi sample_ids(1, coo_buffer_size_per_sc);
  MatrixXf gains(1, coo_buffer_size_per_sc);
  CsrArraysPerHost csr_arrays_per_host(row_pointers, embedding_ids, sample_ids,
                                       gains);

  internal::CsrArraysPerDevice csr_arrays =
      csr_arrays_per_host.GetCsrArraysPerDevice(0);

  int dropped_static = 0;
  FillLocalDeviceBuffer(grouped, row_ptrs_size_per_bucket,
                        coo_buffer_size_per_sc, batch_size_per_sc, opts,
                        csr_arrays,
                        /*dropped_id_count_static_bound=*/dropped_static);

  EXPECT_EQ(stats_per_device.dropped_id_count, 0);
  // Minibatch 0 has 4 IDs, Minibatch 1 has 4 IDs. Buffer size is 6.
  // Minibatch 0 is fully written (4 IDs).
  // Although there are 2 slots left, the entire Minibatch 1 (4 IDs)
  // needs to fit, potentially with padding. Since it doesn't, all 4 IDs
  // from Minibatch 1 are dropped.
  EXPECT_EQ(dropped_static, 4);

  // Row pointers for MB0 all point to the end index 4.
  // Row pointers for MB1 start from 4, and since all are dropped, they clamp at
  // the buffer size 6.
  std::array<int, 2 * row_ptrs_size_per_bucket> expected_row_pointers = {
      4, 6, 6, 6, 6, 6, 6, 6};
  EXPECT_THAT(csr_arrays.row_pointers, ElementsAreArray(expected_row_pointers));

  // Embedding IDs from MB0 are {0, 0, 0, 0}. MB1 is dropped.
  std::array<int, coo_buffer_size_per_sc> expected_embedding_ids = {
      0, 0, 0, 0, INT_MAX, INT_MAX};
  EXPECT_THAT(absl::MakeSpan(csr_arrays.embedding_ids).subspan(0, 6),
              ElementsAreArray(expected_embedding_ids));

  // Sample IDs from MB0 are {0, 1, 2, 3}. MB1 is dropped.
  std::array<int, coo_buffer_size_per_sc> expected_sample_ids = {
      0, 1, 2, 3, INT_MAX, INT_MAX};
  EXPECT_THAT(absl::MakeSpan(csr_arrays.sample_ids).subspan(0, 6),
              ElementsAreArray(expected_sample_ids));
}

}  // namespace

}  // namespace jax_sc_embedding
