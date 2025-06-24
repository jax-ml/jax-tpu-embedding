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
#include <climits>
#include <cstdint>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "Eigen/Core"  // from @eigen_archive

namespace jax_sc_embedding {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::IsNan;
using ::testing::Pair;

TEST(InputPreprocessingUtilTest, ColIds) {
  // int GetColId(int col_id, int col_shift, int col_offset, int num_scs_mod,
  //          int num_scs_mod_inv);
  EXPECT_EQ(GetColId(/*col_id=*/2, /*col_shift=*/4, /*col_offset=*/32,
                     /*num_scs_mod=*/3, /*num_scs_mod_inv=*/-4),
            34);
  EXPECT_EQ(GetColId(/*col_id=*/38, /*col_shift=*/0, /*col_offset=*/0,
                     /*num_scs_mod=*/3, /*num_scs_mod_inv=*/-4),
            38);
  EXPECT_EQ(GetColId(/*col_id=*/10, /*col_shift=*/0, /*col_offset=*/32,
                     /*num_scs_mod=*/7, /*num_scs_mod_inv=*/-8),
            42);
  EXPECT_EQ(GetColId(/*col_id=*/26, /*col_shift=*/0, /*col_offset=*/0,
                     /*num_scs_mod=*/3, /*num_scs_mod_inv=*/-4),
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
  stacked_table_metadata[0].suggested_coo_buffer_size = 48;
  EXPECT_EQ(ComputeCooBufferSizePerDevice(/*num_scs=*/4,
                                          /*num_scs_per_device=*/4,
                                          stacked_table_metadata),
            64);

  stacked_table_metadata[0].suggested_coo_buffer_size = 96;
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
  std::vector<std::vector<CooFormat>> coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          coo_formats, /*batch_size_per_sc=*/2, /*num_scs=*/4,
          /*batch_size_for_device=*/8,
          /*max_ids_per_partition=*/32,
          /*max_unique_ids_per_partition=*/32,
          /*stacked_table_name=*/"stacked_table",
          /*allow_id_dropping=*/false, /*num_sc_per_device=*/4,
          /*total_num_coo_tensors=*/coo_formats.size(), max_id_per_sc,
          max_unique_id_per_sc, required_buffer_sizes_per_sc);

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

  EXPECT_THAT(coo_tensors_by_id[0], ElementsAreArray(expected_sc_0));
  EXPECT_THAT(coo_tensors_by_id[1], ElementsAreArray(expected_sc_1));
  EXPECT_THAT(coo_tensors_by_id[2], ElementsAreArray(expected_sc_2));
  EXPECT_THAT(coo_tensors_by_id[3], ElementsAreArray(expected_sc_3));
  EXPECT_THAT(max_id_per_sc, ElementsAreArray({2, 2, 2, 2}));
  EXPECT_THAT(max_unique_id_per_sc, ElementsAreArray({1, 1, 1, 1}));
  EXPECT_THAT(required_buffer_sizes_per_sc, ElementsAreArray({32, 32, 32, 32}));
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
  std::vector<std::vector<CooFormat>> coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          coo_formats, /*batch_size_per_sc=*/4, /*num_scs=*/2,
          /*batch_size_for_device=*/8,
          /*max_ids_per_partition=*/32,
          /*max_unique_ids_per_partition=*/32,
          /*stacked_table_name=*/"stacked_table",
          /*allow_id_dropping=*/false, /*num_sc_per_device=*/2,
          /*total_num_coo_tensors=*/coo_formats.size(), max_id_per_sc,
          max_unique_id_per_sc, required_buffer_sizes_per_sc);

  EXPECT_THAT(coo_tensors_by_id[0],
              ElementsAre(CooFormat(0, 0, 1.0), CooFormat(1, 0, 1.0),
                          CooFormat(2, 0, 1.0), CooFormat(3, 0, 1.0),
                          CooFormat(0, 2, 1.0), CooFormat(1, 2, 1.0),
                          CooFormat(2, 2, 1.0), CooFormat(3, 2, 1.0),
                          CooFormat(0, 1, 1.0), CooFormat(1, 1, 1.0),
                          CooFormat(2, 1, 1.0), CooFormat(3, 1, 1.0),
                          CooFormat(0, 3, 1.0), CooFormat(1, 3, 1.0),
                          CooFormat(2, 3, 1.0), CooFormat(3, 3, 1.0)));
  EXPECT_THAT(coo_tensors_by_id[1],
              ElementsAre(CooFormat(4, 0, 1.0), CooFormat(5, 0, 1.0),
                          CooFormat(6, 0, 1.0), CooFormat(7, 0, 1.0),
                          CooFormat(4, 2, 1.0), CooFormat(5, 2, 1.0),
                          CooFormat(6, 2, 1.0), CooFormat(7, 2, 1.0),
                          CooFormat(4, 1, 1.0), CooFormat(5, 1, 1.0),
                          CooFormat(6, 1, 1.0), CooFormat(7, 1, 1.0),
                          CooFormat(4, 3, 1.0), CooFormat(5, 3, 1.0),
                          CooFormat(6, 3, 1.0), CooFormat(7, 3, 1.0)));
  EXPECT_THAT(max_id_per_sc, ElementsAreArray({8, 8}));
  EXPECT_THAT(max_unique_id_per_sc, ElementsAreArray({2, 2}));
  EXPECT_THAT(required_buffer_sizes_per_sc, ElementsAreArray({16, 16}));
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

  std::vector<std::vector<CooFormat>> coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          coo_formats, /*batch_size_per_sc=*/2, /*num_scs=*/4,
          /*batch_size_for_device=*/8,
          /*max_ids_per_partition=*/2,
          /*max_unique_ids_per_partition=*/1,
          /*stacked_table_name=*/"stacked_table",
          /*allow_id_dropping=*/false, /*num_sc_per_device=*/4,
          /*total_num_coo_tensors=*/coo_formats.size(), max_id_per_sc,
          max_unique_id_per_sc, required_buffer_sizes_per_sc);
  EXPECT_THAT(max_id_per_sc, ElementsAreArray({2, 2, 2, 2}));
  EXPECT_THAT(max_unique_id_per_sc, ElementsAreArray({1, 1, 1, 1}));
  EXPECT_THAT(required_buffer_sizes_per_sc, ElementsAreArray({32, 32, 32, 32}));
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
  std::vector<std::vector<CooFormat>> coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          coo_formats, /*batch_size_per_sc=*/4, /*num_scs=*/4,
          /*batch_size_for_device=*/16,
          /*max_ids_per_partition=*/4,
          /*max_unique_ids_per_partition=*/1,
          /*stacked_table_name=*/"stacked_table",
          /*allow_id_dropping=*/false, /*num_sc_per_device=*/4,
          /*total_num_coo_tensors=*/coo_formats.size(), max_id_per_sc,
          max_unique_id_per_sc, required_buffer_sizes_per_sc);
  EXPECT_THAT(max_id_per_sc, ElementsAreArray({4, 4, 4, 4}));
  EXPECT_THAT(max_unique_id_per_sc, ElementsAreArray({1, 1, 1, 1}));
  EXPECT_THAT(required_buffer_sizes_per_sc, ElementsAreArray({32, 32, 32, 32}));
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

  std::vector<std::vector<CooFormat>> coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          coo_formats, /*batch_size_per_sc=*/4, /*num_scs=*/4,
          /*batch_size_for_device=*/16,
          /*max_ids_per_partition=*/8,
          /*max_unique_ids_per_partition=*/2,
          /*stacked_table_name=*/"stacked_table",
          /*allow_id_dropping=*/false, /*num_sc_per_device=*/4,
          /*total_num_coo_tensors=*/coo_formats.size(), max_id_per_sc,
          max_unique_id_per_sc, required_buffer_sizes_per_sc);
  EXPECT_THAT(max_id_per_sc, ElementsAreArray({8, 8, 8, 8}));
  EXPECT_THAT(max_unique_id_per_sc, ElementsAreArray({2, 2, 2, 2}));
  // 4 partitions of size 8 with 2 elements each
  EXPECT_THAT(required_buffer_sizes_per_sc, ElementsAreArray({32, 32, 32, 32}));
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

  std::vector<std::vector<CooFormat>> coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          coo_formats, /*batch_size_per_sc=*/32, /*num_scs=*/4,
          /*batch_size_for_device=*/128,
          /*max_ids_per_partition=*/64,
          /*max_unique_ids_per_partition=*/2,
          /*stacked_table_name=*/"stacked_table",
          /*allow_id_dropping=*/false, /*num_sc_per_device=*/4,
          /*total_num_coo_tensors=*/coo_formats.size(), max_id_per_sc,
          max_unique_id_per_sc, required_buffer_sizes_per_sc);
  EXPECT_THAT(max_id_per_sc, ElementsAreArray({64, 64, 64, 64}));
  EXPECT_THAT(max_unique_id_per_sc, ElementsAreArray({2, 2, 2, 2}));
  // 8 partitions of size 256 with 32 elements each
  EXPECT_THAT(required_buffer_sizes_per_sc,
              ElementsAreArray({256, 256, 256, 256}));
}

TEST(InputPreprocessingUtilTest, SortAndGroup_VerifyIdLimitations5) {
  std::vector<CooFormat> coo_formats;

  // With 128 samples, each sample has 8 ids [0, 4, 8, 16]
  // Sparsecore 0 alone serves all 4 rows of data [0, 4, 8, 16]
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

  std::vector<std::vector<CooFormat>> coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          coo_formats, /*batch_size_per_sc=*/32, /*num_scs=*/4,
          /*batch_size_for_device=*/128,
          /*max_ids_per_partition=*/128,
          /*max_unique_ids_per_partition=*/4,
          /*stacked_table_name=*/"stacked_table",
          /*allow_id_dropping=*/false, /*num_sc_per_device=*/4,
          /*total_num_coo_tensors=*/coo_formats.size(), max_id_per_sc,
          max_unique_id_per_sc, required_buffer_sizes_per_sc);
  EXPECT_THAT(max_id_per_sc, ElementsAreArray({128, 0, 0, 0}));
  EXPECT_THAT(max_unique_id_per_sc, ElementsAreArray({4, 0, 0, 0}));
  // 1 partition of size 128 with 128 elements
  EXPECT_THAT(required_buffer_sizes_per_sc,
              ElementsAreArray({128, 128, 128, 128}));
}

TEST(InputPreprocessingUtilTest, SortAndGroup_VerifyIdLimitations6) {
  std::vector<CooFormat> coo_formats;

  // This is one of the worst case scenarios.
  // Every ID is unique, and all IDs come from the same sparsecore.
  //
  // With 128 samples, each sample has 1 id [row * 4]
  // Sparsecore 0 alone serves all 128 rows of data [0, 4, 8, ...]
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

  std::vector<std::vector<CooFormat>> coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          coo_formats, /*batch_size_per_sc=*/32, /*num_scs=*/4,
          /*batch_size_for_device=*/128,
          /*max_ids_per_partition=*/32,
          /*max_unique_ids_per_partition=*/32,
          /*stacked_table_name=*/"stacked_table",
          /*allow_id_dropping=*/false, /*num_sc_per_device=*/4,
          /*total_num_coo_tensors=*/coo_formats.size(), max_id_per_sc,
          max_unique_id_per_sc, required_buffer_sizes_per_sc);
  EXPECT_THAT(max_id_per_sc, ElementsAreArray({32, 0, 0, 0}));
  EXPECT_THAT(max_unique_id_per_sc, ElementsAreArray({32, 0, 0, 0}));
  // 1 partition of size 32 with 32 elements
  EXPECT_THAT(required_buffer_sizes_per_sc, ElementsAreArray({32, 32, 32, 32}));
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

  std::vector<std::vector<CooFormat>> coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          coo_formats, /*batch_size_per_sc=*/4, /*num_scs=*/4,
          /*batch_size_for_device=*/16,
          /*max_ids_per_partition=*/2,
          /*max_unique_ids_per_partition=*/1,
          /*stacked_table_name=*/"stacked_table",
          /*allow_id_dropping=*/true, /*num_sc_per_device=*/4,
          /*total_num_coo_tensors=*/coo_formats.size(), max_id_per_sc,
          max_unique_id_per_sc, required_buffer_sizes_per_sc);
  EXPECT_THAT(max_id_per_sc, ElementsAreArray({4, 4, 4, 4}));
  EXPECT_THAT(max_unique_id_per_sc, ElementsAreArray({1, 1, 1, 1}));
  // 4 partition of size 8 with 4 element each
  EXPECT_THAT(required_buffer_sizes_per_sc, ElementsAreArray({32, 32, 32, 32}));

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

  EXPECT_THAT(coo_tensors_by_id[0], ElementsAreArray(expected_sc_0));
  EXPECT_THAT(coo_tensors_by_id[1], ElementsAreArray(expected_sc_1));
  EXPECT_THAT(coo_tensors_by_id[2], ElementsAreArray(expected_sc_2));
  EXPECT_THAT(coo_tensors_by_id[3], ElementsAreArray(expected_sc_3));
}

TEST(InputPreprocessingUtilTest, FillRowPointers) {
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
  std::vector<std::vector<CooFormat>> coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          coo_formats, /*batch_size_per_sc=*/2, /*num_scs=*/4,
          /*batch_size_for_device=*/8,
          /*max_ids_per_partition=*/32,
          /*max_unique_ids_per_partition=*/32,
          /*stacked_table_name=*/"stacked_table",
          /*allow_id_dropping=*/false, /*num_sc_per_device=*/4,
          /*total_num_coo_tensors=*/coo_formats.size(), max_id_per_sc,
          max_unique_id_per_sc, required_buffer_sizes_per_sc);

  Eigen::VectorXi row_pointers(8 * 4);
  Eigen::VectorXi embedding_ids(32 * 4);
  Eigen::VectorXi sample_ids(32 * 4);
  Eigen::VectorXf gains(32 * 4);
  FillRowPointersPerLocalDevice(coo_tensors_by_id,
                                /*row_pointers_size_per_sc=*/8,
                                /*coo_buffer_size_per_sc=*/32,
                                /*batch_size_per_sc=*/2,
                                /*num_scs=*/4, /*num_sc_per_device=*/4,
                                row_pointers, embedding_ids, sample_ids, gains);

  std::array<int, 32> expected_row_pointers = {
      2, 10, 18, 26, 26, 26, 26, 26, 2, 10, 18, 26, 26, 26, 26, 26,
      2, 10, 18, 26, 26, 26, 26, 26, 2, 10, 18, 26, 26, 26, 26, 26};
  EXPECT_THAT(row_pointers, ElementsAreArray(expected_row_pointers));

  std::array<int, 128> expected_embedding_ids = {
      0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0, 0, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX};

  EXPECT_THAT(embedding_ids, ElementsAreArray(expected_embedding_ids));

  std::array<int, 128> expected_sample_ids = {
      0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
      0, 1, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX,
  };
  EXPECT_THAT(sample_ids, ElementsAreArray(expected_sample_ids));

  EXPECT_THAT(
      gains,
      ElementsAre(
          1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), 1, 1,
          IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), 1, 1, IsNan(),
          IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), 1, 1, IsNan(), IsNan(),
          IsNan(), IsNan(), IsNan(), IsNan(), 1, 1, IsNan(), IsNan(), IsNan(),
          IsNan(), IsNan(), IsNan(), 1, 1, IsNan(), IsNan(), IsNan(), IsNan(),
          IsNan(), IsNan(), 1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),
          IsNan(), 1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),
          1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), 1, 1,
          IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), 1, 1, IsNan(),
          IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), 1, 1, IsNan(), IsNan(),
          IsNan(), IsNan(), IsNan(), IsNan(), 1, 1, IsNan(), IsNan(), IsNan(),
          IsNan(), IsNan(), IsNan(), 1, 1, IsNan(), IsNan(), IsNan(), IsNan(),
          IsNan(), IsNan(), 1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(),
          IsNan(), 1, 1, IsNan(), IsNan(), IsNan(), IsNan(), IsNan(), IsNan()));
}

}  // namespace
}  // namespace jax_sc_embedding
