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
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/synchronization/blocking_counter.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/abstract_input_batch.h"
#include "jax_tpu_embedding/sparsecore/lib/core/coo_format.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "jax_tpu_embedding/sparsecore/lib/core/ragged_tensor_input_batch.h"
#include "tsl/platform/env.h"  // from @tsl
#include "tsl/platform/threadpool.h"  // from @tsl

namespace jax_sc_embedding {
namespace {

using ::testing::Each;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::Gt;
using ::testing::SizeIs;

class TableStackingTest : public ::testing::Test {
  using InputBatch =
      RaggedTensorInputBatch<std::vector<int64_t>, std::vector<int32_t>>;

 protected:
  InputBatch input_a_{std::vector<int64_t>{5, 18,  //
                                           18, 0,  //
                                           0, 20,  //
                                           18, 0,  //
                                           18, 0,  //
                                           0, 20,  //
                                           5, 18,  //
                                           18, 0},
                      std::vector<int32_t>{0, 2, 4, 6, 8, 10, 12, 14, 16}};
  InputBatch input_b_{std::vector<int64_t>{2,   //
                                           10,  //
                                           1,   //
                                           9,   //
                                           3,   //
                                           7,   //
                                           4,   //
                                           8},
                      std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8}};
  InputBatch input_c_{std::vector<int64_t>{1,                    //
                                           2, 2,                 //
                                           3, 3, 3,              //
                                           4, 4, 4, 4,           //
                                           5, 5, 5, 5, 5,        //
                                           6, 6, 6, 6, 6, 6,     //
                                           7, 7, 7, 7, 7, 7, 7,  //
                                           8, 8, 8, 8, 8, 8, 8, 8},
                      std::vector<int32_t>{0, 1, 3, 6, 10, 15, 21, 28, 36}};
  InputBatch input_d_{
      std::vector<int64_t>{
          9,  9,  9,  9,  9,  9,  9,  9,  9,                           //
          10, 10, 10, 10, 10, 10, 10, 10, 10, 10,                      //
          11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,                  //
          12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,              //
          13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,          //
          14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,      //
          15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,  //
          16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16},
      std::vector<int32_t>{0, 9, 19, 30, 42, 55, 69, 84, 100}};

  std::vector<StackedTableMetadata> stacked_table_metadata_multi_{
      StackedTableMetadata(
          /*name=*/"table_0",
          /*feature_index=*/0, /*max_ids_per_partition=*/16,
          /*max_unique_ids_per_partition=*/16, /*row_offset=*/0,
          /*col_offset=*/0, /*col_shift=*/0, /*batch_size=*/16),
      StackedTableMetadata(
          /*name=*/"table_1",
          /*feature_index=*/1, /*max_ids_per_partition=*/16,
          /*max_unique_ids_per_partition=*/16, /*row_offset=*/16,
          /*col_offset=*/32, /*col_shift=*/0, /*batch_size=*/16)};

  std::vector<StackedTableMetadata> stacked_table_metadata_single_{
      StackedTableMetadata(
          /*name=*/"table_0",
          /*feature_index=*/0, /*max_ids_per_partition=*/16,
          /*max_unique_ids_per_partition=*/16, /*row_offset=*/0,
          /*col_offset=*/0, /*col_shift=*/0, /*batch_size=*/8),
      StackedTableMetadata(
          /*name=*/"table_1",
          /*feature_index=*/1, /*max_ids_per_partition=*/16,
          /*max_unique_ids_per_partition=*/16, /*row_offset=*/8,
          /*col_offset=*/32, /*col_shift=*/0, /*batch_size=*/8)};

  std::vector<std::unique_ptr<AbstractInputBatch>> input_batches_multi_,
      input_batches_single_;

  void SetUp() override {
    input_batches_multi_.push_back(std::make_unique<InputBatch>(input_a_));
    input_batches_multi_.push_back(std::make_unique<InputBatch>(input_b_));

    input_batches_single_.push_back(std::make_unique<InputBatch>(input_c_));
    input_batches_single_.push_back(std::make_unique<InputBatch>(input_d_));
  }
};

TEST_F(TableStackingTest, MultiProcessStackingStackThenSplit) {
  PreprocessSparseDenseMatmulInputOptions options{
      .local_device_count = 1,
      .global_device_count = 2,
      .num_sc_per_device = 4,
      .feature_stacking_strategy = FeatureStackingStrategy::kStackThenSplit};

  ExtractedCooTensors extracted_coo_tensors =
      internal::ExtractCooTensorsForAllFeaturesPerLocalDevice(
          stacked_table_metadata_multi_, absl::MakeSpan(input_batches_multi_),
          /*local_device_id=*/0, options);

  EXPECT_EQ(extracted_coo_tensors.batch_size_for_device, 16);
  ASSERT_THAT(extracted_coo_tensors.coo_tensors, SizeIs(24));
  // This results in an uneven ID distribution - 8, 8, 4, 4

  std::vector<CooFormat> expected_coo_tensors;
  // Feature 0, slice 0
  // SC 0 (4 rows, 8 ids)
  expected_coo_tensors.push_back(CooFormat(0, 5, 1));
  expected_coo_tensors.push_back(CooFormat(0, 18, 1));
  expected_coo_tensors.push_back(CooFormat(1, 18, 1));
  expected_coo_tensors.push_back(CooFormat(1, 0, 1));
  expected_coo_tensors.push_back(CooFormat(2, 0, 1));
  expected_coo_tensors.push_back(CooFormat(2, 20, 1));
  expected_coo_tensors.push_back(CooFormat(3, 18, 1));
  expected_coo_tensors.push_back(CooFormat(3, 0, 1));
  // SC 1 (4 rows, 8 ids)
  expected_coo_tensors.push_back(CooFormat(4, 18, 1));
  expected_coo_tensors.push_back(CooFormat(4, 0, 1));
  expected_coo_tensors.push_back(CooFormat(5, 0, 1));
  expected_coo_tensors.push_back(CooFormat(5, 20, 1));
  expected_coo_tensors.push_back(CooFormat(6, 5, 1));
  expected_coo_tensors.push_back(CooFormat(6, 18, 1));
  expected_coo_tensors.push_back(CooFormat(7, 18, 1));
  expected_coo_tensors.push_back(CooFormat(7, 0, 1));

  // Feature 1, slice 0
  // SC 2 (4 rows, 4 ids)
  expected_coo_tensors.push_back(CooFormat(8, 34, 1));
  expected_coo_tensors.push_back(CooFormat(9, 42, 1));
  expected_coo_tensors.push_back(CooFormat(10, 33, 1));
  expected_coo_tensors.push_back(CooFormat(11, 41, 1));
  // SC 3 (4 rows, 4 ids)
  expected_coo_tensors.push_back(CooFormat(12, 35, 1));
  expected_coo_tensors.push_back(CooFormat(13, 39, 1));
  expected_coo_tensors.push_back(CooFormat(14, 36, 1));
  expected_coo_tensors.push_back(CooFormat(15, 40, 1));

  EXPECT_THAT(extracted_coo_tensors.coo_tensors,
              ElementsAreArray(expected_coo_tensors));
}

TEST_F(TableStackingTest, MultiProcessStackingSplitThenStack) {
  PreprocessSparseDenseMatmulInputOptions options{
      .local_device_count = 1,
      .global_device_count = 2,
      .num_sc_per_device = 4,
      .feature_stacking_strategy = FeatureStackingStrategy::kSplitThenStack};

  ExtractedCooTensors extracted_coo_tensors =
      internal::ExtractCooTensorsForAllFeaturesPerLocalDevice(
          stacked_table_metadata_multi_, absl::MakeSpan(input_batches_multi_),
          /*local_device_id=*/0, options);

  EXPECT_EQ(extracted_coo_tensors.batch_size_for_device, 16);
  ASSERT_THAT(extracted_coo_tensors.coo_tensors, SizeIs(24));
  // This results in a more even distribution (actually ideal) - 6,6,6,6

  std::vector<CooFormat> expected_coo_tensors;

  // SC 0 (4 rows, 6 ids)
  // Feature 0, slice 0
  expected_coo_tensors.push_back(CooFormat(0, 5, 1));
  expected_coo_tensors.push_back(CooFormat(0, 18, 1));
  expected_coo_tensors.push_back(CooFormat(1, 18, 1));
  expected_coo_tensors.push_back(CooFormat(1, 0, 1));

  // Feature 1, slice 0
  expected_coo_tensors.push_back(CooFormat(2, 34, 1));
  expected_coo_tensors.push_back(CooFormat(3, 42, 1));

  // SC 1 (4 rows, 6 ids)
  // Feature 0, slice 1
  expected_coo_tensors.push_back(CooFormat(4, 0, 1));
  expected_coo_tensors.push_back(CooFormat(4, 20, 1));
  expected_coo_tensors.push_back(CooFormat(5, 18, 1));
  expected_coo_tensors.push_back(CooFormat(5, 0, 1));

  // Feature 1, slice 1
  expected_coo_tensors.push_back(CooFormat(6, 33, 1));
  expected_coo_tensors.push_back(CooFormat(7, 41, 1));

  // SC 2 (4 rows, 6 ids)
  // Feature 0, slice 2
  expected_coo_tensors.push_back(CooFormat(8, 18, 1));
  expected_coo_tensors.push_back(CooFormat(8, 0, 1));
  expected_coo_tensors.push_back(CooFormat(9, 0, 1));
  expected_coo_tensors.push_back(CooFormat(9, 20, 1));

  // Feature 1, slice 2
  expected_coo_tensors.push_back(CooFormat(10, 35, 1));
  expected_coo_tensors.push_back(CooFormat(11, 39, 1));
  // SC 3 (4 rows, 6 ids)
  // Feature 0, slice 3
  expected_coo_tensors.push_back(CooFormat(12, 5, 1));
  expected_coo_tensors.push_back(CooFormat(12, 18, 1));
  expected_coo_tensors.push_back(CooFormat(13, 18, 1));
  expected_coo_tensors.push_back(CooFormat(13, 0, 1));

  // Feature 1, slice 3
  expected_coo_tensors.push_back(CooFormat(14, 36, 1));
  expected_coo_tensors.push_back(CooFormat(15, 40, 1));

  EXPECT_THAT(extracted_coo_tensors.coo_tensors,
              ElementsAreArray(expected_coo_tensors));
}

TEST_F(TableStackingTest, SingleProcessSingleDeviceSplitThenStack) {
  PreprocessSparseDenseMatmulInputOptions options{
      .local_device_count = 1,
      .global_device_count = 1,
      .num_sc_per_device = 4,
      .feature_stacking_strategy = FeatureStackingStrategy::kSplitThenStack};

  ExtractedCooTensors extracted_coo_tensors =
      internal::ExtractCooTensorsForAllFeaturesPerLocalDevice(
          stacked_table_metadata_single_, absl::MakeSpan(input_batches_single_),
          /*local_device_id=*/0, options);

  EXPECT_EQ(extracted_coo_tensors.batch_size_for_device, 16);
  ASSERT_THAT(extracted_coo_tensors.coo_tensors, SizeIs(16 * 17 / 2));

  const int batch_size_per_sc =
      extracted_coo_tensors.batch_size_for_device / options.num_sc_per_device;
  std::vector<int> ids_per_sc(options.num_sc_per_device, 0);
  for (const auto& coo_tensor : extracted_coo_tensors.coo_tensors) {
    ids_per_sc[coo_tensor.row_id / batch_size_per_sc]++;
  }

  std::vector<int> expected_ids_per_sc = {1 + 2 + 9 + 10, 3 + 4 + 11 + 12,
                                          5 + 6 + 13 + 14, 7 + 8 + 15 + 16};

  EXPECT_EQ(ids_per_sc, expected_ids_per_sc);
}

TEST_F(TableStackingTest, SingleProcessSingleDeviceStackThenSplit) {
  PreprocessSparseDenseMatmulInputOptions options{
      .local_device_count = 1,
      .global_device_count = 1,
      .num_sc_per_device = 4,
      .feature_stacking_strategy = FeatureStackingStrategy::kStackThenSplit};

  ExtractedCooTensors extracted_coo_tensors =
      internal::ExtractCooTensorsForAllFeaturesPerLocalDevice(
          stacked_table_metadata_single_, absl::MakeSpan(input_batches_single_),
          /*local_device_id=*/0, options);

  EXPECT_EQ(extracted_coo_tensors.batch_size_for_device, 16);
  ASSERT_THAT(extracted_coo_tensors.coo_tensors, SizeIs(16 * 17 / 2));

  const int batch_size_per_sc =
      extracted_coo_tensors.batch_size_for_device / options.num_sc_per_device;
  std::vector<int> ids_per_sc(options.num_sc_per_device, 0);
  for (const auto& coo_tensor : extracted_coo_tensors.coo_tensors) {
    ids_per_sc[coo_tensor.row_id / batch_size_per_sc]++;
  }

  std::vector<int> expected_ids_per_sc = {1 + 2 + 3 + 4,     //
                                          5 + 6 + 7 + 8,     //
                                          9 + 10 + 11 + 12,  //
                                          13 + 14 + 15 + 16};

  EXPECT_EQ(ids_per_sc, expected_ids_per_sc);
}

TEST_F(TableStackingTest, MultiChipSplitThenStack) {
  PreprocessSparseDenseMatmulInputOptions options{
      .local_device_count = 2,
      .global_device_count = 2,
      .num_sc_per_device = 4,
      .feature_stacking_strategy = FeatureStackingStrategy::kSplitThenStack};

  std::vector<int> expected_ids_per_sc[] = {{1 + 9, 2 + 10, 3 + 11, 4 + 12},
                                            {5 + 13, 6 + 14, 7 + 15, 8 + 16}};

  for (int local_device_id = 0; local_device_id < options.local_device_count;
       ++local_device_id) {
    ExtractedCooTensors extracted_coo_tensors =
        internal::ExtractCooTensorsForAllFeaturesPerLocalDevice(
            stacked_table_metadata_single_,
            absl::MakeSpan(input_batches_single_), local_device_id, options);
    EXPECT_EQ(extracted_coo_tensors.batch_size_for_device, 8);

    const int batch_size_per_sc =
        extracted_coo_tensors.batch_size_for_device / options.num_sc_per_device;
    std::vector<int> ids_per_sc(options.num_sc_per_device, 0);
    for (const auto& coo_tensor : extracted_coo_tensors.coo_tensors) {
      ids_per_sc[coo_tensor.row_id / batch_size_per_sc]++;
    }

    EXPECT_EQ(ids_per_sc, expected_ids_per_sc[local_device_id])
        << "local_device_id: " << local_device_id;
  }
}

TEST_F(TableStackingTest, MultiChipStackThenSplit) {
  PreprocessSparseDenseMatmulInputOptions options{
      .local_device_count = 2,
      .global_device_count = 2,
      .num_sc_per_device = 4,
      .feature_stacking_strategy = FeatureStackingStrategy::kStackThenSplit};

  std::vector<int> expected_ids_per_sc[] = {{1 + 2, 3 + 4, 9 + 10, 11 + 12},
                                            {5 + 6, 7 + 8, 13 + 14, 15 + 16}};

  for (int local_device_id = 0; local_device_id < options.local_device_count;
       ++local_device_id) {
    ExtractedCooTensors extracted_coo_tensors =
        internal::ExtractCooTensorsForAllFeaturesPerLocalDevice(
            stacked_table_metadata_single_,
            absl::MakeSpan(input_batches_single_), local_device_id, options);

    EXPECT_EQ(extracted_coo_tensors.batch_size_for_device, 8);

    const int batch_size_per_sc =
        extracted_coo_tensors.batch_size_for_device / options.num_sc_per_device;
    std::vector<int> ids_per_sc(options.num_sc_per_device, 0);
    for (const auto& coo_tensor : extracted_coo_tensors.coo_tensors) {
      ids_per_sc[coo_tensor.row_id / batch_size_per_sc]++;
    }

    EXPECT_EQ(ids_per_sc, expected_ids_per_sc[local_device_id])
        << "local_device_id: " << local_device_id;
  }
}

TEST(InputPreprocessingUtilTest, MergeStats) {
  SparseDenseMatmulInputStats stats1;
  SparseDenseMatmulInputStats stats2;

  // Populate stats1
  stats1.max_ids_per_partition["table_0"].resize(4);
  stats1.max_ids_per_partition["table_0"] << 10, 20, 30, 40;
  stats1.max_ids_per_partition["table_1"].resize(4);
  stats1.max_ids_per_partition["table_1"] << 1, 2, 3, 4;
  stats1.max_unique_ids_per_partition["table_0"].resize(4);
  stats1.max_unique_ids_per_partition["table_0"] << 5, 6, 7, 8;
  stats1.max_unique_ids_per_partition["table_1"].resize(4);
  stats1.max_unique_ids_per_partition["table_1"] << 10, 11, 12, 13;
  stats1.required_buffer_sizes["table_0"].resize(4);
  stats1.required_buffer_sizes["table_0"] << 100, 101, 102, 103;
  stats1.required_buffer_sizes["table_1"].resize(4);
  stats1.required_buffer_sizes["table_1"] << 200, 201, 202, 203;

  // Populate stats2
  stats2.max_ids_per_partition["table_0"].resize(4);
  stats2.max_ids_per_partition["table_0"] << 12, 13, 14, 15;
  stats2.max_ids_per_partition["table_2"].resize(4);
  stats2.max_ids_per_partition["table_2"] << 30, 31, 32, 33;
  stats2.max_unique_ids_per_partition["table_0"].resize(4);
  stats2.max_unique_ids_per_partition["table_0"] << 6, 7, 8, 9;
  stats2.max_unique_ids_per_partition["table_2"].resize(4);
  stats2.max_unique_ids_per_partition["table_2"] << 12, 13, 14, 15;
  stats2.required_buffer_sizes["table_0"].resize(4);
  stats2.required_buffer_sizes["table_0"] << 120, 121, 122, 123;
  stats2.required_buffer_sizes["table_2"].resize(4);
  stats2.required_buffer_sizes["table_2"] << 300, 301, 302, 303;

  stats1.merge(stats2);

  // The merged stats should contain the element-wise maximum of the
  // corresponding stats.
  EXPECT_THAT(stats1.max_ids_per_partition["table_0"],
              ElementsAreArray({12, 20, 30, 40}));
  EXPECT_THAT(stats1.max_ids_per_partition["table_1"],
              ElementsAreArray({1, 2, 3, 4}));
  EXPECT_THAT(stats1.max_ids_per_partition["table_2"],
              ElementsAreArray({30, 31, 32, 33}));

  EXPECT_THAT(stats1.max_unique_ids_per_partition["table_0"],
              ElementsAreArray({6, 7, 8, 9}));
  EXPECT_THAT(stats1.max_unique_ids_per_partition["table_1"],
              ElementsAreArray({10, 11, 12, 13}));
  EXPECT_THAT(stats1.max_unique_ids_per_partition["table_2"],
              ElementsAreArray({12, 13, 14, 15}));

  EXPECT_THAT(stats1.required_buffer_sizes["table_0"],
              ElementsAreArray({120, 121, 122, 123}));
  EXPECT_THAT(stats1.required_buffer_sizes["table_1"],
              ElementsAreArray({200, 201, 202, 203}));
  EXPECT_THAT(stats1.required_buffer_sizes["table_2"],
              ElementsAreArray({300, 301, 302, 303}));
}

TEST_F(TableStackingTest, CooTensorsPerScCalculation) {
  PreprocessSparseDenseMatmulInputOptions options{
      .local_device_count = 1,
      .global_device_count = 1,
      .num_sc_per_device = 4,
      .feature_stacking_strategy = FeatureStackingStrategy::kSplitThenStack};

  ExtractedCooTensors extracted_coo_tensors =
      internal::ExtractCooTensorsForAllFeaturesPerLocalDevice(
          stacked_table_metadata_single_, absl::MakeSpan(input_batches_single_),
          /*local_device_id=*/0, options);

  EXPECT_EQ(extracted_coo_tensors.batch_size_for_device, 16);
  ASSERT_THAT(extracted_coo_tensors.coo_tensors, SizeIs(16 * 17 / 2));

  std::vector<int> expected_coo_tensors_per_sc = {
      1 + 2 + 9 + 10, 3 + 4 + 11 + 12, 5 + 6 + 13 + 14, 7 + 8 + 15 + 16};
  EXPECT_EQ(extracted_coo_tensors.coo_tensors_per_sc,
            expected_coo_tensors_per_sc);
}

class MinibatchingTest : public testing::TestWithParam<bool> {
 protected:
  bool IsMinibatchingEnabled() const { return GetParam(); }

  std::vector<int> GetBucketIds() {
    return IsMinibatchingEnabled() ? std::vector<int>{0, 1, 2}
                                   : std::vector<int>{0};
  }

  // Helper to generate keys in the expected order
  std::vector<uint64_t> GenerateGroupingKeys() {
    const int num_scs = 4;
    const int num_scs_bit = std::log2(num_scs);
    std::vector<uint64_t> keys;
    int index = 0;

    for (int bucket_id : GetBucketIds()) {
      for (int global_sc_id : {0, 1, 2}) {
        for (int local_embedding_id : {5, 13, 17}) {
          const int col_shift = global_sc_id;
          const int col_offset = local_embedding_id * num_scs;
          CooFormat coo_format(
              /*sample_id=*/0,
              /*embedding_id=*/local_embedding_id,  // Use local_embedding_id
              /*gain=*/1.0,
              /*col_shift=*/col_shift,
              /*col_offset=*/col_offset,
              /*num_scs_mod=*/3);

          const auto hash_fn = [=](int col_id) { return bucket_id; };

          keys.push_back(coo_format.GetGroupingKey(
              num_scs_bit, index, IsMinibatchingEnabled(), hash_fn));
          ++index;
        }
      }
    }
    return keys;
  }
};

TEST(CooFormatTest, BucketIdCalculationIsCorrect) {
  CooFormat coo_format(/*sample_id=*/1, /*embedding_id=*/70, /*gain=*/1.0,
                       /*col_shift=*/0, /*col_offset=*/0, /*num_scs_mod=*/3);
  EXPECT_EQ(coo_format.col_id, 70);
  EXPECT_EQ(coo_format.GetBucketId(), 70 % CooFormat::kMaxMinibatchingBuckets);

  CooFormat coo_format_2(/*sample_id=*/2, /*embedding_id=*/127, /*gain=*/0.5,
                         /*col_shift=*/1, /*col_offset=*/32, /*num_scs_mod=*/3);

  // 128%4 + 127//4*4 + 32 = 156
  EXPECT_EQ(coo_format_2.col_id, 156);
  EXPECT_EQ(coo_format_2.GetBucketId(),
            156 % CooFormat::kMaxMinibatchingBuckets);
}

// NOTE: We do not want to test for exact key value, since we only care about
// the ordering.
TEST_P(MinibatchingTest, KeysAreSorted) {
  std::vector<uint64_t> keys = GenerateGroupingKeys();

  for (int i = 0; i + 1 < keys.size(); ++i) {
    EXPECT_LT(keys[i], keys[i + 1]) << i;
  }
}

TEST_P(MinibatchingTest, IndexFromKeyIsCorrect) {
  std::vector<uint64_t> keys = GenerateGroupingKeys();
  for (int i = 0; i < keys.size(); ++i) {
    EXPECT_EQ(keys[i] & CooFormat::kIndexMask, i);
  }
}

INSTANTIATE_TEST_SUITE_P(
    MinibatchingTestGroup, MinibatchingTest, testing::Bool(),
    [](const testing::TestParamInfo<MinibatchingTest::ParamType>& info) {
      return info.param ? "MinibatchingEnabled" : "MinibatchingDisabled";
    });

class MinibatchingCountTest : public ::testing::Test {
 protected:
  std::vector<StackedTableMetadata> stacked_table_metadata_{
      StackedTableMetadata(
          /*name=*/"table_0",
          /*feature_index=*/0, /*max_ids_per_partition=*/16,
          /*max_unique_ids_per_partition=*/16, /*row_offset=*/0,
          /*col_offset=*/0, /*col_shift=*/0, /*batch_size=*/16),
      StackedTableMetadata(
          /*name=*/"table_1",
          /*feature_index=*/1, /*max_ids_per_partition=*/16,
          /*max_unique_ids_per_partition=*/16, /*row_offset=*/16,
          /*col_offset=*/32, /*col_shift=*/0, /*batch_size=*/16)};

  // Helper function to create input batches for testing.
  // It generates a sequence of ids based on `max_ids_per_partitions` and
  // `max_unique_ids_per_partitions` for each table.
  std::vector<std::unique_ptr<AbstractInputBatch>> CreateInputBatches(
      absl::Span<const int> max_ids_per_partitions,
      absl::Span<const int> max_unique_ids_per_partitions, int global_sc_count,
      int local_sc_count) {
    std::vector<std::unique_ptr<AbstractInputBatch>> input_batches;
    CHECK_EQ(max_ids_per_partitions.size(),
             max_unique_ids_per_partitions.size());
    CHECK_EQ(max_ids_per_partitions.size(), stacked_table_metadata_.size());
    for (int table_id = 0; table_id < max_ids_per_partitions.size();
         ++table_id) {
      const int batch_size = stacked_table_metadata_[table_id].batch_size;
      const int batch_size_per_sc = batch_size / local_sc_count;
      std::vector<int> ids;
      std::vector<int> row_offsets;
      row_offsets.push_back(0);
      int curr_row_offset = 0;
      for (int local_sc_id = 0; local_sc_id < local_sc_count; ++local_sc_id) {
        CHECK_GE(max_ids_per_partitions[table_id],
                 max_unique_ids_per_partitions[table_id]);
        // Example: max ids = 5, max unique ids = 3 => {0,0,0,1,2}
        // 0 is non-unique (0,0).
        for (int i = 0; i < max_ids_per_partitions[table_id] -
                                max_unique_ids_per_partitions[table_id];
             ++i) {
          ids.push_back(i *
                        global_sc_count);  // embedding id = i, global id = 0.
        }
        // others are unique (0,1,2).
        for (int j = 0; j < max_unique_ids_per_partitions[table_id]; ++j) {
          ids.push_back(j);
        }
      }
      const int ids_per_sc = ids.size() / local_sc_count;
      const int ids_per_row = ids_per_sc / batch_size_per_sc;
      CHECK_GT(ids_per_row, 0);
      for (int local_sc_id = 0; local_sc_id < local_sc_count; ++local_sc_id) {
        for (int row_id = 0; row_id < batch_size_per_sc - 1; ++row_id) {
          curr_row_offset += ids_per_row;
          row_offsets.push_back(curr_row_offset);
        }
        curr_row_offset = (local_sc_id + 1) * ids_per_sc;
        row_offsets.push_back(curr_row_offset);
      }
      CHECK_EQ(row_offsets.back(), ids.size());

      input_batches.push_back(
          std::make_unique<
              RaggedTensorInputBatch<std::vector<int>, std::vector<int>>>(
              ids, row_offsets));
    }
    return input_batches;
  }

  static constexpr int kHosts = 2;

  // Helper to get a static thread pool shared across tests that simulates
  // a multi-host environment.
  static tsl::thread::ThreadPool* MultiHostPool() {
    static tsl::thread::ThreadPool* pool = new tsl::thread::ThreadPool(
        tsl::Env::Default(), "MultiHostPool", /*num_threads=*/kHosts);
    return pool;
  }
};

TEST_F(MinibatchingCountTest,
       SingleHostMinibatchCountIsCorrectWhenNotRequired) {
  // Arrange
  PreprocessSparseDenseMatmulInputOptions options{.local_device_count = 1,
                                                  .global_device_count = 1,
                                                  .num_sc_per_device = 4,
                                                  .enable_minibatching = true};

  std::vector<std::unique_ptr<AbstractInputBatch>> input_batches =
      CreateInputBatches(/*max_ids_per_partitions=*/{4, 6},
                         /*max_unique_ids_per_partitions=*/{4, 6},
                         /*global_sc_count=*/4,
                         /*local_sc_count=*/4);

  absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>
      stacked_tables({{"table_0", stacked_table_metadata_}});

  // Act
  PreprocessSparseDenseMatmulOutput output = PreprocessSparseDenseMatmulInput(
      absl::MakeSpan(input_batches), stacked_tables, options);

  // Assert
  EXPECT_EQ(output.num_minibatches, 1);
}

TEST_F(MinibatchingCountTest, SingleHostMinibatchCountIsCorrectWhenRequired) {
  // Arrange
  PreprocessSparseDenseMatmulInputOptions options{.local_device_count = 1,
                                                  .global_device_count = 1,
                                                  .num_sc_per_device = 4,
                                                  .enable_minibatching = true};

  // Reduce max ids and max unique ids to trigger minibatching.
  // Also increase buffer size.
  stacked_table_metadata_[0].max_ids_per_partition = 5;
  stacked_table_metadata_[0].max_unique_ids_per_partition = 2;
  stacked_table_metadata_[0].suggested_coo_buffer_size_per_device = 2048;

  stacked_table_metadata_[1].max_ids_per_partition = 5;
  stacked_table_metadata_[1].max_unique_ids_per_partition = 6;
  stacked_table_metadata_[1].suggested_coo_buffer_size_per_device = 2048;

  CreateInputBatches(/*max_ids_per_partitions=*/{10, 20},
                     /*max_unique_ids_per_partitions=*/{5, 10},
                     /*global_sc_count=*/4,
                     /*local_sc_count=*/4);
  std::vector<std::unique_ptr<AbstractInputBatch>> input_batches =
      CreateInputBatches(/*max_ids_per_partitions=*/{10, 20},
                         /*max_unique_ids_per_partitions=*/{5, 10},
                         /*global_sc_count=*/4,
                         /*local_sc_count=*/4);

  absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>
      stacked_tables({{"table_0", stacked_table_metadata_}});

  // Act
  PreprocessSparseDenseMatmulOutput output = PreprocessSparseDenseMatmulInput(
      absl::MakeSpan(input_batches), stacked_tables, options);

  // Assert
  EXPECT_GT(output.num_minibatches, 1);
  EXPECT_EQ(output.num_minibatches, 8);
}

TEST_F(MinibatchingCountTest, MultiHostMinibatchCountIsCorrectWhenNotRequired) {
  // Arrange
  absl::BlockingCounter counter(kHosts);

  absl::Mutex mutex;
  std::vector<int> minibatches_per_host(kHosts, -1);

  absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>
      stacked_tables({{"table_0", stacked_table_metadata_}});

  auto input_batch_host0 =
      CreateInputBatches(/*max_ids_per_partitions=*/{4, 6},
                         /*max_unique_ids_per_partitions=*/{4, 6},
                         /*global_sc_count=*/8, /*local_sc_count=*/4);
  auto input_batch_host1 =
      CreateInputBatches(/*max_ids_per_partitions=*/{8, 12},
                         /*max_unique_ids_per_partitions=*/{8, 12},
                         /*global_sc_count=*/8, /*local_sc_count=*/4);
  std::vector<std::vector<std::unique_ptr<AbstractInputBatch>>*> input_batches =
      {&input_batch_host0, &input_batch_host1};

  PreprocessSparseDenseMatmulInputOptions options{.local_device_count = 1,
                                                  .global_device_count = 2,
                                                  .num_sc_per_device = 4,
                                                  .enable_minibatching = true};

  // Act
  for (int host_id = 0; host_id < kHosts; ++host_id) {
    MultiHostPool()->Schedule([&, host_id]() {
      PreprocessSparseDenseMatmulOutput output =
          PreprocessSparseDenseMatmulInput(
              absl::MakeSpan(*input_batches[host_id]), stacked_tables, options);
      {
        absl::MutexLock lock(&mutex);
        minibatches_per_host[host_id] = output.num_minibatches;
      }
      counter.DecrementCount();
    });
  }
  counter.Wait();

  // Assert
  EXPECT_THAT(minibatches_per_host, Each(Eq(1)));
}

// TODO: b/428790659 - This test is disabled because it requires multi-host
//   collectives which is not yet implemented.
TEST_F(MinibatchingCountTest,
       DISABLED_MultiHostMinibatchCountIsCorrectWhenRequired) {
  // Arrange
  absl::BlockingCounter counter(kHosts);

  absl::Mutex mutex;
  std::vector<int> minibatches_per_host(kHosts, -1);

  absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>
      stacked_tables({{"table_0", stacked_table_metadata_}});

  auto input_batch_host0 =
      CreateInputBatches(/*max_ids_per_partitions=*/{80, 120},
                         /*max_unique_ids_per_partitions=*/{48, 50},
                         /*global_sc_count=*/8, /*local_sc_count=*/4);
  auto input_batch_host1 =
      CreateInputBatches(/*max_ids_per_partitions=*/{40, 200},
                         /*max_unique_ids_per_partitions=*/{30, 64},
                         /*global_sc_count=*/8, /*local_sc_count=*/4);
  std::vector<std::vector<std::unique_ptr<AbstractInputBatch>>*> input_batches =
      {&input_batch_host0, &input_batch_host1};

  PreprocessSparseDenseMatmulInputOptions options{.local_device_count = 1,
                                                  .global_device_count = 2,
                                                  .num_sc_per_device = 4,
                                                  .enable_minibatching = true};

  // Act
  for (int host_id = 0; host_id < kHosts; ++host_id) {
    MultiHostPool()->Schedule([&, host_id]() {
      PreprocessSparseDenseMatmulOutput output =
          PreprocessSparseDenseMatmulInput(
              absl::MakeSpan(*input_batches[host_id]), stacked_tables, options);
      {
        absl::MutexLock lock(&mutex);
        minibatches_per_host[host_id] = output.num_minibatches;
      }
      counter.DecrementCount();
    });
  }
  counter.Wait();

  // Assert
  EXPECT_THAT(minibatches_per_host, Each(Gt(1)));
  EXPECT_THAT(minibatches_per_host, Each(Eq(32)));
}
}  // namespace
}  // namespace jax_sc_embedding
