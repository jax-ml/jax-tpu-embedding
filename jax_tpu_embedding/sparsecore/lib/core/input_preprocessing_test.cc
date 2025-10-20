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

#include <climits>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "fuzztest/fuzztest.h"
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/synchronization/blocking_counter.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "Eigen/Core"  // from @eigen_archive
#include "jax_tpu_embedding/sparsecore/lib/core/abstract_input_batch.h"
#include "jax_tpu_embedding/sparsecore/lib/core/all_reduce_interface.h"
#include "jax_tpu_embedding/sparsecore/lib/core/coo_format.h"
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/minibatching_node.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "jax_tpu_embedding/sparsecore/lib/core/minibatching_test_utils.h"
#include "jax_tpu_embedding/sparsecore/lib/core/ragged_tensor_input_batch.h"
#include "tsl/platform/env.h"  // from @tsl
#include "tsl/platform/statusor.h"  // from @tsl
#include "tsl/platform/threadpool.h"  // from @tsl

namespace jax_sc_embedding {
namespace {

using ::absl_testing::IsOk;
using ::jax_sc_embedding::testing_utils::SetUpMinibatchingNodes;
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

namespace testing_utils {

// Wrapper around an AllReduceInterface that causes a test failure if
// BlockingAllReduce is called with an odd sync_key for uint64_t values.
class AllReduceNoMinibatchingRequired : public AllReduceInterface {
 public:
  explicit AllReduceNoMinibatchingRequired(AllReduceInterface* wrapped)
      : wrapped_(wrapped) {}

  absl::StatusOr<bool> BlockingAllReduce(int sync_key,
                                         bool minibatching_required) override {
    return wrapped_->BlockingAllReduce(sync_key, minibatching_required);
  }

  absl::StatusOr<uint64_t> BlockingAllReduce(
      int sync_key, uint64_t minibatching_split) override {
    if (sync_key % 2 == 1) {  // Odd keys are for splits.
      ADD_FAILURE() << "SyncMinibatchingSplit should not be called when no "
                       "host requires minibatching. Called with split: "
                    << minibatching_split;
    }
    return wrapped_->BlockingAllReduce(sync_key, minibatching_split);
  }

 private:
  AllReduceInterface* wrapped_;
};

// Wrapper around an AllReduceInterface that records all sync_key values used in
// calls to BlockingAllReduce.
class AllReduceSyncKeyCollector : public AllReduceInterface {
 public:
  explicit AllReduceSyncKeyCollector(AllReduceInterface* wrapped)
      : wrapped_(wrapped) {}

  absl::StatusOr<bool> BlockingAllReduce(int sync_key,
                                         bool minibatching_required) override {
    {
      absl::MutexLock lock(mutex_);
      sync_keys_.push_back(sync_key);
    }
    return wrapped_->BlockingAllReduce(sync_key, minibatching_required);
  }

  absl::StatusOr<uint64_t> BlockingAllReduce(
      int sync_key, uint64_t minibatching_split) override {
    {
      absl::MutexLock lock(mutex_);
      sync_keys_.push_back(sync_key);
    }
    return wrapped_->BlockingAllReduce(sync_key, minibatching_split);
  }

  std::vector<int> GetSyncKeys() {
    absl::MutexLock lock(mutex_);
    return std::vector<int>(sync_keys_);
  }

 private:
  AllReduceInterface* wrapped_;
  absl::Mutex mutex_;
  std::vector<int> sync_keys_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace testing_utils

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
  EXPECT_EQ(coo_format.GetBucketId(),
            HighwayHash(70) % CooFormat::kMaxMinibatchingBuckets);

  CooFormat coo_format_2(/*sample_id=*/2, /*embedding_id=*/127, /*gain=*/0.5,
                         /*col_shift=*/1, /*col_offset=*/32, /*num_scs_mod=*/3);

  // 128%4 + 127//4*4 + 32 = 156
  EXPECT_EQ(coo_format_2.col_id, 156);
  EXPECT_EQ(coo_format_2.GetBucketId(),
            HighwayHash(156) % CooFormat::kMaxMinibatchingBuckets);
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
  TF_ASSERT_OK_AND_ASSIGN(
      PreprocessSparseDenseMatmulOutput output,
      PreprocessSparseDenseMatmulInput(absl::MakeSpan(input_batches),
                                       stacked_tables, options));

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

  stacked_table_metadata_[1].max_ids_per_partition = 5;
  stacked_table_metadata_[1].max_unique_ids_per_partition = 6;

  auto input_batches =
      CreateInputBatches(/*max_ids_per_partitions=*/{10, 20},
                         /*max_unique_ids_per_partitions=*/{5, 10},
                         /*global_sc_count=*/4,
                         /*local_sc_count=*/4);

  absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>
      stacked_tables({{"table_0", stacked_table_metadata_}});

  // Act
  TF_ASSERT_OK_AND_ASSIGN(
      PreprocessSparseDenseMatmulOutput output,
      PreprocessSparseDenseMatmulInput(absl::MakeSpan(input_batches),
                                       stacked_tables, options));

  // Assert
  EXPECT_GT(output.num_minibatches, 1);
}

TEST_F(MinibatchingCountTest, MultiHostMinibatchCountIsCorrectWhenNotRequired) {
  // Arrange
  absl::BlockingCounter counter(kHosts);

  absl::Mutex mutex;
  std::vector<int> minibatches_per_host(kHosts, -1);

  absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>
      stacked_tables({{"table_0", stacked_table_metadata_}});

  auto nodes = SetUpMinibatchingNodes(kHosts);
  std::vector<testing_utils::AllReduceNoMinibatchingRequired> all_reducers;
  for (int i = 0; i < kHosts; ++i) {
    all_reducers.emplace_back(nodes[i]->GetAllReduceInterface());
  }

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

  // Act
  for (int host_id = 0; host_id < kHosts; ++host_id) {
    MultiHostPool()->Schedule([&, host_id]() {
      PreprocessSparseDenseMatmulInputOptions options{
          .local_device_count = 1,
          .global_device_count = 2,
          .num_sc_per_device = 4,
          .enable_minibatching = true,
          .batch_number = 100,
          .all_reduce_interface = &all_reducers[host_id]};
      TF_ASSERT_OK_AND_ASSIGN(PreprocessSparseDenseMatmulOutput output,
                              PreprocessSparseDenseMatmulInput(
                                  absl::MakeSpan(*input_batches[host_id]),
                                  stacked_tables, options));
      {
        absl::MutexLock lock(mutex);
        minibatches_per_host[host_id] = output.num_minibatches;
      }
      counter.DecrementCount();
    });
  }
  counter.Wait();

  // Assert
  EXPECT_THAT(minibatches_per_host, Each(Eq(1)));
}

TEST_F(MinibatchingCountTest, MultiHostMinibatchCountIsCorrectWhenRequired) {
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

  auto nodes = SetUpMinibatchingNodes(kHosts);

  // Act
  for (int host_id = 0; host_id < kHosts; ++host_id) {
    MultiHostPool()->Schedule([&, host_id]() {
      PreprocessSparseDenseMatmulInputOptions options{
          .local_device_count = 1,
          .global_device_count = 2,
          .num_sc_per_device = 4,
          .enable_minibatching = true,
          .all_reduce_interface = nodes[host_id]->GetAllReduceInterface()};
      TF_ASSERT_OK_AND_ASSIGN(PreprocessSparseDenseMatmulOutput output,
                              PreprocessSparseDenseMatmulInput(
                                  absl::MakeSpan(*input_batches[host_id]),
                                  stacked_tables, options));
      {
        absl::MutexLock lock(mutex);
        minibatches_per_host[host_id] = output.num_minibatches;
      }
      counter.DecrementCount();
    });
  }
  counter.Wait();

  // Assert
  EXPECT_THAT(minibatches_per_host, Each(Gt(1)));
}

TEST_F(MinibatchingCountTest, MultiHostMinibatchCountIsCorrectWhenOneRequires) {
  // Arrange
  absl::BlockingCounter counter(kHosts);

  absl::Mutex mutex;
  std::vector<int> minibatches_per_host(kHosts, -1);

  absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>
      stacked_tables({{"table_0", stacked_table_metadata_}});

  // Host 0: Requires minibatching
  auto input_batch_host0 =
      CreateInputBatches(/*max_ids_per_partitions=*/{80, 120},
                         /*max_unique_ids_per_partitions=*/{48, 50},
                         /*global_sc_count=*/8, /*local_sc_count=*/4);
  // Host 1: Does not require minibatching
  auto input_batch_host1 =
      CreateInputBatches(/*max_ids_per_partitions=*/{4, 6},
                         /*max_unique_ids_per_partitions=*/{4, 6},
                         /*global_sc_count=*/8, /*local_sc_count=*/4);
  std::vector<std::vector<std::unique_ptr<AbstractInputBatch>>*> input_batches =
      {&input_batch_host0, &input_batch_host1};

  auto nodes = SetUpMinibatchingNodes(kHosts);

  // Act
  for (int host_id = 0; host_id < kHosts; ++host_id) {
    MultiHostPool()->Schedule([&, host_id]() {
      PreprocessSparseDenseMatmulInputOptions options{
          .local_device_count = 1,
          .global_device_count = 2,
          .num_sc_per_device = 4,
          .enable_minibatching = true,
          .all_reduce_interface = nodes[host_id]->GetAllReduceInterface()};
      TF_ASSERT_OK_AND_ASSIGN(PreprocessSparseDenseMatmulOutput output,
                              PreprocessSparseDenseMatmulInput(
                                  absl::MakeSpan(*input_batches[host_id]),
                                  stacked_tables, options));
      {
        absl::MutexLock lock(mutex);
        minibatches_per_host[host_id] = output.num_minibatches;
      }
      counter.DecrementCount();
    });
  }
  counter.Wait();

  // Assert
  // Both hosts should agree on the number of minibatches, which should be > 1
  // because host 0 required it.
  EXPECT_THAT(minibatches_per_host, Each(Gt(1)));
  EXPECT_THAT(minibatches_per_host, Each(Eq(minibatches_per_host[0])));
}

TEST_F(MinibatchingCountTest, MinibatchSyncKeysAreDisjoint) {
  // Arrange
  absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>
      stacked_tables({{"table_0", stacked_table_metadata_}});

  auto nodes = SetUpMinibatchingNodes(1);
  auto all_reduce = std::make_unique<testing_utils::AllReduceSyncKeyCollector>(
      nodes[0]->GetAllReduceInterface());
  const int kBatchCount = 10;

  // Act
  for (int batch_num = 0; batch_num < kBatchCount; ++batch_num) {
    PreprocessSparseDenseMatmulInputOptions options{
        .local_device_count = 1,
        .global_device_count = 1,
        .num_sc_per_device = 4,
        .enable_minibatching = true,
        .batch_number = batch_num,
        .all_reduce_interface = all_reduce.get()};
    auto input_batches =
        CreateInputBatches(/*max_ids_per_partitions=*/{80, 90},
                           /*max_unique_ids_per_partitions=*/{80, 90},
                           /*global_sc_count=*/4,
                           /*local_sc_count=*/4);
    ASSERT_THAT(PreprocessSparseDenseMatmulInput(absl::MakeSpan(input_batches),
                                                 stacked_tables, options),
                IsOk());
  }

  // Assert
  auto sync_keys = all_reduce->GetSyncKeys();
  auto sync_keys_set =
      absl::flat_hash_set<int>(sync_keys.begin(), sync_keys.end());
  EXPECT_THAT(sync_keys_set, SizeIs(kBatchCount * 2));
}

void ValidateCooId(int embedding_id, int sample_id, int64_t table_shard_size,
                   int batch_size_per_sc) {
  EXPECT_GE(embedding_id, 0);
  EXPECT_LE(embedding_id, table_shard_size);
  EXPECT_GE(sample_id, 0);
  EXPECT_LT(sample_id, batch_size_per_sc);
}

// Validates a slice of COO buffer data described by a sequence of row
// pointers.
// `row_pointers_slice`: A slice of row pointers for this validation.
// `embedding_ids_slice`: Slice of embedding_ids corresponding to this
// partition.
// `sample_ids_slice`: Slice of sample_ids corresponding to this partition.
void ValidateMinibatchOrSparseCoreSlice(
    const Eigen::Ref<const RowVectorXi>& row_pointers_slice,
    const Eigen::Ref<const RowVectorXi>& embedding_ids_slice,
    const Eigen::Ref<const RowVectorXi>& sample_ids_slice,
    int64_t table_shard_size, int batch_size_per_sc) {
  int32_t start_index = 0;
  for (int i = 0; i < row_pointers_slice.size(); ++i) {
    int end_index = row_pointers_slice(i);
    ASSERT_GE(end_index, start_index);
    ASSERT_LE(end_index, embedding_ids_slice.size());
    for (int j = start_index; j < end_index; ++j) {
      if (embedding_ids_slice(j) != INT_MAX) {
        ValidateCooId(embedding_ids_slice(j), sample_ids_slice(j),
                      table_shard_size, batch_size_per_sc);
      }
    }
    start_index = RoundUpTo(end_index, TPU_VECTOR_REGISTER_ALIGMENT_SIZE);
  }
}

void PreprocessingOutputIsValid(
    std::tuple<
        std::vector<std::vector<int64_t>>, std::vector<std::vector<int64_t>>,
        std::vector<std::vector<int64_t>>, std::vector<std::vector<int64_t>>,
        std::vector<std::vector<int64_t>>>
        samples,
    int num_sc_per_device, int global_device_count, int max_ids_per_partition,
    int max_unique_ids_per_partition,
    FeatureStackingStrategy feature_stacking_strategy,
    bool enable_minibatching) {
  auto create_input_batch =
      [](const std::vector<std::vector<int64_t>>& samples_in) {
        std::vector<int64_t> values;
        std::vector<int32_t> row_splits;
        row_splits.push_back(0);
        for (const auto& sample_ids : samples_in) {
          for (const auto& id : sample_ids) {
            values.push_back(id);
          }
          row_splits.push_back(values.size());
        }

        using InputBatch =
            RaggedTensorInputBatch<std::vector<int64_t>, std::vector<int32_t>>;
        return std::make_unique<InputBatch>(values, row_splits);
      };

  std::vector<std::unique_ptr<AbstractInputBatch>> input_batches;
  std::apply(
      [&](const auto&... table_samples) {
        (input_batches.push_back(create_input_batch(table_samples)), ...);
      },
      samples);

  const int kGlobalDeviceCount = global_device_count;
  const int kBatchSize = input_batches[0]->size();
  const int kNumScs = num_sc_per_device * kGlobalDeviceCount;

  const std::vector<int> kTableVocabs = {400000000, 400000000, 400000000,
                                         400000000, 100000000};

  auto round_up = [&](int value, int multiple) {
    return (value + multiple - 1) / multiple * multiple;
  };

  std::vector<StackedTableMetadata> stacked_table_metadata;
  int64_t current_col_offset = 0;
  for (int i = 0; i < kTableVocabs.size(); ++i) {
    stacked_table_metadata.push_back(StackedTableMetadata(
        /*name=*/absl::StrCat("table_", i),
        /*feature_index=*/i, max_ids_per_partition,
        max_unique_ids_per_partition,
        /*row_offset=*/i * kBatchSize * kGlobalDeviceCount,
        /*col_offset=*/static_cast<int>(current_col_offset),
        /*col_shift=*/i % 4, kBatchSize,
        /*suggested_coo_buffer_size_per_device=*/std::nullopt,
        RowCombiner::kSum, kTableVocabs[i] - 1));
    current_col_offset += round_up(kTableVocabs[i], 8 * kNumScs);
  }

  absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>
      stacked_tables({{"stacked_table", stacked_table_metadata}});

  PreprocessSparseDenseMatmulInputOptions options{
      .local_device_count = 1,
      .global_device_count = kGlobalDeviceCount,
      .num_sc_per_device = num_sc_per_device,
      .allow_id_dropping = true,
      .feature_stacking_strategy = feature_stacking_strategy,
      .enable_minibatching = enable_minibatching,
      .batch_number = 42};

  TF_ASSERT_OK_AND_ASSIGN(
      PreprocessSparseDenseMatmulOutput output,
      PreprocessSparseDenseMatmulInput(absl::MakeSpan(input_batches),
                                       stacked_tables, options));

  const int64_t total_padded_vocab = current_col_offset;
  const int64_t table_shard_size = total_padded_vocab / kNumScs;
  const MatrixXi& row_pointers = output.lhs_row_pointers.at("stacked_table");
  const MatrixXi& embedding_ids = output.lhs_embedding_ids.at("stacked_table");
  const MatrixXi& sample_ids = output.lhs_sample_ids.at("stacked_table");
  const int32_t num_minibatches = output.num_minibatches;
  const int32_t row_pointers_unpadded_size =
      num_minibatches * kNumScs * num_sc_per_device;
  const int batch_size_per_sc =
      CeilOfRatio(kBatchSize * kTableVocabs.size(), num_sc_per_device);

  if (options.enable_minibatching) {
    ValidateMinibatchOrSparseCoreSlice(
        row_pointers.row(0).head(row_pointers_unpadded_size),
        embedding_ids.row(0), sample_ids.row(0), table_shard_size,
        batch_size_per_sc);
  } else {
    const int coo_buffer_size_per_sc = embedding_ids.cols() / num_sc_per_device;
    const int row_pointers_size_per_bucket =
        row_pointers.cols() / num_sc_per_device;
    for (int sc_id = 0; sc_id < num_sc_per_device; ++sc_id) {
      ValidateMinibatchOrSparseCoreSlice(
          row_pointers.row(0).segment(sc_id * row_pointers_size_per_bucket,
                                      kNumScs),
          embedding_ids.row(0).segment(sc_id * coo_buffer_size_per_sc,
                                       coo_buffer_size_per_sc),
          sample_ids.row(0).segment(sc_id * coo_buffer_size_per_sc,
                                    coo_buffer_size_per_sc),
          table_shard_size, batch_size_per_sc);
    }
  }
}

FUZZ_TEST(InputPreprocessingFuzzTest, PreprocessingOutputIsValid)
    .WithDomains(
        fuzztest::TupleOf(
            /*samples[0]=*/
            fuzztest::VectorOf(
                fuzztest::VectorOf(fuzztest::InRange<int64_t>(0, 399999999))
                    .WithMaxSize(10))
                .WithSize(128),
            /*samples[1]=*/
            fuzztest::VectorOf(
                fuzztest::VectorOf(fuzztest::InRange<int64_t>(0, 399999999))
                    .WithMaxSize(5))
                .WithSize(128),
            /*samples[2]=*/
            fuzztest::VectorOf(
                fuzztest::VectorOf(fuzztest::InRange<int64_t>(0, 399999999))
                    .WithMaxSize(20))
                .WithSize(128),
            /*samples[3]=*/
            fuzztest::VectorOf(
                fuzztest::VectorOf(fuzztest::InRange<int64_t>(0, 399999999))
                    .WithMaxSize(1))
                .WithSize(128),
            /*samples[4]=*/
            fuzztest::VectorOf(
                fuzztest::VectorOf(fuzztest::InRange<int64_t>(0, 99999999))
                    .WithMaxSize(15))
                .WithSize(128)),
        /*num_sc_per_device=*/fuzztest::ElementOf<int>({1, 2, 4}),
        /*global_device_count=*/
        fuzztest::ElementOf<int>({1, 2, 4, 8, 16, 32, 64, 128}),
        /*max_ids_per_partition=*/fuzztest::InRange(1, 1024),
        /*max_unique_ids_per_partition=*/fuzztest::InRange(1, 1024),
        /*feature_stacking_strategy=*/
        fuzztest::ElementOf<FeatureStackingStrategy>(
            {FeatureStackingStrategy::kStackThenSplit,
             FeatureStackingStrategy::kSplitThenStack}),
        /*enable_minibatching=*/fuzztest::Arbitrary<bool>());

}  // namespace
}  // namespace jax_sc_embedding
