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
#include "jax_tpu_embedding/sparsecore/lib/core/ragged_tensor_input_batch.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"

namespace jax_sc_embedding {
namespace {

using ::testing::ElementsAre;

class RaggedTensorInputBatchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // (0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (3, 2)
    embedding_ids_ = {0, 1, 2, 0, 0, 2};
    embedding_splits_ = {0, 3, 4, 5, 6};
  }

  std::vector<int64_t> embedding_ids_;
  std::vector<int32_t> embedding_splits_;
};

TEST_F(RaggedTensorInputBatchTest, SliceTest) {
  RaggedTensorInputBatch ragged_tensor_input_batch(embedding_ids_,
                                                   embedding_splits_);

  std::vector<CooFormat> coo_formats_1;
  std::vector<CooFormat> coo_formats_2;
  std::vector<CooFormat> coo_formats_3;

  ragged_tensor_input_batch.ExtractCooTensors(
      /*row_start=*/0, /*row_end=*/4, /*row_offset=*/0, /*col_offset=*/0,
      /*col_shift=*/0, /*num_scs=*/4, /*global_device_count=*/1,
      RowCombiner::kSum, coo_formats_1);
  ragged_tensor_input_batch.ExtractCooTensors(
      /*row_start=*/1, /*row_end=*/3, /*row_offset=*/0, /*col_offset=*/0,
      /*col_shift=*/0, /*num_scs=*/4, /*global_device_count=*/1,
      RowCombiner::kSum, coo_formats_2);
  ragged_tensor_input_batch.ExtractCooTensors(
      /*row_start=*/2, /*row_end=*/4, /*row_offset=*/16, /*col_offset=*/8,
      /*col_shift=*/0, /*num_scs=*/4, /*global_device_count=*/1,
      RowCombiner::kSum, coo_formats_3);

  EXPECT_THAT(coo_formats_1,
              ElementsAre(CooFormat(0, 0, 1.0), CooFormat(0, 1, 1.0),
                          CooFormat(0, 2, 1.0), CooFormat(1, 0, 1.0),
                          CooFormat(2, 0, 1.0), CooFormat(3, 2, 1.0)));
  EXPECT_THAT(coo_formats_2,
              ElementsAre(CooFormat(0, 0, 1.0), CooFormat(1, 0, 1.0)));

  EXPECT_THAT(coo_formats_3, ElementsAre(CooFormat(16, 0 + 8, 1.0),
                                         CooFormat(17, 2 + 8, 1.0)));
}

TEST_F(RaggedTensorInputBatchTest, TestWithMeanCombiner) {
  RaggedTensorInputBatch ragged_tensor_input_batch(embedding_ids_,
                                                   embedding_splits_);

  std::vector<CooFormat> coo_formats;
  ragged_tensor_input_batch.ExtractCooTensors(
      /*row_start=*/0, /*row_end=*/4, /*row_offset=*/0,
      /*col_offset=*/0,
      /*col_shift=*/0, /*num_scs=*/4, /*global_device_count=*/1,
      RowCombiner::kMean, coo_formats);
  EXPECT_THAT(coo_formats,
              ElementsAre(CooFormat(0, 0, 1.0 / 3), CooFormat(0, 1, 1.0 / 3),
                          CooFormat(0, 2, 1.0 / 3), CooFormat(1, 0, 1.0),
                          CooFormat(2, 0, 1.0), CooFormat(3, 2, 1.0)));
}

}  // namespace
}  // namespace jax_sc_embedding
