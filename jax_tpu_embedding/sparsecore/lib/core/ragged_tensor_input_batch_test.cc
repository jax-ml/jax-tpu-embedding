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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/coo_format.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"

namespace jax_sc_embedding {
namespace {

using ::testing::ElementsAre;

TEST(RaggedTensorInputBatchTest, SliceTestWithSumCombiner) {
  // Input: (0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (3, 2)
  RaggedTensorInputBatchWithOwnedData<int64_t, int32_t>
      ragged_tensor_feature_input({0, 1, 2, 0, 0, 2}, {0, 3, 4, 5, 6});

  ExtractedCooTensors extracted_1(4, 4, false, RowCombiner::kSum);
  ExtractedCooTensors extracted_2(4, 4, false, RowCombiner::kSum);
  ExtractedCooTensors extracted_3(4, 20, false,
                                  RowCombiner::kSum);  // row_offset=16

  ragged_tensor_feature_input.ExtractCooTensors(
      {
          .slice_start = 0,
          .slice_end = 4,
          .row_offset = 0,
          .col_offset = 0,
          .col_shift = 0,
          .num_sc_per_device = 4,
          .num_scs = 4,
          .combiner = RowCombiner::kSum,
      },
      extracted_1);
  ragged_tensor_feature_input.ExtractCooTensors(
      {
          .slice_start = 1,
          .slice_end = 3,
          .row_offset = 0,
          .col_offset = 0,
          .col_shift = 0,
          .num_sc_per_device = 4,
          .num_scs = 4,
          .combiner = RowCombiner::kSum,
      },
      extracted_2);
  ragged_tensor_feature_input.ExtractCooTensors(
      {
          // row mapping after offset: (2..3) -> (17..18)
          .slice_start = 2,
          .slice_end = 4,
          .row_offset = 16,
          .col_offset = 8,
          .col_shift = 0,
          .num_sc_per_device = 4,
          .num_scs = 4,
          .combiner = RowCombiner::kSum,
      },
      extracted_3);

  EXPECT_THAT(extracted_1.ToCooVector(),
              ElementsAre(CooFormat(0, 0, 1.0), CooFormat(0, 1, 1.0),
                          CooFormat(0, 2, 1.0), CooFormat(1, 0, 1.0),
                          CooFormat(2, 0, 1.0), CooFormat(3, 2, 1.0)));
  EXPECT_THAT(extracted_2.ToCooVector(),
              ElementsAre(CooFormat(0, 0, 1.0), CooFormat(1, 0, 1.0)));

  EXPECT_THAT(
      extracted_3.ToCooVector(),
      ElementsAre(CooFormat(16, 0 + 8, 1.0), CooFormat(17, 2 + 8, 1.0)));
}

TEST(RaggedTensorInputBatchTest, SliceTestWithMeanCombiner) {
  // Input: (0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (3, 2)
  RaggedTensorInputBatchWithOwnedData<int64_t, int32_t>
      ragged_tensor_feature_input({0, 1, 2, 0, 0, 2}, {0, 3, 4, 5, 6});

  ExtractedCooTensors extracted(4, 4, false, RowCombiner::kMean);
  ragged_tensor_feature_input.ExtractCooTensors(
      {
          .slice_start = 0,
          .slice_end = 4,
          .row_offset = 0,
          .col_offset = 0,
          .col_shift = 0,
          .num_sc_per_device = 4,
          .num_scs = 4,
          .combiner = RowCombiner::kMean,
      },
      extracted);
  EXPECT_THAT(extracted.ToCooVector(),
              ElementsAre(CooFormat(0, 0, 1.0 / 3), CooFormat(0, 1, 1.0 / 3),
                          CooFormat(0, 2, 1.0 / 3), CooFormat(1, 0, 1.0),
                          CooFormat(2, 0, 1.0), CooFormat(3, 2, 1.0)));
}

TEST(RaggedTensorInputBatchTest, SliceTestWithSqrtnCombiner) {
  RaggedTensorInputBatchWithOwnedData<int64_t, int32_t>
      ragged_tensor_feature_input({0, 1, 2, 0, 0, 2}, {0, 3, 4, 5, 6});
  ExtractedCooTensors extracted(4, 4, false, RowCombiner::kSqrtn);
  ragged_tensor_feature_input.ExtractCooTensors(
      {
          .slice_start = 0,
          .slice_end = 4,
          .row_offset = 0,
          .col_offset = 0,
          .col_shift = 0,
          .num_sc_per_device = 4,
          .num_scs = 4,
          .combiner = RowCombiner::kSqrtn,
      },
      extracted);
  EXPECT_THAT(
      extracted.ToCooVector(),
      ElementsAre(CooFormat(0, 0, 1.0 / std::sqrt(3)),
                  CooFormat(0, 1, 1.0 / std::sqrt(3)),
                  CooFormat(0, 2, 1.0 / std::sqrt(3)), CooFormat(1, 0, 1.0),
                  CooFormat(2, 0, 1.0), CooFormat(3, 2, 1.0)));
}

TEST(RaggedTensorInputBatchTest,
       FixedValencyRowOffsetsCooExtractionIsCorrect) {
  std::vector<int64_t> embedding_ids = {0, 1, 0, 2, 0, 3, 0, 4};
  int batch_size = 4;
  int valency = 2;
  FixedValencyRowOffsets row_offsets(batch_size, valency);
  RaggedTensorInputBatch ragged_tensor_feature_input(
      absl::MakeConstSpan(embedding_ids), row_offsets);

  ExtractedCooTensors extracted(4, 4, false, RowCombiner::kSum);
  ragged_tensor_feature_input.ExtractCooTensors(
      {
          .slice_start = 0,
          .slice_end = 4,
          .row_offset = 0,
          .col_offset = 0,
          .col_shift = 0,
          .num_sc_per_device = 4,
          .num_scs = 4,
          .combiner = RowCombiner::kSum,
      },
      extracted);
  EXPECT_THAT(extracted.ToCooVector(),
              ElementsAre(CooFormat(0, 0, 1.0), CooFormat(0, 1, 1.0),  // Row 0
                          CooFormat(1, 0, 1.0), CooFormat(1, 2, 1.0),  // Row 1
                          CooFormat(2, 0, 1.0), CooFormat(2, 3, 1.0),  // Row 2
                          CooFormat(3, 0, 1.0), CooFormat(3, 4, 1.0))  // Row 3
  );
}

struct FixedValencyRowOffsetsTestCase {
  int batch_size;
  int valency;
  std::vector<int64_t> expected_offsets;
};

using FixedValencyRowOffsetsTest =
    testing::TestWithParam<FixedValencyRowOffsetsTestCase>;

TEST_P(FixedValencyRowOffsetsTest, FixedValencyRowOffsetsAreCorrect) {
  const FixedValencyRowOffsetsTestCase& test_case = GetParam();
  FixedValencyRowOffsets row_offsets(test_case.batch_size, test_case.valency);
  ASSERT_EQ(row_offsets.size(), test_case.expected_offsets.size());
  for (size_t i = 0; i < test_case.expected_offsets.size(); ++i) {
    EXPECT_EQ(row_offsets[i], test_case.expected_offsets[i]);
  }
}

INSTANTIATE_TEST_SUITE_P(FixedValencyRowOffsetsTests,
                         FixedValencyRowOffsetsTest,
                         testing::ValuesIn<FixedValencyRowOffsetsTestCase>(
                             {{
                                  .batch_size = 4,
                                  .valency = 2,
                                  .expected_offsets = {0, 2, 4, 6, 8},
                              },
                              {
                                  .batch_size = 2,
                                  .valency = 3,
                                  .expected_offsets = {0, 3, 6},
                              },
                              {
                                  .batch_size = 5,
                                  .valency = 1,
                                  .expected_offsets = {0, 1, 2, 3, 4, 5},
                              },
                              {
                                  .batch_size = 1,
                                  .valency = 5,
                                  .expected_offsets = {0, 5},
                              }}));

}  // namespace
}  // namespace jax_sc_embedding
