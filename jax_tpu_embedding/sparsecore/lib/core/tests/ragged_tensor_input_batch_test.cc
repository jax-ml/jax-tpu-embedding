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
#include <numeric>
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

  ExtractedCooTensorsPerSparseCore extracted_1(4, false, RowCombiner::kSum);
  ExtractedCooTensorsPerSparseCore extracted_2(2, false, RowCombiner::kSum);
  ExtractedCooTensorsPerSparseCore extracted_3(
      2, false,
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

  ExtractedCooTensorsPerSparseCore extracted(4, false, RowCombiner::kMean);
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
  ExtractedCooTensorsPerSparseCore extracted(4, false, RowCombiner::kSqrtn);
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

  ExtractedCooTensorsPerSparseCore extracted(4, false, RowCombiner::kSum);
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

TEST(RaggedTensorInputBatchTest, EmptyRowOffsetsTriggersCheckFailure) {
  std::vector<int64_t> embedding_ids = {};
  std::vector<int32_t> row_splits = {};
  auto construct_empty_batch = [&]() {
    RaggedTensorInputBatchWithOwnedData<int64_t, int32_t>
        ragged_tensor_feature_input(embedding_ids, row_splits);
  };
  EXPECT_DEATH(construct_empty_batch(), "row_offsets cannot be empty");
}

TEST(RaggedTensorInputBatchTest,
     ExtractCooTensors64SkipsEmptyRowsWithMeanCombiner) {
  using Coo64 = CooFormatT<int64_t>;
  constexpr int64_t kBaseId = int64_t{1} << 32;
  // Row 0 has 3 IDs, row 1 is empty, and row 2 has 1 ID.
  RaggedTensorInputBatchWithOwnedData<int64_t, int32_t> batch(
      {kBaseId, kBaseId + 1, kBaseId + 2, kBaseId + 5},
      /*row_splits=*/{0, 3, 3, 4});
  ExtractedCooTensorsPerSparseCoreT<int64_t> extracted(
      /*batch_size_per_sc=*/3, /*has_variable_weights=*/false,
      RowCombiner::kMean,
      KeyLayout<int64_t>::Make(/*num_scs_bit=*/2,
                               /*max_local_emb_id=*/int64_t{1} << 30));

  batch.ExtractCooTensors64(
      {
          .slice_start = 0,
          .slice_end = 3,
          .row_offset = 0,
          .col_offset = 0,
          .col_shift = 0,
          .num_sc_per_device = 4,
          .num_scs = 4,
          .combiner = RowCombiner::kMean,
      },
      extracted);

  EXPECT_THAT(extracted.ToCooVector(),
              ElementsAre(Coo64(0, kBaseId, 1.0f / 3.0f),
                          Coo64(0, kBaseId + 1, 1.0f / 3.0f),
                          Coo64(0, kBaseId + 2, 1.0f / 3.0f),
                          Coo64(2, kBaseId + 5, 1.0f)));
}

TEST(RaggedTensorInputBatchTest,
     ExtractCooTensors64AppliesShiftOffsetAndSqrtnCombiner) {
  using Coo64 = CooFormatT<int64_t>;
  constexpr int64_t kBaseId = int64_t{1} << 32;
  // With num_scs = 4, col_shift = 1, col_offset = 8:
  //   kBaseId + 0 (shard 0 -> shard 1) maps to kBaseId + 8 + 1 = kBaseId + 9.
  //   kBaseId + 3 (shard 3 -> shard 0) maps to kBaseId + 8 + 0 = kBaseId + 8.
  RaggedTensorInputBatchWithOwnedData<int64_t, int32_t> batch(
      {kBaseId, kBaseId + 3}, /*row_splits=*/{0, 2});
  ExtractedCooTensorsPerSparseCoreT<int64_t> extracted(
      /*batch_size_per_sc=*/1, /*has_variable_weights=*/false,
      RowCombiner::kSqrtn,
      KeyLayout<int64_t>::Make(/*num_scs_bit=*/2,
                               /*max_local_emb_id=*/int64_t{1} << 30));

  batch.ExtractCooTensors64(
      {
          .slice_start = 0,
          .slice_end = 1,
          .row_offset = 10,
          .col_offset = 8,
          .col_shift = 1,
          .num_sc_per_device = 4,
          .num_scs = 4,
          .combiner = RowCombiner::kSqrtn,
      },
      extracted);

  const float expected_gain = 1.0f / std::sqrt(2.0f);
  EXPECT_THAT(extracted.ToCooVector(),
              ElementsAre(Coo64(10, kBaseId + 9, expected_gain),
                          Coo64(10, kBaseId + 8, expected_gain)));
}

TEST(RaggedTensorInputBatchTest, ExtractCooTensors64SpansMultipleKeyBatches) {
  constexpr int64_t kBaseId = int64_t{1} << 32;
  // Row 0 (64 IDs) hits the exact 64-key batch boundary; row 1 (150 IDs) spans
  // two full 64-key batches plus a 22-key tail.
  std::vector<int64_t> embedding_ids(64 + 150);
  std::iota(embedding_ids.begin(), embedding_ids.end(), kBaseId);
  RaggedTensorInputBatchWithOwnedData<int64_t, int32_t> batch(
      embedding_ids, /*row_splits=*/{0, 64, 214});
  ExtractedCooTensorsPerSparseCoreT<int64_t> extracted(
      /*batch_size_per_sc=*/2, /*has_variable_weights=*/false,
      RowCombiner::kSum,
      KeyLayout<int64_t>::Make(/*num_scs_bit=*/2,
                               /*max_local_emb_id=*/int64_t{1} << 30));

  batch.ExtractCooTensors64(
      {
          .slice_start = 0,
          .slice_end = 2,
          .row_offset = 0,
          .col_offset = 0,
          .col_shift = 0,
          .num_sc_per_device = 4,
          .num_scs = 4,
          .combiner = RowCombiner::kSum,
      },
      extracted);

  std::vector<CooFormatT<int64_t>> expected;
  expected.reserve(64 + 150);
  for (int i = 0; i < 64; ++i) {
    expected.emplace_back(/*row_id=*/0, kBaseId + i, /*gain=*/1.0f);
  }
  for (int i = 0; i < 150; ++i) {
    expected.emplace_back(/*row_id=*/1, kBaseId + 64 + i, /*gain=*/1.0f);
  }
  EXPECT_EQ(extracted.ToCooVector(), expected);
}

}  // namespace
}  // namespace jax_sc_embedding
