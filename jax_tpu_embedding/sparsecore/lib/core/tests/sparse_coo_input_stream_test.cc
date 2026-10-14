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
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/abstract_input_batch.h"
#include "jax_tpu_embedding/sparsecore/lib/core/coo_format.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "jax_tpu_embedding/sparsecore/lib/core/process_coo_tensors_impl.h"
#include "jax_tpu_embedding/sparsecore/lib/core/sparse_coo_input_stream_impl.h"
#include "jax_tpu_embedding/sparsecore/lib/core/sparse_csr_input_stream_impl.h"
#include "jax_tpu_embedding/sparsecore/lib/core/unity_weights_stream_impl.h"

namespace jax_sc_embedding {
namespace {

using ::testing::Combine;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::Values;

// Lightweight trivially-copyable 2D view over `(row, col)` pairs for testing.
struct TestIndices2DView {
  const std::pair<int64_t, int64_t>* data;

  int64_t operator()(int64_t idx, int64_t col) const {
    return col == 0 ? data[idx].first : data[idx].second;
  }
};

TEST(SparseCooInputStreamTest, LowerBoundRowFindsCorrectBoundaries) {
  // Rows present: 1 (x2), 3 (x1), 6 (x3).
  const std::vector<std::pair<int64_t, int64_t>> coords = {
      {1, 0}, {1, 2}, {3, 0}, {6, 1}, {6, 4}, {6, 5}};
  const TestIndices2DView view{coords.data()};
  const int64_t n = coords.size();

  EXPECT_EQ(LowerBoundRow(view, /*start_idx=*/0, n, /*target_row=*/0), 0);
  EXPECT_EQ(LowerBoundRow(view, /*start_idx=*/0, n, /*target_row=*/1), 0);
  EXPECT_EQ(LowerBoundRow(view, /*start_idx=*/0, n, /*target_row=*/2), 2);
  EXPECT_EQ(LowerBoundRow(view, /*start_idx=*/0, n, /*target_row=*/3), 2);
  EXPECT_EQ(LowerBoundRow(view, /*start_idx=*/0, n, /*target_row=*/4), 3);
  EXPECT_EQ(LowerBoundRow(view, /*start_idx=*/0, n, /*target_row=*/6), 3);
  EXPECT_EQ(LowerBoundRow(view, /*start_idx=*/0, n, /*target_row=*/7), 6);
  // Empty subrange returns start_idx == end_idx.
  EXPECT_EQ(LowerBoundRow(view, /*start_idx=*/3, /*end_idx=*/3,
                          /*target_row=*/1),
            3);
}

TEST(SparseCooInputStreamTest, IteratesNonEmptyRowsAndSkipsEmptyRows) {
  // Batch of 6 rows (0..5), where rows 1, 3, and 5 are empty:
  //   Row 0: [10, 20]
  //   Row 2: [30]
  //   Row 4: [40, 50, 60]
  const std::vector<std::pair<int64_t, int64_t>> coords = {
      {0, 0}, {0, 1}, {2, 0}, {4, 0}, {4, 2}, {4, 5}};
  const std::vector<int32_t> values = {10, 20, 30, 40, 50, 60};
  const TestIndices2DView indices_view{coords.data()};
  const absl::Span<const int32_t> values_view(values);

  SparseCooInputBatchStream<int32_t, TestIndices2DView,
                            absl::Span<const int32_t>>
      stream(indices_view, values_view, /*row_start=*/0, /*row_end=*/6);

  std::vector<std::pair<int, std::vector<int32_t>>> visited_rows;
  for (; stream.row() < 6; stream.NextRow()) {
    ASSERT_EQ(stream.cols(), stream.getRowSpan().size());
    visited_rows.emplace_back(stream.row(),
                              std::vector<int32_t>(stream.getRowSpan().begin(),
                                                   stream.getRowSpan().end()));
  }

  EXPECT_THAT(visited_rows, ElementsAre(Pair(0, ElementsAre(10, 20)),
                                        Pair(2, ElementsAre(30)),
                                        Pair(4, ElementsAre(40, 50, 60))));
}

TEST(SparseCooInputStreamTest, IteratesOnlyRowsWithinSliceSubrange) {
  const std::vector<std::pair<int64_t, int64_t>> coords = {
      {1, 0}, {2, 0}, {2, 1}, {5, 0}};
  const std::vector<int32_t> values = {100, 200, 201, 500};
  const TestIndices2DView indices_view{coords.data()};
  const absl::Span<const int32_t> values_view(values);

  // Slice [2, 5) should only visit row 2 (rows 3 and 4 are empty).
  SparseCooInputBatchStream<int32_t, TestIndices2DView,
                            absl::Span<const int32_t>>
      stream(indices_view, values_view, /*row_start=*/2, /*row_end=*/5);

  ASSERT_EQ(stream.row(), 2);
  EXPECT_EQ(stream.cols(), 2);
  EXPECT_THAT(stream.getRowSpan(), ElementsAre(200, 201));

  stream.NextRow();
  EXPECT_EQ(stream.row(), 5);
  EXPECT_EQ(stream.cols(), 0);
  EXPECT_THAT(stream.getRowSpan(), IsEmpty());
}

TEST(SparseCooInputStreamTest, EmptySliceSubrangeIsImmediatelyAtRowEnd) {
  const std::vector<std::pair<int64_t, int64_t>> coords = {
      {1, 0}, {2, 0}, {2, 1}, {5, 0}};
  const std::vector<int32_t> values = {100, 200, 201, 500};
  const TestIndices2DView indices_view{coords.data()};
  const absl::Span<const int32_t> values_view(values);

  // Slice [3, 5) contains only empty rows -> immediately at row_end (5).
  SparseCooInputBatchStream<int32_t, TestIndices2DView,
                            absl::Span<const int32_t>>
      stream(indices_view, values_view, /*row_start=*/3, /*row_end=*/5);

  EXPECT_EQ(stream.row(), 5);
  EXPECT_EQ(stream.cols(), 0);
  EXPECT_THAT(stream.getRowSpan(), IsEmpty());
}

TEST(SparseCooInputStreamTest, EmptyTensorIsImmediatelyAtRowEnd) {
  const std::vector<std::pair<int64_t, int64_t>> empty_coords;
  const std::vector<int32_t> empty_values;

  SparseCooInputBatchStream<int32_t, TestIndices2DView,
                            absl::Span<const int32_t>>
      stream(TestIndices2DView{empty_coords.data()},
             absl::MakeConstSpan(empty_values), /*row_start=*/0, /*row_end=*/4);

  EXPECT_EQ(stream.row(), 4);
  EXPECT_EQ(stream.cols(), 0);
  EXPECT_THAT(stream.getRowSpan(), IsEmpty());
}

class SparseCooVsCsrStreamEquivalenceTest
    : public ::testing::TestWithParam<
          std::tuple<RowCombiner, std::pair<int, int>>> {};

TEST_P(SparseCooVsCsrStreamEquivalenceTest,
       ProducesIdenticalExtractedCooTensors) {
  // Arrange: 8 rows (0..7) with empty rows at 0, 3, 5, 7:
  //   Row 0: []
  //   Row 1: [2, 5]
  //   Row 2: [1, 3, 7]
  //   Row 3: []
  //   Row 4: [4]
  //   Row 5: []
  //   Row 6: [0, 6]
  //   Row 7: []
  const auto [combiner, slice_range] = GetParam();
  const auto [slice_start, slice_end] = slice_range;
  const std::vector<std::pair<int64_t, int64_t>> coords = {
      {1, 0}, {1, 3}, {2, 0}, {2, 1}, {2, 2}, {4, 5}, {6, 1}, {6, 2}};
  const std::vector<int32_t> values = {2, 5, 1, 3, 7, 4, 0, 6};
  const std::vector<int64_t> csr_row_pointers = {0, 0, 2, 5, 5, 6, 6, 8, 8};

  const TestIndices2DView indices_view{coords.data()};
  const absl::Span<const int32_t> values_view(values);
  const absl::Span<const int64_t> row_ptrs_view(csr_row_pointers);
  const AbstractInputBatch::ExtractCooTensorsOptions options{
      .slice_start = slice_start,
      .slice_end = slice_end,
      .row_offset = 4,
      .col_offset = 8,
      .col_shift = 0,
      .num_sc_per_device = 4,
      .num_scs = 4,
      .combiner = combiner,
  };

  ExtractedCooTensorsPerSparseCore coo_extracted(
      /*batch_size_per_sc=*/8, /*has_variable_weights=*/false, combiner);
  SparseCooInputBatchStream<int32_t, TestIndices2DView,
                            absl::Span<const int32_t>>
      coo_stream(indices_view, values_view, slice_start, slice_end);
  UnityWeightsStream coo_weights(coo_stream);

  ExtractedCooTensorsPerSparseCore csr_extracted(
      /*batch_size_per_sc=*/8, /*has_variable_weights=*/false, combiner);
  SparseCsrInputBatchStream<int32_t, absl::Span<const int32_t>,
                            absl::Span<const int64_t>>
      csr_stream(values_view, row_ptrs_view, slice_start, slice_end);
  UnityWeightsStream csr_weights(csr_stream);

  // Act.
  ProcessCooTensors(options, coo_stream, coo_weights, coo_extracted);
  ProcessCooTensors(options, csr_stream, csr_weights, csr_extracted);

  // Assert.
  EXPECT_EQ(coo_extracted.ToCooVector(), csr_extracted.ToCooVector());
}

INSTANTIATE_TEST_SUITE_P(
    AllCombinersAndSlices, SparseCooVsCsrStreamEquivalenceTest,
    Combine(Values(RowCombiner::kSum, RowCombiner::kMean, RowCombiner::kSqrtn),
            Values(std::pair<int, int>{0, 8}, std::pair<int, int>{0, 4},
                   std::pair<int, int>{4, 8}, std::pair<int, int>{2, 6})));

}  // namespace
}  // namespace jax_sc_embedding
