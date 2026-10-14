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
#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_SPARSE_COO_INPUT_STREAM_IMPL_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_SPARSE_COO_INPUT_STREAM_IMPL_H_

#include <cstdint>
#include <type_traits>

#include "absl/base/attributes.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace jax_sc_embedding {

// Returns the first index `i` in `[start_idx, end_idx)` where
// `indices(i, 0) >= target_row`.
template <typename IndicesView>
int64_t LowerBoundRow(const IndicesView& indices, int64_t start_idx,
                      int64_t end_idx, int64_t target_row) {
  int64_t left = start_idx;
  int64_t right = end_idx;
  while (left < right) {
    const int64_t mid = left + (right - left) / 2;
    if (indices(mid, 0) < target_row) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left;
}

// Class to iterate over non-empty rows of a row-major sorted 2D COO tensor
// within `[row_start, row_end)`.
//
// Example:
//   indices = [[0, 0], [0, 1], [2, 0], [2, 3], [2, 4]]
//   values  = [10, 20, 30, 40, 50]
//   This represents a sparse matrix with non-empty rows 0 and 2 (row 1 is
//   empty):
//     Row 0: [10, 20]
//     Row 2: [30, 40, 50]
//
// Relationship to `SparseCsrInputBatchStream`:
//   Both classes implement the row-stream interface consumed by
//   `ProcessCooTensors` (`row()`, `cols()`, `getRowSpan()`, `NextRow()`).
//   Whereas `SparseCsrInputBatchStream` requires a dense `row_pointers` array
//   of size `batch_size + 1` and steps through every row in `[row_start,
//   row_end)` (including empty rows with `cols() == 0`),
//   `SparseCooInputBatchStream` operates directly on the sorted COO `(indices,
//   values)` views. It locates the starting offset via `LowerBoundRow` in
//   O(log(nnz)) time and advances directly from one non-empty row to the next
//   in O(nnz_slice) total time, skipping empty rows in O(1) time without
//   allocating a dense `row_pointers` vector.
//
// Template parameters:
//   - `T`: Element type of `values` (e.g., `int32_t`).
//   - `IndicesView`: View into an `(N, 2)` row-major array supporting
//     `indices(i, col)`.
//   - `ValuesView`: View into a 1D array of length `N` supporting
//     `values[i]` and `values.size()`.
template <typename T, typename IndicesView, typename ValuesView>
class ABSL_ATTRIBUTE_VIEW SparseCooInputBatchStream {
 public:
  static_assert(std::is_trivially_copyable_v<IndicesView>,
                "IndicesView must be trivially copyable.");
  static_assert(std::is_trivially_copyable_v<ValuesView>,
                "ValuesView must be trivially copyable.");

  SparseCooInputBatchStream(IndicesView indices ABSL_ATTRIBUTE_LIFETIME_BOUND,
                            ValuesView values ABSL_ATTRIBUTE_LIFETIME_BOUND,
                            int row_start, int row_end)
      : indices_ref_(indices),
        values_ref_(values),
        row_end_(row_end),
        num_elements_(values.size()),
        curr_row_start_idx_(row_start < row_end
                                ? LowerBoundRow(indices, /*start_idx=*/0,
                                                num_elements_, row_start)
                                : num_elements_),
        curr_row_cols_(0),
        curr_row_(row_end) {
    if (curr_row_start_idx_ > 0 && curr_row_start_idx_ < num_elements_) {
      DCHECK_GE(indices_ref_(curr_row_start_idx_, 0),
                indices_ref_(curr_row_start_idx_ - 1, 0))
          << "Decreasing row id values for row-major.";
    }
    AdvanceToCurrentRow();
  }

  // Returns number of values in current row.
  int cols() const { return curr_row_cols_; }

  void NextRow() {
    const int prev_row = curr_row_;
    curr_row_start_idx_ += curr_row_cols_;
    AdvanceToCurrentRow();
    if (curr_row_ < row_end_) {
      DCHECK_GT(curr_row_, prev_row)
          << "Decreasing row id values for row-major.";
    }
  }

  absl::Span<const T> getRowSpan() const {
    return curr_row_cols_ == 0
               ? absl::Span<const T>()
               : absl::MakeSpan(&values_ref_[curr_row_start_idx_],
                                curr_row_cols_);
  }

  int row() const { return curr_row_; }

 private:
  void AdvanceToCurrentRow() {
    if (curr_row_start_idx_ >= num_elements_) {
      curr_row_ = row_end_;
      curr_row_cols_ = 0;
      return;
    }
    const int64_t next_row = indices_ref_(curr_row_start_idx_, 0);
    if (next_row >= row_end_) {
      curr_row_ = row_end_;
      curr_row_cols_ = 0;
      return;
    }
    curr_row_ = static_cast<int>(next_row);
    int64_t idx = curr_row_start_idx_ + 1;
    int64_t last_col_id = indices_ref_(curr_row_start_idx_, 1);
    while (idx < num_elements_ && indices_ref_(idx, 0) == next_row) {
      const int64_t col_id = indices_ref_(idx, 1);
      DCHECK_GT(col_id, last_col_id)
          << "Non-increasing col id values for row-major.";
      last_col_id = col_id;
      ++idx;
    }
    curr_row_cols_ = static_cast<int>(idx - curr_row_start_idx_);
  }

  IndicesView indices_ref_;
  ValuesView values_ref_;
  int row_end_;
  int64_t num_elements_;
  int64_t curr_row_start_idx_;
  int curr_row_cols_;
  int curr_row_;
};

}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_SPARSE_COO_INPUT_STREAM_IMPL_H_
