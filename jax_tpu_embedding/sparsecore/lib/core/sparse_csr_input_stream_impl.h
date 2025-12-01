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
#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_SPARSE_CSR_INPUT_STREAM_IMPL_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_SPARSE_CSR_INPUT_STREAM_IMPL_H_

#include <limits>
#include <type_traits>

#include "absl/base/attributes.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace jax_sc_embedding {

// Class to iterate over a sparse CSR array.
// Example:
//   values = [1, 2, 3, 4, 5, 6]
//   row_pointers = [0, 2, 5, 6]
//   This represents a sparse matrix with 3 rows:
//     Row 0: [1, 2]
//     Row 1: [3, 4, 5]
//     Row 2: [6]
// ValuesView and RowPointersView are template parameters that represent a view
// into the underlying data (or even the actual data itself):
//   - ValuesView is required to support `operator[]`.
//   - RowPointersView is required to support `operator[]`.
// This allows the class to be used with different types of data sources, such
// as vectors, arrays, or other data structures.
template <typename T, typename ValuesView, typename RowPointersView>
class ABSL_ATTRIBUTE_VIEW SparseCsrInputBatchStream {
 public:
  // Ensures that ValuesView and RowPointersView are view-like types that are
  // cheap to copy. This prevents expensive copies of underlying data
  // containers (e.g. std::vector) and encourages passing views (e.g.
  // absl::Span) instead.
  static_assert(std::is_trivially_copyable_v<ValuesView>,
                "ValuesView must be trivially copyable.");
  static_assert(std::is_trivially_copyable_v<RowPointersView>,
                "RowPointersView must be trivially copyable.");

  SparseCsrInputBatchStream(
      ValuesView values ABSL_ATTRIBUTE_LIFETIME_BOUND,
      RowPointersView row_pointers ABSL_ATTRIBUTE_LIFETIME_BOUND, int row_start,
      int row_end, absl::string_view table_name = "unknown_table_name",
      T max_vocab_id = std::numeric_limits<T>::max())
      : values_ref_(values),
        row_pointers_(row_pointers),
        row_start_(row_start),
        curr_row_(row_start),
        row_end_(row_end),
        curr_idx_(row_pointers[row_start]),
        max_vocab_id_(max_vocab_id),
        table_name_(table_name) {
    curr_row_cols_ =
        curr_row_ == row_end_
            ? 0
            : row_pointers_[curr_row_ + 1] - row_pointers_[curr_row_];
  }

  // Returns number of values in current row.
  int cols() const { return curr_row_cols_; }

  void NextRow() {
    ++curr_row_;
    if (curr_row_ < row_end_) {
      curr_idx_ = row_pointers_[curr_row_];
      curr_row_cols_ = row_pointers_[curr_row_ + 1] - row_pointers_[curr_row_];
    }
  }

  void NextCol() { ++curr_idx_; }

  void SeekCol(int col) { curr_idx_ = row_pointers_[curr_row_] + col; }

  int row() const { return curr_row_; }

  int col() const { return curr_idx_ - row_pointers_[curr_row_]; }

  T get() const {
    DCHECK_LT(curr_idx_, row_pointers_[curr_row_ + 1]);
    T embedding_id = values_ref_[curr_idx_];
    CHECK(embedding_id >= 0 && embedding_id <= max_vocab_id_)
        << "Invalid vocabulary id: " << embedding_id << " for table "
        << table_name_ << " with vocabulary size: " << max_vocab_id_;
    return embedding_id;
  }

 private:
  ValuesView values_ref_;
  RowPointersView row_pointers_;
  int row_start_;
  int curr_row_;
  int row_end_;
  int curr_idx_;
  int curr_row_cols_;
  T max_vocab_id_;
  absl::string_view table_name_;
};

}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_SPARSE_CSR_INPUT_STREAM_IMPL_H_
