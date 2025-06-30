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
#include "jax_tpu_embedding/sparsecore/lib/core/sparse_coo_input_batch.h"

#include <Python.h>

#include <limits>
#include <vector>

#include "absl/base/call_once.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "pybind11/gil.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "tsl/profiler/lib/traceme.h"

namespace jax_sc_embedding {

namespace {

// Class to iterate over a sparse CSR array.
// Example:
//   values = [1, 2, 3, 4, 5, 6]
//   row_pointers = [0, 2, 5, 6]
//   This represents a sparse matrix with 3 rows:
//     Row 0: [1, 2]
//     Row 1: [3, 4, 5]
//     Row 2: [6]
template <typename T, typename ValuesView, typename RowPointersView>
class SparseCsrInputBatchStream {
 public:
  SparseCsrInputBatchStream(ValuesView values,
                            RowPointersView row_pointers, int row_start,
                            int row_end,
                            T max_vocab_id = std::numeric_limits<T>::max())
      : values_ref_(values),
        row_pointers_(row_pointers),
        curr_row_(row_start),
        row_end_(row_end),
        curr_idx_(row_pointers[row_start]),
        max_vocab_id_(max_vocab_id) {}

  int size() const { return row_pointers_[row_end_] - row_pointers_[0]; }

  int cols() const {
    return row_pointers_[curr_row_ + 1] - row_pointers_[curr_row_];
  }
  void next_row() {
    ++curr_row_;
    if (curr_row_ < row_end_) {
      curr_idx_ = row_pointers_[curr_row_];
    }
  }

  void next_col() { ++curr_idx_; }

  void seek_col(int col) { curr_idx_ = row_pointers_[curr_row_] + col; }

  int row() const { return curr_row_; }

  int col() const { return curr_idx_ - row_pointers_[curr_row_]; }

  T get() const {
    DCHECK_LT(curr_idx_, row_pointers_[curr_row_ + 1]);
    T embedding_id = values_ref_[curr_idx_];
    DCHECK(embedding_id >= 0 && embedding_id <= max_vocab_id_)
        << "Invalid vocabulary id: " << embedding_id
        << " for table vocabulary size: " << max_vocab_id_;
    return embedding_id;
  }

 private:
  ValuesView values_ref_;
  RowPointersView row_pointers_;
  int curr_row_;
  int row_end_;
  int curr_idx_;
  T max_vocab_id_;
};

template <typename T, typename U1, typename U2>
SparseCsrInputBatchStream(U1, U2, int, int, T)
    -> SparseCsrInputBatchStream<T, U1, U2>;

// Class to iterate over a sparse CSR array, providing unity weights for each
// value. This class takes an existing `ValuesStream` (e.g.,
// `SparseCsrInputBatchStream`) and provides an interface to iterate over the
// same structure, but returning a weight of 1.0 for each value instead of the
// actual value. This is useful when the input does not have associated weights
// but the processing logic expects a weight stream.
template <typename ValuesStream>
class UnityWeightsStream {
 public:
  UnityWeightsStream(const ValuesStream& value_stream)
      : value_stream_(value_stream), curr_col_(0) {}

  int size() const { return value_stream_.size(); }

  int cols() const { return value_stream_.cols(); }

  void next_row() { curr_col_ = 0; }

  void next_col() { ++curr_col_; }

  void seek_col(int col) { curr_col_ = 0; }

  int row() const { return value_stream_.row(); }

  int col() const { return curr_col_; }

  float get() const { return 1.0f; }

 private:
  const ValuesStream& value_stream_;
  int curr_col_;
};

template <typename T>
UnityWeightsStream(T) -> UnityWeightsStream<T>;

}  // namespace

void PySparseCooInputBatch::ConstructRowPointers() {
  if (!row_pointers_.empty()) {
    return;
  }
  auto indices_array = indices_.unchecked<2>();
  // Precompute indexes for row starts. Add a sentinel node for last row.
  row_pointers_.reserve(batch_size_ + 1);
  int row_pointers_index = 0;
  int last_row_id = -1;  // Only for DCHECK.
  int last_col_id = -1;  // Only for DCHECK.
  int last_val = -1;     // Only for DCHECK.
  for (int i = 0; i < indices_array.shape(0); ++i) {
    const int row_id = indices_array(i, 0), col_id = indices_array(i, 1),
              val = values_.at(i);
    DCHECK_GE(row_id, last_row_id) << "Decreasing row id values for row-major.";
    while (row_pointers_index <= row_id) {
      // Increment index until we reach the current row. Keep storing the row
      // pointers.
      row_pointers_.push_back(i);
      ++row_pointers_index;
    }

    // Loop Invariant: The index should point to one beyond the current row id.
    DCHECK_EQ(row_pointers_index, row_id + 1);

    if (row_id == last_row_id) {  // Same Row should have increasing col values.
      DCHECK_GT(col_id, last_col_id)
          << "Non-increasing col id values for row-major.";
    }

    last_row_id = row_id;  // NOMUTANTS - debugging.
    last_col_id = col_id;  // NOMUTANTS - debugging.
    last_val = val;        // NOMUTANTS - debugging.
  }
  while (row_pointers_index <= batch_size_) {
    row_pointers_.push_back(indices_array.shape(0));
    row_pointers_index++;
  }

  DCHECK_EQ(row_pointers_.size(), batch_size_ + 1);
}

void PySparseCooInputBatch::ConstructRowPointersIfRequired() {
  absl::call_once(row_pointer_construction_flag_,
                  &PySparseCooInputBatch::ConstructRowPointers, this);
}

void PySparseCooInputBatch::ExtractCooTensors(
    int row_start, int row_end, int row_offset, int col_offset, int col_shift,
    int num_scs, int global_device_count, RowCombiner combiner,
    std::vector<CooFormat>& coo_tensors) {
  DCHECK(!PyGILState_Check());  // Does not require external GIL.
  tsl::profiler::TraceMe t([] { return "ExtractCooTensors"; });

  ConstructRowPointersIfRequired();

  SparseCsrInputBatchStream values_stream(values_.unchecked<1>(),
                                          absl::MakeConstSpan(row_pointers_),
                                          row_start, row_end, max_vocab_id_);
  UnityWeightsStream weights_stream(values_stream);

  ProcessCooTensors(row_start, row_end, row_offset, col_offset, col_shift,
                    num_scs, global_device_count, combiner, values_stream,
                    weights_stream, coo_tensors);
}
}  // namespace jax_sc_embedding
