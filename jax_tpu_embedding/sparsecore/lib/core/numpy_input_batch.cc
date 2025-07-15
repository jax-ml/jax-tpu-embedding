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
#include "jax_tpu_embedding/sparsecore/lib/core/numpy_input_batch.h"

#include <memory>
#include <vector>

#include "absl/log/check.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "pybind11/gil.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "tsl/profiler/lib/traceme.h"

namespace jax_sc_embedding {

namespace py = ::pybind11;

namespace {

// Class to iterate over a dense 2D numpy array.
// Example:
//   arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
template <typename T>
class NumpyDenseInputBatchStream {
 public:
  explicit NumpyDenseInputBatchStream(const py::array_t<T>& matrix,
                                      int row_start, int row_end)
      : matrix_ref_(matrix.template unchecked<2>()),
        curr_row_(row_start),
        curr_col_(0),
        row_end_(row_end),
        rows_(row_end - row_start),
        cols_(matrix_ref_.shape(1)),
        size_(rows_ * cols_) {}

  // avoid implicit type conversions (leading to undefined behavior).
  template <typename U>
  NumpyDenseInputBatchStream(U matrix, int row_start, int row_end) = delete;

  int size() const { return size_; }

  int cols() const { return cols_; }

  void NextRow() {
    ++curr_row_;
    curr_col_ = 0;
  }

  void NextCol() { ++curr_col_; }

  void SeekCol(int col) { curr_col_ = col; }

  int row() const { return curr_row_; }

  int col() const { return curr_col_; }

  T get() const {
    DCHECK_LT(curr_row_, row_end_);
    DCHECK_LT(curr_col_, cols_);
    return matrix_ref_(curr_row_, curr_col_);
  }

 private:
  py::detail::unchecked_reference<T, 2> matrix_ref_;
  int curr_row_;
  int curr_col_;
  int row_end_;
  int rows_;
  int cols_;
  int size_;
};

// Class to iterate over a ragged 2D numpy array.
// Example:
//   arr = np.array([np.array([1, 2, 3]), np.array([4, 5])], dtype=object)
template <typename T>
class NumpyRaggedInputBatchStream {
 public:
  NumpyRaggedInputBatchStream(const py::array& rows, int row_start, int row_end)
      : rows_ref_(rows.template unchecked<py::array_t<T>, 1>()),
        row_ref_(std::make_unique<py::detail::unchecked_reference<T, 1>>(
            rows_ref_(row_start).template unchecked<1>())),
        curr_row_(row_start),
        curr_col_(0),
        size_(row_end - row_start),
        row_end_(row_end) {}

  // estimate of total embedding ids (currently a lower bound).
  int size() const { return size_; }

  int cols() const { return row_ref_->shape(0); }

  void NextRow() {
    ++curr_row_;
    curr_col_ = 0;
    if (curr_row_ < row_end_) {
      row_ref_ = std::make_unique<py::detail::unchecked_reference<T, 1>>(
          rows_ref_(curr_row_).template unchecked<1>());
    }
  }

  void NextCol() { ++curr_col_; }

  void SeekCol(int col) { curr_col_ = col; }

  int row() const { return curr_row_; }

  int col() const { return curr_col_; }

  T get() const {
    DCHECK_LT(curr_col_, row_ref_->shape(0));
    return (*row_ref_)(curr_col_);
  }

 private:
  py::detail::unchecked_reference<py::array_t<T>, 1> rows_ref_;
  // defined as a unique_ptr since the copy assignment operator is implicitly
  // deleted (due to const dims_ member).
  std::unique_ptr<py::detail::unchecked_reference<T, 1>> row_ref_;
  int curr_row_;
  int curr_col_;
  int size_;
  int row_end_;
};

}  // namespace

void NumpySparseInputBatch::ExtractCooTensors(
    const ExtractCooTensorsOptions& options,
    std::vector<CooFormat>& coo_tensors) {
  DCHECK(!PyGILState_Check());  // Does not require external GIL
  tsl::profiler::TraceMe t([] { return "ExtractCooTensors"; });

  if (feature_.ndim() == 2) {
    py::gil_scoped_acquire _;
    // I'm not sure but without casting, passing feature_ as `const py::array&`
    // and using feature_.unchecked_reference<T,2> seems to give garbage values.
    auto feature_array = feature_.cast<py::array_t<int>>();
    auto weights_array = weights_.cast<py::array_t<float>>();
    py::gil_scoped_release __;
    NumpyDenseInputBatchStream<int> values_stream(
        feature_array, options.slice_start, options.slice_end);
    NumpyDenseInputBatchStream<float> weights_stream(
        weights_array, options.slice_start, options.slice_end);
    ProcessCooTensors(options, values_stream, weights_stream, coo_tensors);
  } else {
    NumpyRaggedInputBatchStream<int> values_stream(
        feature_, options.slice_start, options.slice_end);
    NumpyRaggedInputBatchStream<float> weights_stream(
        weights_, options.slice_start, options.slice_end);
    ProcessCooTensors(options, values_stream, weights_stream, coo_tensors);
  }
}

}  // namespace jax_sc_embedding
