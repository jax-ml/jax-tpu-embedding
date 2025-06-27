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

#include <cmath>
#include <memory>
#include <vector>

#include "absl/log/check.h"  // from @com_google_absl
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
  void nextRow() {
    ++curr_row_;
    curr_col_ = 0;
  }

  void nextCol() { ++curr_col_; }

  void seekCol(int col) { curr_col_ = col; }

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

  void nextRow() {
    ++curr_row_;
    curr_col_ = 0;
    if (curr_row_ < row_end_) {
      row_ref_ = std::make_unique<py::detail::unchecked_reference<T, 1>>(
          rows_ref_(curr_row_).template unchecked<1>());
    }
  }

  void nextCol() { ++curr_col_; }

  void seekCol(int col) { curr_col_ = col; }

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

template <typename WeightsStreamT>
float ComputeWeightDivisor(RowCombiner combiner,
                           WeightsStreamT& weights_stream) {
  switch (combiner) {
    case RowCombiner::kSum:
      return 1.0f;
    case RowCombiner::kMean: {
      // Sum of elements.
      float sum = 0.0f;
      for (; weights_stream.col() < weights_stream.cols();
           weights_stream.nextCol()) {
        sum += weights_stream.get();
      }
      return sum;
    }
    case RowCombiner::kSqrtn: {
      // Sqrt of sum of squares.
      float sum = 0.0f;
      for (; weights_stream.col() < weights_stream.cols();
           weights_stream.nextCol()) {
        sum += std::pow(weights_stream.get(), 2);
      }
      return std::sqrt(sum);
    }
  }
}

// This might be moved to
// third_party/py/jax_tpu_embedding/sparsecore/lib/core/input_preprocessing.cc
template <typename ValuesStreamT, typename WeightsStreamT>
void ProcessCooTensors(int start_index, int end_index, int row_offset,
                       int col_offset, int col_shift, int num_scs,
                       int global_device_count, RowCombiner combiner,
                       ValuesStreamT& values_stream,
                       WeightsStreamT& weights_stream,
                       std::vector<CooFormat>& coo_tensors) {
  CHECK(num_scs > 0 && (num_scs & (num_scs - 1)) == 0);
  const int num_scs_bit = std::log2(num_scs);
  const int num_scs_mod = (1 << num_scs_bit) - 1;
  const int num_scs_mod_inv = ~num_scs_mod;

  coo_tensors.reserve(values_stream.size());

  const int row_offset_per_device = row_offset / global_device_count;

  DCHECK_EQ(values_stream.size(), weights_stream.size());

  for (; values_stream.row() < end_index && weights_stream.row() < end_index;
       values_stream.nextRow(), weights_stream.nextRow()) {
    DCHECK_EQ(values_stream.cols(), weights_stream.cols());
    DCHECK_EQ(values_stream.row(), weights_stream.row());
    DCHECK_EQ(values_stream.col(), weights_stream.col());
    DCHECK_EQ(values_stream.col(), 0);

    const int sample_id =
        values_stream.row() - start_index + row_offset_per_device;
    const float divisor = ComputeWeightDivisor(combiner, weights_stream);

    for (weights_stream.seekCol(0); values_stream.col() < values_stream.cols();
         values_stream.nextCol(), weights_stream.nextCol()) {
      const int embedding_id = values_stream.get();
      const float gain = weights_stream.get() / divisor;
      DCHECK_GE(embedding_id, 0);

      coo_tensors.emplace_back(sample_id,
                               GetColId(embedding_id, col_shift, col_offset,
                                        num_scs_mod, num_scs_mod_inv),
                               gain);
    }
  }
}

}  // namespace

void NumpySparseInputBatch::ExtractCooTensors(
    int start_index, int end_index, int row_offset, int col_offset,
    int col_shift, int num_scs, int global_device_count, RowCombiner combiner,
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
    NumpyDenseInputBatchStream<int> values_stream(feature_array, start_index,
                                                  end_index);
    NumpyDenseInputBatchStream<float> weights_stream(weights_array, start_index,
                                                     end_index);
    ProcessCooTensors(start_index, end_index, row_offset, col_offset, col_shift,
                      num_scs, global_device_count, combiner, values_stream,
                      weights_stream, coo_tensors);
  } else {
    NumpyRaggedInputBatchStream<int> values_stream(feature_, start_index,
                                                   end_index);
    NumpyRaggedInputBatchStream<float> weights_stream(weights_, start_index,
                                                      end_index);
    ProcessCooTensors(start_index, end_index, row_offset, col_offset, col_shift,
                      num_scs, global_device_count, combiner, values_stream,
                      weights_stream, coo_tensors);
  }
}

}  // namespace jax_sc_embedding
