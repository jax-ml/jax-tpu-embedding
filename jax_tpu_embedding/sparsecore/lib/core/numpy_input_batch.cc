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
#include "jax_tpu_embedding/sparsecore/lib/core/abstract_input_batch.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "tsl/profiler/lib/traceme.h"

namespace jax_sc_embedding {

namespace py = ::pybind11;

namespace {
float ComputeWeightDivisor(RowCombiner combiner, const float* gains_buffer,
                           py::ssize_t stride, py::ssize_t size) {
  DCHECK(!PyGILState_Check());  // Does not require GIL
  switch (combiner) {
    case RowCombiner::kSum:
      return 1.0f;
    case RowCombiner::kMean: {
      // Sum of elements.
      float sum = 0.0f;
      for (py::ssize_t i = 0; i < size; ++i) {
        sum += gains_buffer[i * stride];
      }
      return sum;
    }
    case RowCombiner::kSqrtn: {
      // Sqrt of sum of squares.
      float sum = 0.0f;
      for (py::ssize_t i = 0; i < size; ++i) {
        float gain = gains_buffer[i * stride];
        sum += gain * gain;
      }
      return std::sqrt(sum);
    }
  }
}
// `features` and `feature_weights` are 2D arrays, which means they are
// rectangular shaped arrays of dtype int (features) and float
// (feature_weights).
void ExtractCooTensorsFrom2dArray(const py::array& features,
                                  const py::array& feature_weights,
                                  const int row_offset, const int col_offset,
                                  const int col_shift, const int num_scs_mod,
                                  const int num_scs_mod_inv,
                                  const int global_device_count,
                                  const RowCombiner combiner,
                                  std::vector<CooFormat>& coo_tensors) {
  DCHECK(PyGILState_Check());  // Requires GIL.

  auto features_array = py::cast<py::array_t<int>>(features);
  auto features_weight_array = py::cast<py::array_t<float>>(feature_weights);

  py::gil_scoped_release _;

  auto features_array_t = features_array.unchecked<2>();
  auto features_weight_array_t = features_weight_array.unchecked<2>();

  const py::ssize_t nrows = features_array_t.shape(0);
  const py::ssize_t ncols = features_array_t.shape(1);
  const py::ssize_t cstride = features_weight_array.strides(1) / sizeof(float);

  coo_tensors.reserve(nrows * ncols);
  CHECK_EQ(nrows, features_weight_array_t.shape(0));
  CHECK_EQ(ncols, features_weight_array_t.shape(1));
  const int row_offset_per_device = row_offset / global_device_count;
  for (py::ssize_t i = 0; i < nrows; ++i) {
    const int row_id = i + row_offset_per_device;
    const float divisor = ComputeWeightDivisor(
        combiner, features_weight_array_t.data(i, 0), cstride, ncols);
    for (py::ssize_t j = 0; j < ncols; ++j) {
      const int col = features_array_t(i, j);
      const float gain = features_weight_array_t(i, j) / divisor;
      coo_tensors.emplace_back(
          row_id,
          GetColId(col, col_shift, col_offset, num_scs_mod, num_scs_mod_inv),
          gain);
    }
  }
}

// `features` and `feature_weights` are 1D arrays of arrays. That is, they
// are numpy arrays with dtype=object where the object is a 1D array of ints
// (features) and floats (feature_weights). When looping over the inner arrays,
// we have to cast the object to a py::array_t<T> and then access the inner
// arrays.
void ExtractCooTensorsFrom1dArray(const py::array& features,
                                  const py::array& feature_weights,
                                  const int row_offset, const int col_offset,
                                  const int col_shift, const int num_scs_mod,
                                  const int num_scs_mod_inv,
                                  const int global_device_count,
                                  const RowCombiner combiner,
                                  std::vector<CooFormat>& coo_tensors) {
  DCHECK(PyGILState_Check());  // Requires GIL
  py::gil_scoped_release _;
  // The assumption here is that the gains are always represented as 32bit
  // float arrays (np array with dtype=np.float32) and the features are always
  // represented as 32bit int arrays (np array with dtype=np.int32).
  auto f = features.unchecked<py::array_t<int>, 1>();
  auto fw = feature_weights.unchecked<py::array_t<float>, 1>();

  coo_tensors.reserve(f.shape(0));

  const int row_offset_per_device = row_offset / global_device_count;
  for (int i = 0; i < f.shape(0); ++i) {
    auto curr_features_t = f(i).unchecked<1>();
    auto curr_feature_weights_t = fw(i).unchecked<1>();
    const py::ssize_t stride = fw(i).strides(0) / sizeof(float);
    const py::ssize_t size = fw(i).shape(0);
    CHECK_EQ(curr_features_t.shape(0), size);
    const int row_id = i + row_offset_per_device;
    const float divisor = ComputeWeightDivisor(
        combiner, curr_feature_weights_t.data(0), stride, size);
    for (int j = 0; j < size; ++j) {
      const float gain = curr_feature_weights_t(j) / divisor;
      coo_tensors.emplace_back(
          row_id,
          GetColId(curr_features_t(j), col_shift, col_offset, num_scs_mod,
                   num_scs_mod_inv),
          gain);
    }
  }
}
}  // namespace

void NumpySparseInputBatch::ExtractCooTensors(
    int start_index, int end_index, int row_offset, int col_offset,
    int col_shift, int num_scs, int global_device_count, RowCombiner combiner,
    std::vector<CooFormat>& coo_tensors) const {
  DCHECK(!PyGILState_Check());  // Does not require external GIL
  tsl::profiler::TraceMe t([] { return "ExtractCooTensors"; });
  CHECK(num_scs > 0 && (num_scs & (num_scs - 1)) == 0);
  const int num_scs_bit = std::log2(num_scs);
  const int num_scs_mod = (1 << num_scs_bit) - 1;
  const int num_scs_mod_inv = ~num_scs_mod;

  py::gil_scoped_acquire _;
  py::slice slice = py::slice(start_index, end_index, 1);
  auto feature_slice = feature_[slice];
  auto weights_slice = weights_[slice];

  // We have to differentiate between 2D and 1D np.ndarray.
  // In the case of a 1D array of arrays, we have to iterate over the inner
  // arrays individually, collecting the COOFormat objects since the dtype of
  // the array is a py::object.
  if (feature_.ndim() == 2) {
    ExtractCooTensorsFrom2dArray(feature_slice, weights_slice, row_offset,
                                 col_offset, col_shift, num_scs_mod,
                                 num_scs_mod_inv, global_device_count, combiner,
                                 coo_tensors);
  } else {
    ExtractCooTensorsFrom1dArray(feature_slice, weights_slice, row_offset,
                                 col_offset, col_shift, num_scs_mod,
                                 num_scs_mod_inv, global_device_count, combiner,
                                 coo_tensors);
  }
}

}  // namespace jax_sc_embedding
