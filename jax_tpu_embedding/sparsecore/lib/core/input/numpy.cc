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

#include "jax_tpu_embedding/sparsecore/lib/core/input/numpy.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "pybind11/gil.h"  // from @pybind11
#include "pybind11/numpy.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "tsl/profiler/lib/traceme.h"

namespace jax_sc_embedding {

namespace py = ::pybind11;

int NumpyArrayInput::ExtractCooTensorsFrom2dArray(
    const py::array& features, const py::array& feature_weights,
    const int row_offset, const int col_offset, const int col_shift,
    const int num_scs_mod, const int num_scs_mod_inv,
    const int global_device_count, const RowCombiner combiner,
    std::vector<CooFormat>& coo_tensors) const {
  auto features_array = py::cast<py::array_t<int>>(features);
  auto features_weight_array = py::cast<py::array_t<float>>(feature_weights);
  auto features_array_t = features_array.unchecked<2>();
  auto features_weight_array_t = features_weight_array.unchecked<2>();

  // The remaining section doesn't require the GIL.
  py::gil_scoped_release release_gil;

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
  return nrows * ncols;
}

float NumpyArrayInput::ComputeWeightDivisor(RowCombiner combiner,
                                            const float* gains_buffer,
                                            py::ssize_t stride,
                                            py::ssize_t size) const {
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

int NumpyArrayInput::ExtractCooTensorsFrom1dArray(
    const py::array& features, const py::array& feature_weights,
    const int row_offset, const int col_offset, const int col_shift,
    const int num_scs_mod, const int num_scs_mod_inv,
    const int global_device_count, const RowCombiner combiner,
    std::vector<CooFormat>& coo_tensors) const {
  // We use proxy objects to the python array for the remainder of the
  // function and can hence release the GIL.
  py::gil_scoped_release release_gil;
  // The assumption here is that the gains are always represented as 32bit
  // float arrays (np array with dtype=np.float32) and the features are always
  // represented as 32bit int arrays (np array with dtype=np.int32).
  auto f = features.unchecked<py::array_t<int>, 1>();
  auto fw = feature_weights.unchecked<py::array_t<float>, 1>();

  coo_tensors.reserve(f.shape(0));
  int coo_tensors_extracted = 0;

  const int row_offset_per_device = row_offset / global_device_count;
  for (int i = 0; i < f.shape(0); ++i) {
    auto curr_features_t = f(i).unchecked<1>();
    auto curr_feature_weights_t = fw(i).unchecked<1>();
    const py::ssize_t stride = fw(i).strides(0) / sizeof(float);
    const py::ssize_t size = fw(i).shape(0);
    CHECK_EQ(curr_features_t.shape(0), size);
    coo_tensors_extracted += size;
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
  return coo_tensors_extracted;
}

int NumpyArrayInput::ExtractCooTensors(
    py::array features_split, py::array feature_weights_split,
    const int row_offset, const int col_offset, const int col_shift,
    const int num_scs, const int global_device_count,
    const RowCombiner combiner, std::vector<CooFormat>& coo_tensors) const {
  // We have to differentiate between 2D and 1D np.ndarray.
  // In the case of a 1D array of arrays, we have to iterate over the inner
  // arrays individually, collecting the COOFormat objects since the dtype of
  // the array is a py::object.
  tsl::profiler::TraceMe t([] { return "ExtractCooTensors"; });

  const py::buffer_info& features_buffer_info = features_split.request();
  CHECK(features_buffer_info.ndim == feature_weights_split.request().ndim &&
        (features_buffer_info.ndim == 1 || features_buffer_info.ndim == 2));
  CHECK(num_scs > 0 && (num_scs & (num_scs - 1)) == 0);
  const int num_scs_bit = std::log2(num_scs);
  const int num_scs_mod = (1 << num_scs_bit) - 1;
  const int num_scs_mod_inv = ~num_scs_mod;
  return features_buffer_info.ndim == 2
             ? ExtractCooTensorsFrom2dArray(
                   features_split, feature_weights_split, row_offset,
                   col_offset, col_shift, num_scs_mod, num_scs_mod_inv,
                   global_device_count, combiner, coo_tensors)
             : ExtractCooTensorsFrom1dArray(
                   features_split, feature_weights_split, row_offset,
                   col_offset, col_shift, num_scs_mod, num_scs_mod_inv,
                   global_device_count, combiner, coo_tensors);
}

absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>
NumpyArrayInput::GetStackedTableMetadata() const {
  tsl::profiler::TraceMe t([] { return "GetStackedTableMetadata"; });
  py::gil_scoped_acquire acq;
  absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>
      stacked_table_metadata;
  for (int i = 0; i < feature_specs.size(); ++i) {
    const py::object& feature_spec = feature_specs[i];
    const py::array& feature = this->features[i].cast<py::array>();
    const py::object& feature_transformation =
        feature_spec.attr("_id_transformation");
    const py::object& table_spec = feature_spec.attr("table_spec");
    const py::object& stacked_table_spec =
        table_spec.attr("stacked_table_spec");
    const std::string stacked_table_name = py::cast<std::string>(
        table_spec.attr("setting_in_stack").attr("stack_name"));
    int col_shift = 0;
    int col_offset = 0;
    int row_offset = 0;
    const int max_ids_per_partition =
        py::cast<int>(stacked_table_spec.attr("max_ids_per_partition"));
    const int max_unique_ids_per_partition =
        py::cast<int>(stacked_table_spec.attr("max_unique_ids_per_partition"));
    std::optional<int> suggested_coo_buffer_size;
    py::object suggested_coo_buffer_size_attr =
        stacked_table_spec.attr("suggested_coo_buffer_size");
    if (!suggested_coo_buffer_size_attr.is_none()) {
      suggested_coo_buffer_size = py::cast<int>(suggested_coo_buffer_size_attr);
    }
    const std::string row_combiner =
        py::cast<std::string>(stacked_table_spec.attr("combiner"));
    if (!feature_transformation.is_none()) {
      row_offset = py::cast<int>(feature_transformation.attr("row_offset"));
      col_shift = py::cast<int>(feature_transformation.attr("col_shift"));
      col_offset = py::cast<int>(feature_transformation.attr("col_offset"));
    }
    stacked_table_metadata[stacked_table_name].emplace_back(
        stacked_table_name, i, max_ids_per_partition,
        max_unique_ids_per_partition, row_offset, col_offset, col_shift,
        /*batch_size=*/feature.shape(0), suggested_coo_buffer_size,
        GetRowCombiner(row_combiner),
        /*max_col_id=*/std::numeric_limits<int>::max());
  }
  // Sort the stacked tables by row_offset.
  for (auto& [_, t] : stacked_table_metadata) {
    std::sort(t.begin(), t.end(),
              [](const StackedTableMetadata& a, const StackedTableMetadata& b) {
                return a.row_offset < b.row_offset;
              });
  }
  return stacked_table_metadata;
}

std::vector<CooFormat> NumpyArrayInput::ExtractCooTensorsForAllFeatures(
    const absl::Span<const StackedTableMetadata> stacked_table_metadata,
    const int local_device_id, const int local_device_count, const int num_scs,
    const int num_global_devices, int& batch_size_for_device,
    int& total_num_coo_tensors) const {
  py::gil_scoped_acquire acq;
  std::vector<CooFormat> coo_tensors;
  for (int i = 0; i < stacked_table_metadata.size(); ++i) {
    const StackedTableMetadata& metadata = stacked_table_metadata[i];
    const int feature_index = metadata.feature_index;
    const int row_offset = metadata.row_offset;
    const int col_offset = metadata.col_offset;
    const int col_shift = metadata.col_shift;
    const py::array& curr_feature =
        py::cast<py::array>(features[feature_index]);
    const py::array& curr_feature_weight =
        py::cast<py::array>(feature_weights[feature_index]);

    const int num_samples = curr_feature.shape(0);
    const int num_samples_per_split = num_samples / local_device_count;
    const int start_index = local_device_id * num_samples_per_split;
    int end_index = (local_device_id + 1) * num_samples_per_split;
    if (local_device_id == local_device_count - 1) {
      // Just in case the last split is not a full batch.
      end_index = num_samples;
    }
    py::slice feature_slice = py::slice(start_index, end_index, 1);
    const py::array feature_split = curr_feature[feature_slice];
    const py::array feature_weights_split = curr_feature_weight[feature_slice];
    batch_size_for_device += feature_split.shape(0);

    // In the case of feature stacking, we need to group all the COO tensors
    // at this stage (i.e., before the sorting later on).
    total_num_coo_tensors += ExtractCooTensors(
        feature_split, feature_weights_split, row_offset, col_offset, col_shift,
        num_scs, num_global_devices, metadata.row_combiner, coo_tensors);
  }
  return coo_tensors;
}

}  // namespace jax_sc_embedding
