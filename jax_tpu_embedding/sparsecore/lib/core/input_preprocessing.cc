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
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/blocking_counter.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_threads.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/gil.h"  // from @pybind11
#include "pybind11/numpy.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/traceme.h"

namespace jax_sc_embedding {

namespace {

namespace py = ::pybind11;

namespace {

float ComputeWeightDivisor(RowCombiner combiner, const float* gains_buffer,
                         py::ssize_t stride, py::ssize_t size) {
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

}  // namespace

// `features` and `feature_weights` are 2D arrays, which means they are
// rectangular shaped arrays of dtype int (features) and float
// (feature_weights).
int ExtractCooTensorsFrom2dArray(const py::array& features,
                                 const py::array& feature_weights,
                                 const int row_offset, const int col_offset,
                                 const int col_shift, const int num_scs_mod,
                                 const int num_scs_mod_inv,
                                 const int global_device_count,
                                 const RowCombiner combiner,
                                 std::vector<CooFormat>& coo_tensors) {
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

// `features` and `feature_weights` are 1D arrays of arrays. That is, they
// are numpy arrays with dtype=object where the object is a 1D array of ints
// (features) and floats (feature_weights). When looping over the inner arrays,
// we have to cast the object to a py::array_t<T> and then access the inner
// arrays.
int ExtractCooTensorsFrom1dArray(const py::array& features,
                                 const py::array& feature_weights,
                                 const int row_offset, const int col_offset,
                                 const int col_shift, const int num_scs_mod,
                                 const int num_scs_mod_inv,
                                 const int global_device_count,
                                 const RowCombiner combiner,
                                 std::vector<CooFormat>& coo_tensors) {
  // The assumption here is that the gains are always represented as 32bit
  // float arrays (np array with dtype=np.float32) and the features are always
  // represented as 32bit int arrays (np array with dtype=np.int32).
  auto f = features.unchecked<py::array_t<int>, 1>();
  auto fw = feature_weights.unchecked<py::array_t<float>, 1>();

  // We use proxy objects to the python array for the remainder of the function
  // and can hence release the GIL.
  py::gil_scoped_release release_gil;

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

int ExtractCooTensors(const py::array& features,
                      const py::array& feature_weights, const int row_offset,
                      const int col_offset, const int col_shift,
                      const int num_scs, const int global_device_count,
                      const RowCombiner combiner,
                      std::vector<CooFormat>& coo_tensors) {
  // We have to differentiate between 2D and 1D np.ndarray.
  // In the case of a 1D array of arrays, we have to iterate over the inner
  // arrays individually, collecting the COOFormat objects since the dtype of
  // the array is a py::object.
  tsl::profiler::TraceMe t([] { return "ExtractCooTensors"; });

  const py::buffer_info& features_buffer_info = features.request();
  CHECK(features_buffer_info.ndim == feature_weights.request().ndim &&
        (features_buffer_info.ndim == 1 || features_buffer_info.ndim == 2));
  CHECK(num_scs > 0 && (num_scs & (num_scs - 1)) == 0);
  const int num_scs_bit = std::log2(num_scs);
  const int num_scs_mod = (1 << num_scs_bit) - 1;
  const int num_scs_mod_inv = ~num_scs_mod;
  return features_buffer_info.ndim == 2
             ? ExtractCooTensorsFrom2dArray(
                   features, feature_weights, row_offset, col_offset, col_shift,
                   num_scs_mod, num_scs_mod_inv, global_device_count, combiner,
                   coo_tensors)
             : ExtractCooTensorsFrom1dArray(
                   features, feature_weights, row_offset, col_offset, col_shift,
                   num_scs_mod, num_scs_mod_inv, global_device_count, combiner,
                   coo_tensors);
}

absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>
GetStackedTableMetadata(py::list feature_specs, py::list features) {
  tsl::profiler::TraceMe t([] { return "GetStackedTableMetadata"; });
  absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>
      stacked_table_metadata;
  for (int i = 0; i < feature_specs.size(); ++i) {
    const py::object& feature_spec = feature_specs[i];
    const py::array& feature = features[i].cast<py::array>();
    const py::object& feature_transformation =
        feature_spec.attr("_id_transformation");
    const py::object& table_spec = feature_spec.attr("table_spec");
    const py::object& stacked_table_spec =
        table_spec.attr("stacked_table_spec");
    const std::string stacked_table_name = py::cast<std::string>(
        table_spec.attr("_setting_in_stack").attr("stack_name"));
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
    if (!suggested_coo_buffer_size_attr.is_none()){
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

// Preprocess inputs for a single table. Stacked table here refers to a
// a table that has no parent in the table stacking hierarchy. So in the case
// of table stacking, the stacked table is the top level table and in the case
// where we don't have any table stacking, the table itself is top level.
//
// IMPORTANT: Assumes that GIL is held.
void PreprocessInputForStackedTablePerLocalDevice(
    const absl::Span<const StackedTableMetadata> stacked_table_metadata,
    py::list features, py::list feature_weights, const int local_device_id,
    const int local_device_count, const int coo_buffer_size,
    const int row_pointers_size_per_sc, const int num_global_devices,
    const int num_sc_per_device, const int sharding_strategy,
    const absl::string_view stacked_table_name, const bool allow_id_dropping,
    py::array_t<int> row_pointer_buffer, py::array_t<int> embedding_id_buffer,
    py::array_t<int> sample_id_buffer, py::array_t<float> gain_buffer,
    py::array_t<int> max_ids_buffer, py::array_t<int> max_unique_ids_buffer,
    py::array_t<int> required_buffer_size_per_sc_buffer) {
  const int num_scs = num_sc_per_device * num_global_devices;
  int batch_size_for_device = 0;
  int total_num_coo_tensors = 0;

  //
  // Step 1: Extract the COO tensors for each feature.
  //

  // Note that the stacked_table_metadata list is sorted by row offsets of the
  // features.
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
  row_pointer_buffer[py::make_tuple(py::ellipsis())] = coo_buffer_size;

  auto* row_pointer_data = row_pointer_buffer.mutable_data();
  auto* embedding_ids_data = embedding_id_buffer.mutable_data();
  auto* sample_ids_data = sample_id_buffer.mutable_data();
  auto* gains_data = gain_buffer.mutable_data();
  auto* total_max_ids_per_sc = max_ids_buffer.mutable_data();
  auto* total_max_unique_ids_per_sc = max_unique_ids_buffer.mutable_data();
  auto* required_buffer_size_per_sc =
      required_buffer_size_per_sc_buffer.mutable_data();
  // The remaining section does not require GIL.
  py::gil_scoped_release release;

  //
  // Step 2: Sort the COO tensors and group them by SC.
  //
  const int batch_size_per_sc =
      CeilOfRatio(batch_size_for_device, num_sc_per_device);

  std::vector<std::vector<CooFormat>> coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          coo_tensors, batch_size_per_sc, num_scs, batch_size_for_device,
          stacked_table_metadata[0].max_ids_per_partition,
          stacked_table_metadata[0].max_unique_ids_per_partition,
          stacked_table_name, allow_id_dropping, num_sc_per_device,
          total_num_coo_tensors, total_max_ids_per_sc,
          total_max_unique_ids_per_sc, required_buffer_size_per_sc);
  for (int i = 0; i < num_sc_per_device; ++i) {
    coo_tensors_by_id[i].emplace_back(batch_size_per_sc * (i + 1), 0, 0.0);
  }
  //
  // Step 3: Compute the row pointers for each group of IDs.
  //
  {
    const int coo_buffer_size_per_sc = coo_buffer_size / num_sc_per_device;
    FillRowPointersPerLocalDevice(
        coo_tensors_by_id, row_pointers_size_per_sc, coo_buffer_size_per_sc,
        batch_size_per_sc, num_scs, num_sc_per_device, row_pointer_data,
        embedding_ids_data, sample_ids_data, gains_data);
  }
}

// Helper function to get the shape container for the output arrays.
// If `has_leading_dimension` is true, the shape will be
// [local_device_count, inner_dim_size]. Otherwise, the shape will be
// [local_device_count * inner_dim_size].
static inline py::array::ShapeContainer GetArrayShapeBasedOnLeadingDim(
    bool has_leading_dimension, int local_device_count, int inner_dim_size) {
  return has_leading_dimension
             ? py::array::ShapeContainer({local_device_count, inner_dim_size})
             : py::array::ShapeContainer({local_device_count * inner_dim_size});
}

// Helper function to get the slice for a given device.
// If `has_leading_dimension` is true, the slice will be
// [device_index:device_index+1]. Otherwise, the slice will be
// [device_index * first_dim_size:(device_index + 1) * first_dim_size].
static inline py::slice GetBufferSliceForGivenDevice(bool has_leading_dimension,
                                                     int start_index,
                                                     int first_dim_size) {
  return has_leading_dimension
             ? py::slice(start_index, start_index + 1, 1)
             : py::slice(start_index * first_dim_size,
                         (start_index + 1) * first_dim_size, 1);
}

py::tuple PreprocessSparseDenseMatmulInput(
    py::list features, py::list feature_weights, py::list feature_specs,
    const int local_device_count, const int global_device_count,
    const int num_sc_per_device, const int sharding_strategy,
    const bool has_leading_dimension, const bool allow_id_dropping) {
  tsl::profiler::TraceMe t([=] {
    return absl::StrCat("input_preprocessing_cc-", local_device_count, "/",
                        global_device_count);
  });
  // Only mod sharding is supported for now.
  CHECK_EQ(sharding_strategy, 1);
  CHECK_GT(local_device_count, 0);
  CHECK(features.size() == feature_weights.size());
  CHECK(features.size() == feature_specs.size());

  py::dict lhs_row_pointers;
  py::dict lhs_embedding_ids;
  py::dict lhs_sample_ids;
  py::dict lhs_gains;
  py::dict max_ids_per_partition;
  py::dict max_unique_ids_per_partition;
  py::dict required_buffer_sizes;
  const int num_scs = num_sc_per_device * global_device_count;
  const int row_pointers_size_per_sc =
      std::max(num_scs, TPU_VECTOR_REGISTER_ALIGMENT_SIZE);

  // Get the stacked table metadata for each top level table.
  // The keys are stacked table names (or the table itself if not stacked) and
  // the values are a vector of StackedTableMetadata for each feature that is
  // mapped to the table.
  const absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>
      stacked_tables = GetStackedTableMetadata(feature_specs, features);

  // Main thread release GIL so that the other threads can acquire / release.
  // The input preprocessing is essentially broken into 3 parts.
  //   1. COO extraction
  //   2. Sorting
  //   3. Fill row pointers.
  // Most of these can be parallelized, except part (1). For part (1), although
  // 2D numpy arrays (rectangles) can be run in parallel, 1D arrays require
  // casting each sample to a numpy array.
  absl::BlockingCounter counter(stacked_tables.size());

  tsl::profiler::TraceMeProducer producer("InputPreprocessingMainThread");
  {
    py::gil_scoped_release main_release;
    for (const auto& [stacked_table_name, stacked_table_metadata] :
         stacked_tables) {
      PreprocessingThreadPool()->Schedule([&, context_id =
                                                  producer.GetContextId()] {
        tsl::profiler::TraceMeConsumer consumer(
            [&] {
              return absl::StrCat("InputPreprocessingTable-",
                                  stacked_table_name);
            },
            context_id);
        // Allocate the static buffers.
        const int coo_buffer_size_per_device = ComputeCooBufferSize(
            num_scs, num_sc_per_device, stacked_table_metadata);

        // Acquire GIL before creating Python arrays.
        py::gil_scoped_acquire acq;
        py::array_t<int> row_pointers_per_device =
            py::array_t<int>(GetArrayShapeBasedOnLeadingDim(
                has_leading_dimension, local_device_count,
                row_pointers_size_per_sc * num_sc_per_device));

        py::array::ShapeContainer shape_container =
            GetArrayShapeBasedOnLeadingDim(has_leading_dimension,
                                           local_device_count,
                                           coo_buffer_size_per_device);
        py::array_t<int> embedding_ids_per_device =
            py::array_t<int>(shape_container);
        py::array_t<int> sample_ids_per_device =
            py::array_t<int>(shape_container);
        py::array_t<float> gains_per_device =
            py::array_t<float>(shape_container);
        const int stats_size_per_device = num_scs;
        py::array::ShapeContainer stats_shape = GetArrayShapeBasedOnLeadingDim(
            /*has_leading_dimension=*/false, local_device_count,
            stats_size_per_device);
        py::array_t<int> max_ids_per_partition_per_sc =
            py::array_t<int>(stats_shape);
        py::array_t<int> max_unique_ids_per_partition_per_sc =
            py::array_t<int>(stats_shape);
        py::array_t<int> required_buffer_size_per_sc =
            py::array_t<int>(stats_shape);
        for (int local_device = 0; local_device < local_device_count;
             ++local_device) {
          // Get the tuple outputs for the current split.
          auto row_pointer_buffer =
              row_pointers_per_device[GetBufferSliceForGivenDevice(
                  has_leading_dimension, local_device,
                  row_pointers_size_per_sc * num_sc_per_device)];
          py::slice static_buffer_slice = GetBufferSliceForGivenDevice(
              has_leading_dimension, local_device, coo_buffer_size_per_device);
          auto embedding_id_buffer =
              embedding_ids_per_device[static_buffer_slice];
          auto sample_id_buffer = sample_ids_per_device[static_buffer_slice];
          auto gain_buffer = gains_per_device[static_buffer_slice];
          py::slice stats_slice =
              GetBufferSliceForGivenDevice(/*has_leading_dimension=*/false,
                                           local_device, stats_size_per_device);
          auto max_ids_per_partition_per_sc_buffer =
              max_ids_per_partition_per_sc[stats_slice];
          auto max_unique_ids_per_partition_per_sc_buffer =
              max_unique_ids_per_partition_per_sc[stats_slice];
          auto required_buffer_size_per_sc_buffer =
              required_buffer_size_per_sc[stats_slice];
          PreprocessInputForStackedTablePerLocalDevice(
              stacked_table_metadata, features, feature_weights, local_device,
              local_device_count, coo_buffer_size_per_device,
              row_pointers_size_per_sc, global_device_count, num_sc_per_device,
              sharding_strategy, stacked_table_name, allow_id_dropping,
              py::cast<py::array_t<int>>(row_pointer_buffer),
              py::cast<py::array_t<int>>(embedding_id_buffer),
              py::cast<py::array_t<int>>(sample_id_buffer),
              py::cast<py::array_t<float>>(gain_buffer),
              py::cast<py::array_t<int>>(max_ids_per_partition_per_sc_buffer),
              py::cast<py::array_t<int>>(
                  max_unique_ids_per_partition_per_sc_buffer),
              py::cast<py::array_t<int>>(required_buffer_size_per_sc_buffer));
        }
        lhs_row_pointers[stacked_table_name.c_str()] =
            std::move(row_pointers_per_device);
        lhs_embedding_ids[stacked_table_name.c_str()] =
            std::move(embedding_ids_per_device);
        lhs_sample_ids[stacked_table_name.c_str()] =
            std::move(sample_ids_per_device);
        lhs_gains[stacked_table_name.c_str()] = std::move(gains_per_device);
        max_ids_per_partition[stacked_table_name.c_str()] =
            std::move(max_ids_per_partition_per_sc);
        max_unique_ids_per_partition[stacked_table_name.c_str()] =
            std::move(max_unique_ids_per_partition_per_sc);
        required_buffer_sizes[stacked_table_name.c_str()] =
            std::move(required_buffer_size_per_sc);
        counter.DecrementCount();
      });
    }
    counter.Wait();
  }
  py::dict stats;
  stats["max_ids"] = max_ids_per_partition;
  stats["max_unique_ids"] = max_unique_ids_per_partition;
  stats["required_buffer_size"] = std::move(required_buffer_sizes);

  // GIL is held at this point.
  return py::make_tuple(lhs_row_pointers, lhs_embedding_ids, lhs_sample_ids,
                        lhs_gains, stats);
}

}  // namespace

PYBIND11_MODULE(input_preprocessing_cc, m) {
  m.def("PreprocessSparseDenseMatmulInput", &PreprocessSparseDenseMatmulInput,
        pybind11::arg("features"), pybind11::arg("feature_weights"),
        pybind11::arg("feature_specs"), pybind11::arg("local_device_count"),
        pybind11::arg("global_device_count"),
        pybind11::arg("num_sc_per_device"), pybind11::arg("sharding_strategy"),
        pybind11::arg("has_leading_dimension"),
        pybind11::arg("allow_id_dropping"));
}

}  // namespace jax_sc_embedding
