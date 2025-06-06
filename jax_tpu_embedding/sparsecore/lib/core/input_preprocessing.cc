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
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/blocking_counter.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "third_party/eigen3/Eigen/Core"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_threads.h"  // IWYU: pragma keep
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/eigen.h"  // from @pybind11
#include "pybind11/eigen/matrix.h"  // from @pybind11
#include "pybind11/gil.h"  // from @pybind11
#include "pybind11/numpy.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "third_party/pybind11_abseil/absl_casters.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/traceme.h"

namespace jax_sc_embedding {

namespace {

namespace py = ::pybind11;

// Input Structures (To be moved to a separate file soon).
template <typename Scalar>
using Sample1D = py::EigenDRef<
    const Eigen::Matrix<Scalar, 1, Eigen::Dynamic, Eigen::RowMajor>>;
template <typename Scalar>
using Samples1D = absl::Span<const Sample1D<Scalar>>;
template <typename Scalar>
using Samples2D =
    py::EigenDRef<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                                      Eigen::RowMajor>>;
template <typename Scalar>
struct FeatureWrapper {
  std::variant<Samples1D<Scalar>, Samples2D<Scalar>> samples;
  std::vector<Sample1D<Scalar>> batch_storage;  // pointers for ragged tensors

  FeatureWrapper() = delete;

  // creation from views
  explicit FeatureWrapper(Samples1D<Scalar> samples)
      : samples(std::move(samples)) {}
  explicit FeatureWrapper(Samples2D<Scalar> samples)
      : samples(std::move(samples)) {}

  // creation from py objects
  explicit FeatureWrapper(py::array_t<py::object> c_samples)
      : samples(c_samples.cast<Samples2D<Scalar>>()) {}

  explicit FeatureWrapper(py::array c_samples) {
    for (const auto& c_sample : c_samples) {
      batch_storage.push_back(c_sample.cast<Sample1D<Scalar>>());
    }
    samples = absl::MakeConstSpan(batch_storage);
  }

  bool is_1d() const {
    return std::holds_alternative<Samples1D<Scalar>>(samples);
  }
  Samples1D<Scalar> Get1D() const {
    return std::get<Samples1D<Scalar>>(samples);
  }
  Samples2D<Scalar> Get2D() const {
    return std::get<Samples2D<Scalar>>(samples);
  }

  int GetSampleCount() const {
    if (is_1d()) {
      return Get1D().size();
    } else {
      return Get2D().rows();
    }
  }

  FeatureWrapper<Scalar> Slice(int start_index, int end_index) const {
    if (is_1d()) {
      const auto& vec = Get1D();
      return FeatureWrapper<Scalar>(
          absl::MakeConstSpan(vec.data(), vec.size())
              .subspan(start_index, end_index - start_index));
    } else {
      return FeatureWrapper<Scalar>(
          Get2D()(Eigen::seq(start_index, end_index), Eigen::indexing::all));
    }
  }
};

using Feature = FeatureWrapper<int>;
using FeatureWeight = FeatureWrapper<float>;

template <typename Derived>
float ComputeWeightDivisor(const RowCombiner combiner,
                           const Eigen::MatrixBase<Derived>& weights) {
  switch (combiner) {
    case RowCombiner::kSum:
      return 1.0f;
    case RowCombiner::kMean: {
      return weights.sum();
    }
    case RowCombiner::kSqrtn: {
      return weights.norm();
    }
  }
}

void ExtractCooTensorsFrom2dArray(Samples2D<int> feature,
                                  Samples2D<float> feature_weight,
                                  const int row_offset, const int col_offset,
                                  const int col_shift, const int num_scs_mod,
                                  const int num_scs_mod_inv,
                                  const int global_device_count,
                                  const RowCombiner combiner,
                                  std::vector<CooFormat>& coo_tensors) {
  const size_t nrows = feature.rows();
  const size_t ncols = feature.cols();

  coo_tensors.reserve(feature.size());
  CHECK_EQ(nrows, feature_weight.rows());
  CHECK_EQ(ncols, feature_weight.cols());
  const int row_offset_per_device = row_offset / global_device_count;
  for (size_t i = 0; i < nrows; ++i) {
    const int row_id = i + row_offset_per_device;
    const float divisor = ComputeWeightDivisor(combiner, feature_weight.row(i));
    for (size_t j = 0; j < ncols; ++j) {
      const int col = feature(i, j);
      const float gain = feature_weight(i, j) / divisor;
      coo_tensors.emplace_back(
          row_id,
          GetColId(col, col_shift, col_offset, num_scs_mod, num_scs_mod_inv),
          gain);
    }
  }
}

void ExtractCooTensorsFrom1dArray(const Samples1D<int>& feature,
                                  const Samples1D<float>& feature_weight,
                                  const int row_offset, const int col_offset,
                                  const int col_shift, const int num_scs_mod,
                                  const int num_scs_mod_inv,
                                  const int global_device_count,
                                  const RowCombiner combiner,
                                  std::vector<CooFormat>& coo_tensors) {
  coo_tensors.reserve(feature.size());
  int coo_tensors_extracted = 0;

  const int row_offset_per_device = row_offset / global_device_count;
  for (int i = 0; i < feature.size(); ++i) {
    Sample1D<int> curr_samples = feature[i];
    Sample1D<float> curr_sample_weights = feature_weight[i];
    const ssize_t sample_count = curr_samples.size();
    CHECK_EQ(feature[i].size(), feature_weight[i].size());
    coo_tensors_extracted += sample_count;
    const int row_id = i + row_offset_per_device;
    const float divisor = ComputeWeightDivisor(combiner, curr_sample_weights);
    for (int j = 0; j < sample_count; ++j) {
      const float gain = curr_sample_weights(j) / divisor;
      coo_tensors.emplace_back(row_id,
                               GetColId(curr_samples(j), col_shift, col_offset,
                                        num_scs_mod, num_scs_mod_inv),
                               gain);
    }
  }
}

void ExtractCooTensors(Feature& feature, FeatureWeight& feature_weight,
                       const int row_offset, const int col_offset,
                       const int col_shift, const int num_scs,
                       const int global_device_count,
                       const RowCombiner combiner,
                       std::vector<CooFormat>& coo_tensors) {
  tsl::profiler::TraceMe t([] { return "ExtractCooTensors"; });
  CHECK(num_scs > 0 && (num_scs & (num_scs - 1)) == 0);
  const int num_scs_bit = std::log2(num_scs);
  const int num_scs_mod = (1 << num_scs_bit) - 1;
  const int num_scs_mod_inv = ~num_scs_mod;
  if (feature.is_1d()) {
    ExtractCooTensorsFrom1dArray(feature.Get1D(), feature_weight.Get1D(),
                                 row_offset, col_offset, col_shift, num_scs_mod,
                                 num_scs_mod_inv, global_device_count, combiner,
                                 coo_tensors);
  } else {
    ExtractCooTensorsFrom2dArray(feature.Get2D(), feature_weight.Get2D(),
                                 row_offset, col_offset, col_shift, num_scs_mod,
                                 num_scs_mod_inv, global_device_count, combiner,
                                 coo_tensors);
  }
}

absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>
GetStackedTableMetadata(py::list& feature_specs,
                        absl::Span<const Feature> features) {
  tsl::profiler::TraceMe t([] { return "GetStackedTableMetadata"; });
  absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>
      stacked_table_metadata;
  py::gil_scoped_acquire acq;
  for (int i = 0; i < feature_specs.size(); ++i) {
    const py::object& feature_spec = feature_specs[i];
    const Feature& feature = features[i];
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
        /*batch_size=*/feature.GetSampleCount(), suggested_coo_buffer_size,
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

// Extract the COO tensors for all features.
ExtractedCooTensors ExtractCooTensorsForAllFeatures(
    absl::Span<const StackedTableMetadata> stacked_table_metadata,
    absl::Span<const Feature> features,
    absl::Span<const FeatureWeight> feature_weights, const int local_device_id,
    const int local_device_count, const int num_scs,
    const int num_global_devices) {
  py::gil_scoped_acquire acq;
  ExtractedCooTensors extracted_coo_tensors;
  for (int i = 0; i < stacked_table_metadata.size(); ++i) {
    const StackedTableMetadata& metadata = stacked_table_metadata[i];
    const int feature_index = metadata.feature_index;
    const int row_offset = metadata.row_offset;
    const int col_offset = metadata.col_offset;
    const int col_shift = metadata.col_shift;
    const Feature& curr_feature = features[feature_index];
    const FeatureWeight& curr_feature_weight = feature_weights[feature_index];

    const int num_samples = curr_feature.GetSampleCount();
    const int num_samples_per_split = num_samples / local_device_count;
    const int start_index = local_device_id * num_samples_per_split;
    int end_index = (local_device_id + 1) * num_samples_per_split;
    if (local_device_id == local_device_count - 1) {
      // Just in case the last split is not a full batch.
      end_index = num_samples;
    }
    Feature feature_split = curr_feature.Slice(start_index, end_index);
    FeatureWeight feature_weights_split =
        curr_feature_weight.Slice(start_index, end_index);
    extracted_coo_tensors.batch_size_for_device +=
        feature_split.GetSampleCount();

    // In the case of feature stacking, we need to group all the COO tensors
    // at this stage (i.e., before the sorting later on).
    ExtractCooTensors(feature_split, feature_weights_split, row_offset,
                      col_offset, col_shift, num_scs, num_global_devices,
                      metadata.row_combiner, extracted_coo_tensors.coo_tensors);
  }
  return extracted_coo_tensors;
}

// Preprocess inputs for a single table. Stacked table here refers to a
// a table that has no parent in the table stacking hierarchy. So in the case
// of table stacking, the stacked table is the top level table and in the case
// where we don't have any table stacking, the table itself is top level.
void PreprocessInputForStackedTablePerLocalDevice(
    const absl::Span<const StackedTableMetadata> stacked_table_metadata,
    absl::Span<const Feature> features,
    absl::Span<const FeatureWeight> feature_weights, const int local_device_id,
    const PreprocessSparseDenseMatmulInputOptions& options,
    const int coo_buffer_size, const int row_pointers_size_per_sc,
    const absl::string_view stacked_table_name,
    Eigen::Ref<RowVectorXi> row_pointer_buffer,
    Eigen::Ref<RowVectorXi> embedding_id_buffer,
    Eigen::Ref<RowVectorXi> sample_id_buffer,
    Eigen::Ref<RowVectorXf> gain_buffer, Eigen::Ref<RowVectorXi> max_ids_buffer,
    Eigen::Ref<RowVectorXi> max_unique_ids_buffer,
    Eigen::Ref<RowVectorXi> required_buffer_size_per_sc_buffer) {
  const int num_scs = options.GetNumScs();

  //
  // Step 1: Extract the COO tensors for each feature.
  //

  // Note that the stacked_table_metadata list is sorted by row offsets of the
  // features.

  ExtractedCooTensors extracted_coo_tensors = ExtractCooTensorsForAllFeatures(
      stacked_table_metadata, features, feature_weights, local_device_id,
      options.local_device_count, num_scs, options.global_device_count);

  int total_num_coo_tensors = extracted_coo_tensors.coo_tensors.size();

  row_pointer_buffer.setConstant(coo_buffer_size);

  //
  // Step 2: Sort the COO tensors and group them by SC.
  //
  const int batch_size_per_sc = CeilOfRatio(
      extracted_coo_tensors.batch_size_for_device, options.num_sc_per_device);

  std::vector<std::vector<CooFormat>> coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          extracted_coo_tensors.coo_tensors, batch_size_per_sc, num_scs,
          extracted_coo_tensors.batch_size_for_device,
          stacked_table_metadata[0].max_ids_per_partition,
          stacked_table_metadata[0].max_unique_ids_per_partition,
          stacked_table_name, options.allow_id_dropping,
          options.num_sc_per_device, total_num_coo_tensors, max_ids_buffer,
          max_unique_ids_buffer, required_buffer_size_per_sc_buffer);
  for (int i = 0; i < options.num_sc_per_device; ++i) {
    coo_tensors_by_id[i].emplace_back(batch_size_per_sc * (i + 1), 0, 0.0);
    required_buffer_size_per_sc_buffer[i]++;
  }
  //
  // Step 3: Compute the row pointers for each group of IDs.
  //
  const int coo_buffer_size_per_sc =
      coo_buffer_size / options.num_sc_per_device;
  FillRowPointersPerLocalDevice(
      coo_tensors_by_id, row_pointers_size_per_sc, coo_buffer_size_per_sc,
      batch_size_per_sc, num_scs, options.num_sc_per_device, row_pointer_buffer,
      embedding_id_buffer, sample_id_buffer, gain_buffer);
}

PreprocessSparseDenseMatmulOutput PreprocessSparseDenseMatmulInput(
    absl::Span<const Feature> features,
    absl::Span<const FeatureWeight> feature_weights, py::list& feature_specs,
    const PreprocessSparseDenseMatmulInputOptions& options) {
  tsl::profiler::TraceMe t([=] {
    return absl::StrCat("input_preprocessing_cc-", options.local_device_count,
                        "/", options.global_device_count);
  });
  // Only mod sharding is supported for now.
  CHECK_EQ(options.sharding_strategy, 1);
  CHECK_GT(options.local_device_count, 0);

  absl::Mutex mutex;
  PreprocessSparseDenseMatmulOutput out;
  const int num_scs = options.GetNumScs();
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
        const int coo_buffer_size_per_device = ComputeCooBufferSizePerDevice(
            num_scs, options.num_sc_per_device, stacked_table_metadata);

        MatrixXi row_pointers_per_device(
            options.local_device_count,
            row_pointers_size_per_sc * options.num_sc_per_device);
        MatrixXi embedding_ids_per_device(options.local_device_count,
                                          coo_buffer_size_per_device);
        MatrixXi sample_ids_per_device(options.local_device_count,
                                       coo_buffer_size_per_device);
        MatrixXf gains_per_device(options.local_device_count,
                                  coo_buffer_size_per_device);

        const int stats_size_per_device = num_scs;
        // NOTE: max ids and max unique ids are {global_sc_count *
        //   num_devices}, where they are then aggregated(max) along device
        //   dimension to get {global_sc_count} (i.e. max [unique] ids for each
        //   sc), which can be further aggregated(max) for a single value for
        //   all SCs.
        MatrixXi max_ids_per_partition_per_sc(options.local_device_count,
                                              stats_size_per_device);
        MatrixXi max_unique_ids_per_partition_per_sc(options.local_device_count,
                                                     stats_size_per_device);
        // NOTE: required buffer size is {local_sc_count * num_devices}, which
        //   is same as {global_sc_count}, and can be further aggregated to get
        //   the maximum size of any SC buffer shard.
        MatrixXi required_buffer_size_per_sc(options.local_device_count,
                                             options.num_sc_per_device);
        for (int local_device = 0; local_device < options.local_device_count;
             ++local_device) {
          // Get the tuple outputs for the current split.
          Eigen::Ref<RowVectorXi> row_pointer_buffer =
              row_pointers_per_device.row(local_device);
          Eigen::Ref<RowVectorXi> embedding_id_buffer =
              embedding_ids_per_device.row(local_device);
          Eigen::Ref<RowVectorXi> sample_id_buffer =
              sample_ids_per_device.row(local_device);
          Eigen::Ref<RowVectorXf> gain_buffer =
              gains_per_device.row(local_device);
          Eigen::Ref<RowVectorXi> max_ids_per_partition_per_sc_buffer =
              max_ids_per_partition_per_sc.row(local_device);
          Eigen::Ref<RowVectorXi> max_unique_ids_per_partition_per_sc_buffer =
              max_unique_ids_per_partition_per_sc.row(local_device);
          Eigen::Ref<RowVectorXi> required_buffer_size_per_sc_buffer =
              required_buffer_size_per_sc.row(local_device);
          PreprocessInputForStackedTablePerLocalDevice(
              stacked_table_metadata, features, feature_weights, local_device,
              options, coo_buffer_size_per_device, row_pointers_size_per_sc,
              stacked_table_name, row_pointer_buffer, embedding_id_buffer,
              sample_id_buffer, gain_buffer,
              max_ids_per_partition_per_sc_buffer,
              max_unique_ids_per_partition_per_sc_buffer,
              required_buffer_size_per_sc_buffer);
        }
        max_ids_per_partition_per_sc.resize(
            1, max_ids_per_partition_per_sc.size());
        max_unique_ids_per_partition_per_sc.resize(
            1, max_unique_ids_per_partition_per_sc.size());
        required_buffer_size_per_sc.resize(1,
                                           required_buffer_size_per_sc.size());
        {
          // This used to be (unintentionally) synchronized using GIL, but
          // there's a possible race condition with the threadpool without the
          // python objects since we don't use the GIL anymore.
          absl::MutexLock lock(&mutex);
          out.lhs_row_pointers[stacked_table_name.c_str()] =
              std::move(row_pointers_per_device);
          out.lhs_embedding_ids[stacked_table_name.c_str()] =
              std::move(embedding_ids_per_device);
          out.lhs_sample_ids[stacked_table_name.c_str()] =
              std::move(sample_ids_per_device);
          out.lhs_gains[stacked_table_name.c_str()] =
              std::move(gains_per_device);

          out.stats.max_ids_per_partition[stacked_table_name.c_str()] =
              std::move(max_ids_per_partition_per_sc);
          out.stats.max_unique_ids_per_partition[stacked_table_name.c_str()] =
              std::move(max_unique_ids_per_partition_per_sc);
          out.stats.required_buffer_sizes[stacked_table_name.c_str()] =
              std::move(required_buffer_size_per_sc);
        }
        counter.DecrementCount();
      });
    }
    counter.Wait();
  }

  return out;
}

py::tuple PyNumpyPreprocessSparseDenseMatmulInput(
    absl::Span<const py::array> features,
    absl::Span<const py::array> feature_weights, py::list feature_specs,
    int local_device_count, int global_device_count, int num_sc_per_device,
    int sharding_strategy, bool has_leading_dimension, bool allow_id_dropping) {
  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = local_device_count,
      .global_device_count = global_device_count,
      .num_sc_per_device = num_sc_per_device,
      .sharding_strategy = sharding_strategy,
      .allow_id_dropping = allow_id_dropping,
  };
  // Union type implicit conversion from python runs into problems, we need to
  // construct the proxy objects. We have union type because the features could
  // be heterogeneous/mixed. We were doing this either way earlier. We could
  // wrap std::vector<Feature> into a FeatureList class to enable implicit
  // conversions and abstract this explicit construction here.
  CHECK_EQ(features.size(), feature_weights.size());
  std::vector<Feature> features_;
  std::vector<FeatureWeight> feature_weights_;
  features_.reserve(features.size());
  feature_weights_.reserve(feature_weights.size());
  for (int i = 0; i < features.size(); ++i) {
    features_.emplace_back(features[i]);
    feature_weights_.emplace_back(feature_weights[i]);
  }
  PreprocessSparseDenseMatmulOutput out;
  {
    // We release the lock by default and acquire it when we deal with python
    // objects (features, specs and weights).
    py::gil_scoped_release release;
    out = PreprocessSparseDenseMatmulInput(features_, feature_weights_,
                                           feature_specs, options);
  }
  // need the GIL back to create tuple. The tuple creation implicitly wraps
  // Eigen matrices into numpy arrays (without copying), which makes it easier
  // to flatten them using `reshape(-1)`.
  py::tuple ret_object =
      py::make_tuple(out.lhs_row_pointers, out.lhs_embedding_ids,
                     out.lhs_sample_ids, out.lhs_gains, out.stats);
  // Skip the last element (stats).
  for (size_t i = 0; i < ret_object.size() - 1; ++i) {
    if (!has_leading_dimension) {
      for (auto& iterator : ret_object[i].cast<py::dict>()) {
        ret_object[i][iterator.first] =
            py::cast<py::array>(iterator.second).reshape({-1});
      }
    }
  }
  return ret_object;
}

}  // namespace

PYBIND11_MODULE(input_preprocessing_cc, m) {
  m.def("PreprocessSparseDenseMatmulInput",
        &PyNumpyPreprocessSparseDenseMatmulInput, pybind11::arg("features"),
        pybind11::arg("feature_weights"), pybind11::arg("feature_specs"),
        pybind11::arg("local_device_count"),
        pybind11::arg("global_device_count"),
        pybind11::arg("num_sc_per_device"), pybind11::arg("sharding_strategy"),
        pybind11::arg("has_leading_dimension"),
        pybind11::arg("allow_id_dropping"));
  py::class_<SparseDenseMatmulInputStats>(m, "SparseDenseMatmulInputStats")
      .def(py::init<>())
      .def_readonly("max_ids_per_partition",
                    &SparseDenseMatmulInputStats::max_ids_per_partition)
      .def_readonly("max_unique_ids_per_partition",
                    &SparseDenseMatmulInputStats::max_unique_ids_per_partition)
      .def_readonly("required_buffer_sizes",
                    &SparseDenseMatmulInputStats::required_buffer_sizes);
}

}  // namespace jax_sc_embedding
