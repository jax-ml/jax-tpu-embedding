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
#include <bitset>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/blocking_counter.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "Eigen/Core"  // from @eigen_archive
#include "jax_tpu_embedding/sparsecore/lib/core/abstract_input_batch.h"
#include "jax_tpu_embedding/sparsecore/lib/core/coo_format.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "jax_tpu_embedding/sparsecore/lib/core/partitioned_coo_tensors.h"
#include "jax_tpu_embedding/sparsecore/lib/core/sort_and_group_coo_tensors_impl.h"
#include "xla/tsl/concurrency/async_value.h"  // from @xla
#include "xla/tsl/concurrency/async_value_ref.h"  // from @xla
#include "tsl/platform/env.h"  // from @tsl
#include "tsl/platform/statusor.h"  // from @tsl
#include "xla/util.h"  // from @xla
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"

namespace jax_sc_embedding {

namespace {

struct FeatureSliceInfo {
  int local_device_id;
  int feature_slice_id;
  int feature_slices_per_device;
  int start_index;
  int end_index;
  int batch_size_per_slice;
};

struct FeatureSlice {
  FeatureSliceInfo info;
  const FeatureMetadataInStack* metadata;
};

void AllocateOutputCsrBuffersIfNeeded(
    MatrixXi& row_pointers, MatrixXi& embedding_ids, MatrixXi& sample_ids,
    MatrixXf& gains, const PreprocessSparseDenseMatmulInputOptions& options,
    int row_pointers_size_per_device, int coo_buffer_size_per_device) {
  row_pointers.resize(options.local_device_count, row_pointers_size_per_device);
  embedding_ids.resize(options.local_device_count, coo_buffer_size_per_device);
  sample_ids.resize(options.local_device_count, coo_buffer_size_per_device);
  gains.resize(options.local_device_count, coo_buffer_size_per_device);
}

// Extract the COO tensors for a single feature slice.
void ExtractCooTensorsForSingleFeatureSlice(
    const FeatureMetadataInStack& metadata,
    absl::Span<std::unique_ptr<AbstractInputBatch>> input_batches,
    const FeatureSliceInfo& slice_info,
    const PreprocessSparseDenseMatmulInputOptions& options,
    ExtractedCooTensorsPerSparseCore& extracted_coo_tensors) {
  const int feature_index = metadata.feature_index;
  const std::unique_ptr<AbstractInputBatch>& curr_batch =
      input_batches[feature_index];

  CHECK_GT(slice_info.feature_slices_per_device, 0);
  CHECK_GT(options.global_device_count, 0);
  CHECK_GT(options.local_device_count, 0);

  const int row_offset_per_slice =
      metadata.row_offset /
      (options.global_device_count * slice_info.feature_slices_per_device);
  const int row_offset =
      slice_info.feature_slice_id * slice_info.batch_size_per_slice +
      row_offset_per_slice;
  const int col_offset = metadata.col_offset;
  const int col_shift = metadata.col_shift;

  // In the case of feature stacking, we need to group all the COO tensors
  // at this stage (i.e., before the sorting later on).
  VLOG(2) << absl::StrFormat(
      "Extracting COO Tensor from feature #%d from row %d to %d "
      "(local_device_id = %d, feature_slice_id = %d, row_offset = %d, "
      "batch_size_per_slice = %d)",
      feature_index, slice_info.start_index, slice_info.end_index,
      slice_info.local_device_id, slice_info.feature_slice_id, row_offset,
      slice_info.batch_size_per_slice);
  curr_batch->ExtractCooTensors(
      {
          .slice_start = slice_info.start_index,
          .slice_end = slice_info.end_index,
          .row_offset = row_offset,
          .col_offset = col_offset,
          .col_shift = col_shift,
          .num_sc_per_device = options.num_sc_per_device,
          .num_scs = options.GetNumScs(),
          .combiner = metadata.row_combiner,
      },
      extracted_coo_tensors);
}

void CheckDeviceBatchSize(int batch_size_for_device, int num_sc_per_device,
                          absl::string_view stacked_table_name) {
  CHECK_EQ(batch_size_for_device % num_sc_per_device, 0) << absl::StrFormat(
      "batch_size_for_device (%d) for stacked table '%s' must be divisible by "
      "num_sc_per_device (%d).",
      batch_size_for_device, stacked_table_name, num_sc_per_device);
}

// We consider a stack to have variable weights if any feature in the stack
// has explicitly variable weights.
bool StackHasVariableWeights(
    absl::Span<std::unique_ptr<AbstractInputBatch>> input_batches,
    absl::Span<const FeatureMetadataInStack> stacked_table_metadata) {
  for (const auto& metadata : stacked_table_metadata) {
    if (input_batches[metadata.feature_index]->HasVariableWeights()) {
      return true;
    }
  }
  return false;
}

// Holds the state for processing a single stacked table across all local
// devices. This includes extracted COO tensors, partitioned COO tensors,
// CSR arrays, and statistics.
struct TableState {
  absl::string_view stacked_table_name;
  absl::Span<const FeatureMetadataInStack> stacked_table_metadata;
  bool has_variable_weights;
  int coo_buffer_size_per_device;
  CsrArraysPerHost csr_arrays_per_host;
  StatsPerHost stats_per_host;
  int batch_size_for_device;
  bool table_minibatching_required = false;
  MinibatchingSplit table_minibatching_split = 0;
  std::vector<ExtractedCooTensors> extracted_coo_tensors_per_device;
  std::vector<tsl::AsyncValueRef<DeviceSortingTaskResult>>
      device_sorting_results;

  TableState(absl::string_view name,
             absl::Span<const FeatureMetadataInStack> metadata,
             bool has_variable_weights,
             const PreprocessSparseDenseMatmulInputOptions& options,
             int num_scs, int coo_buffer_size_per_device,
             Eigen::Ref<MatrixXi> row_pointers,
             Eigen::Ref<MatrixXi> embedding_ids,
             Eigen::Ref<MatrixXi> sample_ids, Eigen::Ref<MatrixXf> gains)
      : stacked_table_name(name),
        stacked_table_metadata(metadata),
        has_variable_weights(has_variable_weights),
        coo_buffer_size_per_device(coo_buffer_size_per_device),
        csr_arrays_per_host(row_pointers, embedding_ids, sample_ids, gains),
        stats_per_host(options.local_device_count, options.GetNumScs(),
                       options.num_sc_per_device),
        batch_size_for_device(0) {
    extracted_coo_tensors_per_device.resize(options.local_device_count);
    device_sorting_results.resize(options.local_device_count);
  }
};

template <typename SplitType>
tsl::AsyncValueRef<DeviceSortingTaskResult>
SortAndGroupCooTensorsForTableStateAsync(
    TableState& state, int local_device,
    const PreprocessSparseDenseMatmulInputOptions& options,
    internal::StatsPerDevice stats) {
  if (state.has_variable_weights) {
    return SortAndGroupCooTensorsPerLocalDeviceAsync<true, SplitType>(
        state.extracted_coo_tensors_per_device[local_device],
        state.stacked_table_name, state.stacked_table_metadata[0], options,
        stats);
  } else {
    return SortAndGroupCooTensorsPerLocalDeviceAsync<false, SplitType>(
        state.extracted_coo_tensors_per_device[local_device],
        state.stacked_table_name, state.stacked_table_metadata[0], options,
        stats);
  }
}

// Extracts, sorts, and groups COO tensors for a single stacked table across
// all local devices. This function populates
// `state.extracted_coo_tensors_per_device` and
// `state.partitioned_coo_tensors_per_device`.
void ExtractSortAndGroupCooTensorsForTable(
    TableState& state,
    absl::Span<std::unique_ptr<AbstractInputBatch>> input_batches,
    const PreprocessSparseDenseMatmulInputOptions& options) {
  tsl::profiler::TraceMe traceme([&] {
    return tsl::profiler::TraceMeEncode(
        absl::StrCat("ScheduleExtractSortGroup/", state.stacked_table_name),
        {{"batch_number", options.batch_number}});
  });
  for (int local_device = 0; local_device < options.local_device_count;
       ++local_device) {
    // This extracts Coo tensors per SC in parallel by launching one async
    // task per SC via async_task_scheduler within
    // ExtractCooTensorsForAllFeaturesPerLocalDevice.
    state.extracted_coo_tensors_per_device[local_device] =
        internal::ExtractCooTensorsForAllFeaturesPerLocalDevice(
            state.stacked_table_metadata, input_batches, local_device, options,
            state.has_variable_weights);

    internal::StatsPerDevice stats_per_device =
        state.stats_per_host.GetStatsPerDevice(local_device);

    // This sorts and groups Coo tensors per SC in parallel. Each sorting
    // task waits for the corresponding SC extraction to finish before
    // starting (dependency is handled in
    // SortAndGroupCooTensorsPerLocalDeviceImpl via RunWhenReady).
    state.device_sorting_results[local_device] =
        SortAndGroupCooTensorsForTableStateAsync<bool>(
            state, local_device, options, stats_per_device);
  }
}

void PostProcessTableState(TableState& state) {
  state.stats_per_host.dropped_id_count = 0;
  for (const auto& result_av : state.device_sorting_results) {
    state.stats_per_host.dropped_id_count +=
        result_av.get().total_dropped_id_count;
    state.table_minibatching_required |=
        result_av.get().table_minibatching_required;
    state.table_minibatching_split |= result_av.get().table_minibatching_split;
  }
}

// Creates minibatching buckets for a single stacked table across all local
// devices. This function re-sorts and groups the extracted COO tensors
// based on the minibatching split determined in a previous stage.
// `state`: The TableState holding the COO tensors and statistics.
// `options`: Preprocessing options.
void CreateMinibatchingBucketsForTable(
    TableState& state, const PreprocessSparseDenseMatmulInputOptions& options) {
  tsl::profiler::TraceMe traceme([&] {
    return tsl::profiler::TraceMeEncode(
        absl::StrCat("ScheduleCreateMinibatchingBuckets/",
                     state.stacked_table_name),
        {{"batch_number", options.batch_number}});
  });
  state.stats_per_host.dropped_id_count = 0;
  state.device_sorting_results.assign(
      options.local_device_count,
      tsl::AsyncValueRef<DeviceSortingTaskResult>());
  absl::BlockingCounter counter(options.local_device_count);
  for (int local_device = 0; local_device < options.local_device_count;
       ++local_device) {
    options.async_task_scheduler([&, local_device] {
      // Note: We create a dummy stats object here because we don't want to
      // overwrite the stats from the first pass, which are authoritative.
      // The only stat we care about from this second pass is the number of
      // dropped IDs.
      auto dummy_stats_host = std::make_shared<StatsPerHost>(
          /*local_device_count=*/1, options.GetNumScs(),
          options.num_sc_per_device);
      internal::StatsPerDevice dummy_stats =
          dummy_stats_host->GetStatsPerDevice(0);
      auto result_av =
          SortAndGroupCooTensorsForTableStateAsync<MinibatchingSplit>(
              state, local_device, options, dummy_stats);
      state.device_sorting_results[local_device] = result_av;
      // `dummy_stats_host` must be kept alive until `result_av` is
      // ready, because `dummy_stats` (which is used in the async task
      // that produces `result_av`) holds references to data owned by
      // `dummy_stats_host`. Moving `dummy_stats_host` into this
      // `AndThen` callback ensures its lifetime is extended until the
      // async operation completes.
      result_av.GetAsyncValue()->AndThen(
          [dummy_stats_host = std::move(dummy_stats_host), &counter] {
            counter.DecrementCount();
          });
    });
  }
  // Wait for all local devices to finish.
  counter.Wait();
}

}  // namespace

namespace internal {

inline uint64_t Serialize(MinibatchingSplit value) { return value.to_ullong(); }
inline MinibatchingSplit Deserialize(uint64_t value) {
  return MinibatchingSplit(value);
}

inline bool Serialize(bool value) { return value; }
inline bool Deserialize(bool value) { return value; }

// Extract the COO tensors for all features.
ExtractedCooTensors ExtractCooTensorsForAllFeaturesPerLocalDevice(
    const absl::Span<const FeatureMetadataInStack> stacked_table_metadata,
    absl::Span<std::unique_ptr<AbstractInputBatch>> input_batches,
    const int local_device_id,
    const PreprocessSparseDenseMatmulInputOptions& options,
    bool has_variable_weights) {
  // Calculate the total batch size for this device across all features.
  int batch_size_for_device = 0;
  for (const auto& feature_metadata : stacked_table_metadata) {
    batch_size_for_device +=
        input_batches[feature_metadata.feature_index]->size() /
        options.local_device_count;
  }

  // Determine the number of slices per feature based on stacking strategy.
  // With split-then-stack, each feature is split into num_sc_per_device slices.
  const int feature_slices_per_device = options.num_sc_per_device;

  CheckDeviceBatchSize(batch_size_for_device, options.num_sc_per_device,
                       stacked_table_metadata[0].name);

  CHECK_GE(batch_size_for_device,
           feature_slices_per_device * stacked_table_metadata.size())
      << "Batch size must be greater or equal to the number of "
         "features stacked together (per feature slice).";

  // Initialize COO tensor structure and calculate batch size per slice.
  ExtractedCooTensors extracted_coo_tensors(
      options.num_sc_per_device, batch_size_for_device, has_variable_weights,
      stacked_table_metadata[0].row_combiner);
  const int batch_size_per_slice =
      xla::CeilOfRatio(batch_size_for_device, feature_slices_per_device);

  // Precompute slice information for all features and slices.
  std::vector<FeatureSlice> feature_slices(feature_slices_per_device *
                                           stacked_table_metadata.size());

  bool exact_size_known = true;
  std::vector<int64_t> ids_per_sc(options.num_sc_per_device, 0);

  for (int feature_idx = 0; feature_idx < stacked_table_metadata.size();
       ++feature_idx) {
    const auto& feature_metadata = stacked_table_metadata[feature_idx];
    const std::unique_ptr<AbstractInputBatch>& curr_batch =
        input_batches[feature_metadata.feature_index];
    const int num_samples = curr_batch->size();
    // This division is exact as checked earlier in CheckDeviceBatchSize.
    const int num_samples_per_split =
        num_samples / (options.local_device_count * feature_slices_per_device);
    for (int feature_slice_id = 0; feature_slice_id < feature_slices_per_device;
         ++feature_slice_id) {
      const int start_index =
          (local_device_id * feature_slices_per_device + feature_slice_id) *
          num_samples_per_split;
      int end_index =
          std::min(num_samples, start_index + num_samples_per_split);

      // Store slice information in slice-major order.
      feature_slices[feature_slice_id * stacked_table_metadata.size() +
                     feature_idx] = {
          .info =
              FeatureSliceInfo{
                  .local_device_id = local_device_id,
                  .feature_slice_id = feature_slice_id,
                  .feature_slices_per_device = feature_slices_per_device,
                  .start_index = start_index,
                  .end_index = end_index,
                  .batch_size_per_slice = batch_size_per_slice},
          .metadata = &feature_metadata};

      if (exact_size_known) {
        std::optional<int64_t> ids_in_slice =
            curr_batch->GetIdsCountInSlice(start_index, end_index);
        if (ids_in_slice.has_value()) {
          ids_per_sc[feature_slice_id] += ids_in_slice.value();
        } else {
          exact_size_known = false;
        }
      }
    }
  }

  // If exact size is not known, fall back to using the total id_count per
  // feature.
  if (!exact_size_known) {
    int64_t total_ids_for_device = 0;
    for (const auto& feature_metadata : stacked_table_metadata) {
      total_ids_for_device +=
          input_batches[feature_metadata.feature_index]->id_count() /
          options.local_device_count;
    }
    int64_t ids_per_sc_est = total_ids_for_device / options.num_sc_per_device;
    for (int i = 0; i < options.num_sc_per_device; ++i) {
      ids_per_sc[i] = ids_per_sc_est;
    }
  }

  // Extract COO tensors for each slice asynchronously per SC.
  for (int sc_id = 0; sc_id < options.num_sc_per_device; ++sc_id) {
    auto av = extracted_coo_tensors.per_sc_tensors_av[sc_id];
    options.async_task_scheduler([=] {
      tsl::profiler::TraceMe traceme([&] {
        return tsl::profiler::TraceMeEncode(
            absl::StrCat("ExtractCooTensors/", stacked_table_metadata[0].name,
                         "/SC", sc_id),
            {{"batch_number", options.batch_number}});
      });
      ExtractedCooTensorsPerSparseCore coo_tensors_per_sc(
          batch_size_per_slice, has_variable_weights,
          stacked_table_metadata[0].row_combiner);
      coo_tensors_per_sc.reserve(ids_per_sc[sc_id]);
      for (int feature_idx = 0; feature_idx < stacked_table_metadata.size();
           ++feature_idx) {
        const auto& slice =
            feature_slices[sc_id * stacked_table_metadata.size() + feature_idx];
        ExtractCooTensorsForSingleFeatureSlice(*slice.metadata, input_batches,
                                               slice.info, options,
                                               coo_tensors_per_sc);
      }
      av.emplace(
          std::move(coo_tensors_per_sc));
    });
  }
  return extracted_coo_tensors;
}

}  // namespace internal

namespace {

// Check the buffer usage ratio and log a warning if it exceeds a certain
// threshold.
// `max_required_buffer_size_per_device`: The maximum buffer size required by
// the input for a single device.
// `coo_buffer_size_per_device`: The allocated buffer size per device.
// `stacked_table_name`: The name of the stacked table.
// `batch_number`: The current batch number, used for conditional logging.
void CheckBufferUsage(int max_required_buffer_size_per_device,
                      int coo_buffer_size_per_device,
                      absl::string_view stacked_table_name, int batch_number) {
  CHECK_GT(coo_buffer_size_per_device, 0);
  const double usage_ratio =
      static_cast<double>(max_required_buffer_size_per_device) /
      static_cast<double>(coo_buffer_size_per_device);
  static constexpr double kUsageDivergence = 0.2;
  if (std::abs(usage_ratio - 1.0) >= kUsageDivergence) {
    // The size of one element in the COO buffer, which consists of an embedding
    // ID (int), a sample ID (int), and a gain value (float).
    static constexpr int kElementSize =
        sizeof(int) + sizeof(int) + sizeof(float);

    const int64_t required_buffer_bytes =
        int64_t{max_required_buffer_size_per_device} * kElementSize;
    const int64_t coo_buffer_bytes =
        int64_t{coo_buffer_size_per_device} * kElementSize;

    const int64_t wasted_space =
        std::max(int64_t{0}, coo_buffer_bytes - required_buffer_bytes);
    const int64_t buffer_shortfall =
        std::max(int64_t{0}, required_buffer_bytes - coo_buffer_bytes);

    LOG_IF(WARNING, batch_number % 10000 == 0) << absl::StrFormat(
        "Required usage %.2f%% (%d bytes) of computed/given buffer size "
        "(%d bytes) for stacked table %s (Wasted space: %d bytes, "
        "Buffer shortfall: %d bytes)",
        usage_ratio * 100.0f, required_buffer_bytes, coo_buffer_bytes,
        stacked_table_name, wasted_space, buffer_shortfall);
  }
}

void MergeStats(
    absl::flat_hash_map<std::string, RowVectorXi>& current_stats,
    const absl::flat_hash_map<std::string, RowVectorXi>& other_stats) {
  for (const auto& [table_name, other_values] : other_stats) {
    auto it = current_stats.find(table_name);
    if (it == current_stats.end()) {
      current_stats[table_name] = other_values;
    } else {
      CHECK_EQ(it->second.size(), other_values.size());
      it->second = it->second.cwiseMax(other_values);
    }
  }
}

// Synchronizes the `table_minibatching_required` flag across all participating
// devices. If `options.all_reduce_interface` is provided, it performs an
// all-reduce operation to determine if minibatching is required on any device.
// Otherwise, it returns the locally computed `local_minibatching_required`.
absl::StatusOr<bool> SyncMinibatchingRequired(
    const PreprocessSparseDenseMatmulInputOptions& options,
    absl::Span<const TableState> table_states) {
  tsl::profiler::TraceMe traceme([&] {
    return tsl::profiler::TraceMeEncode(
        "SyncMinibatchingRequired", {{"batch_number", options.batch_number}});
  });
  if (!options.enable_minibatching) {
    return false;
  }
  bool local_minibatching_required = false;
  for (const auto& state : table_states) {
    for (const auto& sorting_result_av : state.device_sorting_results) {
      tsl::BlockUntilReady(sorting_result_av);
      local_minibatching_required |=
          sorting_result_av.get().table_minibatching_required;
    }
  }
  if (options.all_reduce_interface != nullptr) {
    TF_ASSIGN_OR_RETURN(auto reduced_value,
                        options.all_reduce_interface->BlockingAllReduce(
                            options.batch_number * 2,
                            internal::Serialize(local_minibatching_required)));
    return internal::Deserialize(reduced_value);
  } else {
    return local_minibatching_required;
  }
}

// Synchronizes the `MinibatchingSplit` across all participating devices.
// If `options.all_reduce_interface` is provided, it performs an all-reduce
// operation to get the combined `MinibatchingSplit` from all local devices.
// Otherwise, it returns the locally computed `MinibatchingSplit`.
absl::StatusOr<MinibatchingSplit> SyncMinibatchingSplit(
    const PreprocessSparseDenseMatmulInputOptions& options,
    absl::Span<const TableState> table_states) {
  tsl::profiler::TraceMe traceme([&] {
    return tsl::profiler::TraceMeEncode(
        "SyncMinibatchingSplit", {{"batch_number", options.batch_number}});
  });
  MinibatchingSplit local_minibatching_split = 0;
  for (const auto& state : table_states) {
    for (const auto& sorting_result_av : state.device_sorting_results) {
      local_minibatching_split |=
          sorting_result_av.get().table_minibatching_split;
    }
  }
  if (options.all_reduce_interface != nullptr) {
    TF_ASSIGN_OR_RETURN(auto reduced_value,
                        options.all_reduce_interface->BlockingAllReduce(
                            options.batch_number * 2 + 1,
                            internal::Serialize(local_minibatching_split)));
    return internal::Deserialize(reduced_value);
  } else {
    return local_minibatching_split;
  }
}

// Populates the output structure `out` with the processed data from the
// `TableState`. This includes moving the CSR arrays and statistics for the
// current stacked table. The `output_mutex` is used to protect access to `out`.
void PopulateOutputStats(TableState& state,
                         SparseDenseMatmulInputStats& stats,
                         absl::Mutex& output_mutex) {
  state.stats_per_host.Flatten();

  absl::MutexLock lock(output_mutex);

  stats.max_ids_per_partition[state.stacked_table_name] =
      std::move(state.stats_per_host.max_ids_per_partition);
  stats.max_unique_ids_per_partition[state.stacked_table_name] =
      std::move(state.stats_per_host.max_unique_ids_per_partition);
  stats.required_buffer_sizes[state.stacked_table_name] =
      std::move(state.stats_per_host.required_buffer_size);
  stats.dropped_id_count[state.stacked_table_name] =
      state.stats_per_host.dropped_id_count;
}

// Fills the device-specific CSR buffers for a single stacked table.
// This involves merging minibatching buckets if required and then populating
// the `csr_arrays_per_host` within the provided `TableState`. The statistics
// are also updated, and the results are moved to the output structure `out`.
// `state`: The TableState holding the COO tensors and statistics.
// `options`: Preprocessing options.
// `global_minibatching_required`: Whether minibatching is required across all
//   devices.
// `global_minibatching_split`: The determined split for minibatching.
// `row_pointers_size_per_bucket`: The size of row pointers per bucket.
void FillDeviceBuffersForTable(
    TableState& state, const PreprocessSparseDenseMatmulInputOptions& options,
    int row_pointers_size_per_bucket, bool global_minibatching_required,
    MinibatchingSplit global_minibatching_split,
    absl::BlockingCounter& counter) {
  tsl::profiler::TraceMe traceme([&] {
    return tsl::profiler::TraceMeEncode(
        absl::StrCat("ScheduleFillBuffer/", state.stacked_table_name),
        {{"batch_number", options.batch_number}});
  });
  for (int local_device = 0; local_device < options.local_device_count;
       ++local_device) {
    tsl::AsyncValueRef<DeviceSortingTaskResult>& sorting_result_av =
        state.device_sorting_results[local_device];
    // This continuation is scheduled when sorting for `local_device` completes.
    sorting_result_av.AndThen([&, local_device, sorting_result_av,
                               &state = state, row_pointers_size_per_bucket,
                               global_minibatching_required,
                               global_minibatching_split] {
      state.batch_size_for_device =
          state.extracted_coo_tensors_per_device[0].batch_size_for_device;
      for (const auto& extracted_coo : state.extracted_coo_tensors_per_device) {
        DCHECK_EQ(state.batch_size_for_device,
                  extracted_coo.batch_size_for_device);
      }

      DevicePartitionedCooTensors& grouped_coo_tensors =
          sorting_result_av.get().grouped_coo_tensors;
      // If minibatching is required by any host, merge buckets
      // according to the globally synchronized split.
      if (options.enable_minibatching && global_minibatching_required) {
        grouped_coo_tensors.Merge(global_minibatching_split);
      }

      const int batch_size_per_sc = xla::CeilOfRatio(
          state.batch_size_for_device, options.num_sc_per_device);
      const int coo_buffer_size_per_sc =
          state.coo_buffer_size_per_device / options.num_sc_per_device;
      internal::CsrArraysRefPerDevice csr_arrays_per_device =
          state.csr_arrays_per_host.GetCsrArraysRefForDevice(local_device);

      const BlockRow<int>& required_sc_buffer_sizes =
          state.stats_per_host.GetStatsPerDevice(local_device)
              .required_buffer_size;

      tsl::AsyncValueRef<int> dropped_id_count_av = FillLocalDeviceBufferAsync(
          grouped_coo_tensors, row_pointers_size_per_bucket,
          coo_buffer_size_per_sc, batch_size_per_sc, required_sc_buffer_sizes,
          options, state.stacked_table_name, csr_arrays_per_device);

      dropped_id_count_av.AndThen(
          [sorting_result_av, &counter, dropped_id_count_av]() {
            sorting_result_av.get().total_dropped_id_count +=
                dropped_id_count_av.get();
            counter.DecrementCount();
          });
    });
  }
}

std::tuple<Eigen::Ref<MatrixXi>, Eigen::Ref<MatrixXi>, Eigen::Ref<MatrixXi>,
           Eigen::Ref<MatrixXf>>
GetOutputCsrBuffers(const std::string& stacked_table_name,
                    const PreprocessSparseDenseMatmulInputOptions& options,
                    int row_pointers_size_per_device,
                    int coo_buffer_size_per_device,
                    OutputCsrArrays* output_csr_arrays,
                    PreprocessSparseDenseMatmulOutput& out) {
  if (output_csr_arrays != nullptr) {
    DCHECK(output_csr_arrays->lhs_row_pointers.contains(stacked_table_name))
        << "Missing lhs_row_pointers for table: " << stacked_table_name;
    DCHECK(output_csr_arrays->lhs_embedding_ids.contains(stacked_table_name))
        << "Missing lhs_embedding_ids for table: " << stacked_table_name;
    DCHECK(output_csr_arrays->lhs_sample_ids.contains(stacked_table_name))
        << "Missing lhs_sample_ids for table: " << stacked_table_name;
    DCHECK(output_csr_arrays->lhs_gains.contains(stacked_table_name))
        << "Missing lhs_gains for table: " << stacked_table_name;
    Eigen::Map<MatrixXi>& row_pointers =
        output_csr_arrays->lhs_row_pointers.find(stacked_table_name)->second;
    DCHECK_EQ(row_pointers.rows(), options.local_device_count);
    DCHECK_EQ(row_pointers.cols(), row_pointers_size_per_device);

    Eigen::Map<MatrixXi>& embedding_ids =
        output_csr_arrays->lhs_embedding_ids.find(stacked_table_name)->second;
    DCHECK_EQ(embedding_ids.rows(), options.local_device_count);
    DCHECK_EQ(embedding_ids.cols(), coo_buffer_size_per_device);

    Eigen::Map<MatrixXi>& sample_ids =
        output_csr_arrays->lhs_sample_ids.find(stacked_table_name)->second;
    DCHECK_EQ(sample_ids.rows(), options.local_device_count);
    DCHECK_EQ(sample_ids.cols(), coo_buffer_size_per_device);

    Eigen::Map<MatrixXf>& gains =
        output_csr_arrays->lhs_gains.find(stacked_table_name)->second;
    DCHECK_EQ(gains.rows(), options.local_device_count);
    DCHECK_EQ(gains.cols(), coo_buffer_size_per_device);

    return {row_pointers, embedding_ids, sample_ids, gains};
  }
  MatrixXi& row_pointers = out.lhs_row_pointers[stacked_table_name];
  MatrixXi& embedding_ids = out.lhs_embedding_ids[stacked_table_name];
  MatrixXi& sample_ids = out.lhs_sample_ids[stacked_table_name];
  MatrixXf& gains = out.lhs_gains[stacked_table_name];

  AllocateOutputCsrBuffersIfNeeded(row_pointers, embedding_ids, sample_ids,
                                   gains, options, row_pointers_size_per_device,
                                   coo_buffer_size_per_device);
  return {row_pointers, embedding_ids, sample_ids, gains};
}

void FillDeviceBuffersAllTables(
    absl::Span<TableState> table_states,
    const PreprocessSparseDenseMatmulInputOptions& options,
    int row_pointers_size_per_bucket, bool global_minibatching_required,
    MinibatchingSplit global_minibatching_split) {
  tsl::profiler::TraceMe traceme([&] {
    return tsl::profiler::TraceMeEncode(
        "ScheduleFillDeviceBuffers", {{"batch_number", options.batch_number}});
  });
  absl::BlockingCounter counter(table_states.size() *
                                options.local_device_count);
  for (auto& state : table_states) {
    FillDeviceBuffersForTable(state, options, row_pointers_size_per_bucket,
                              global_minibatching_required,
                              global_minibatching_split, counter);
  }
  traceme.Stop();
  tsl::profiler::TraceMe traceme_wait([&] {
    return tsl::profiler::TraceMeEncode(
        "WaitForFillDeviceBuffers", {{"batch_number", options.batch_number}});
  });
  counter.Wait();
}

}  // namespace

void SparseDenseMatmulInputStats::merge(
    const SparseDenseMatmulInputStats& other) {
  MergeStats(max_ids_per_partition, other.max_ids_per_partition);
  MergeStats(max_unique_ids_per_partition, other.max_unique_ids_per_partition);
  MergeStats(required_buffer_sizes, other.required_buffer_sizes);
  for (const auto& [table, count] : other.dropped_id_count) {
    dropped_id_count[table] += count;
  }
}

absl::StatusOr<PreprocessSparseDenseMatmulOutput>
PreprocessSparseDenseMatmulInput(
    absl::Span<std::unique_ptr<AbstractInputBatch>> input_batches,
    const absl::flat_hash_map<std::string, std::vector<FeatureMetadataInStack>>&
        stacked_tables,
    const PreprocessSparseDenseMatmulInputOptions& options,
    OutputCsrArrays* output_csr_arrays) {
  tsl::profiler::TraceMe traceme([&] {
    return tsl::profiler::TraceMeEncode(
        absl::StrCat("jax_sc_embedding::PreprocessSparseDenseMatmulInput"),
        {{"batch_number", options.batch_number}});
  });
  if (options.sharding_strategy != ShardingStrategy::kMod) {
    LOG(FATAL) << "Only mod sharding is supported for now.";
  }
  CHECK_GT(options.local_device_count, 0);
  CHECK_GT(input_batches.size(), 0) << "input_batches cannot be empty.";

  absl::Mutex output_mutex;
  PreprocessSparseDenseMatmulOutput out ABSL_GUARDED_BY(output_mutex);

  const int num_scs = options.GetNumScs();
  const int row_pointers_size_per_bucket =
      std::max(num_scs, TPU_VECTOR_REGISTER_ALIGNMENT_SIZE);
  const int num_buckets =
      options.enable_minibatching ? CooFormat::kMaxMinibatchingBuckets : 1;
  const int row_pointers_size_per_device =
      row_pointers_size_per_bucket * num_buckets * options.num_sc_per_device;

  std::vector<TableState> table_states;
  table_states.reserve(stacked_tables.size());
  for (const auto& [stacked_table_name, stacked_table_metadata] :
       stacked_tables) {
    const bool stack_has_weights =
        StackHasVariableWeights(input_batches, stacked_table_metadata);
    const int coo_buffer_size_per_device = ComputeCooBufferSizePerDevice(
        num_scs, options.num_sc_per_device, stacked_table_metadata,
        options.batch_number, options.enable_minibatching);

    auto [row_pointers, embedding_ids, sample_ids, gains] = GetOutputCsrBuffers(
        stacked_table_name, options, row_pointers_size_per_device,
        coo_buffer_size_per_device, output_csr_arrays, out);

    table_states.emplace_back(stacked_table_name, stacked_table_metadata,
                              stack_has_weights, options, num_scs,
                              coo_buffer_size_per_device, row_pointers,
                              embedding_ids, sample_ids, gains);
  }

  // Stage 1: COO Extraction and Initial Sort/Group
  {
    tsl::profiler::TraceMe traceme([&] {
      return tsl::profiler::TraceMeEncode(
          "ScheduleExtractSortGroup",
          {{"batch_number", options.batch_number}});
    });
    std::vector<tsl::AsyncValueRef<DeviceSortingTaskResult>> sorting_avs;
    for (auto& state : table_states) {
      ExtractSortAndGroupCooTensorsForTable(state, input_batches, options);
      for (const auto& av : state.device_sorting_results) {
        sorting_avs.push_back(av);
      }
    }
  }

  tsl::AsyncValueRef<absl::StatusOr<bool>> global_minibatching_required_avr;
  std::unique_ptr<tsl::Thread> sync_thread;
  if (options.enable_minibatching) {
    global_minibatching_required_avr =
        tsl::MakeUnconstructedAsyncValueRef<absl::StatusOr<bool>>();
    sync_thread.reset(tsl::Env::Default()->StartThread(
        tsl::ThreadOptions(), "SyncMinibatchingRequired",
        [&options, &table_states, avr = global_minibatching_required_avr]() {
          avr.emplace(SyncMinibatchingRequired(options, table_states));
        }));
  }

  // If minibatching is not enabled, or if it is enabled, we fill
  // device buffers assuming minibatching is not required globally.
  // If it turns out that minibatching is required globally, we will
  // re-fill the buffers later.
  FillDeviceBuffersAllTables(absl::MakeSpan(table_states), options,
                             row_pointers_size_per_bucket,
                             /* global_minibatching_required= */ false,
                             /* global_minibatching_split= */ 0);

  for (auto& state : table_states) {
    tsl::RunWhenReady(absl::MakeConstSpan(state.device_sorting_results),
                      [&state]() { PostProcessTableState(state); });
  }

  bool global_minibatching_required = false;
  if (options.enable_minibatching) {
    tsl::profiler::TraceMe traceme([&] {
      return tsl::profiler::TraceMeEncode(
          "WaitForGlobalMinibatchingSync",
          {{"batch_number", options.batch_number}});
    });
    tsl::BlockUntilReady(global_minibatching_required_avr);
    TF_ASSIGN_OR_RETURN(global_minibatching_required,
                        *global_minibatching_required_avr);
  }

  MinibatchingSplit global_minibatching_split = 0;

  // Minibatching slow path: Optional Re-Sort/Group
  if (options.enable_minibatching && global_minibatching_required) {
    {
      tsl::profiler::TraceMe traceme([&] {
        return tsl::profiler::TraceMeEncode(
            "ScheduleCreateMinibatchingBuckets",
            {{"batch_number", options.batch_number}});
      });
      std::vector<tsl::AsyncValueRef<DeviceSortingTaskResult>> sorting_avs;
      for (auto& state : table_states) {
        CreateMinibatchingBucketsForTable(state, options);
        for (const auto& av : state.device_sorting_results) {
          sorting_avs.push_back(av);
        }
      }
      for (auto& state : table_states) {
        tsl::RunWhenReady(absl::MakeConstSpan(state.device_sorting_results),
                          [&state]() { PostProcessTableState(state); });
      }
    }

    TF_ASSIGN_OR_RETURN(global_minibatching_split,
                        SyncMinibatchingSplit(options, table_states));
    FillDeviceBuffersAllTables(
        absl::MakeSpan(table_states), options, row_pointers_size_per_bucket,
        global_minibatching_required, global_minibatching_split);
  }

  for (auto& state : table_states) {
    state.stats_per_host.dropped_id_count = 0;
    for (const auto& result_av : state.device_sorting_results) {
      state.stats_per_host.dropped_id_count +=
          result_av.get().total_dropped_id_count;
    }
    // NOMUTANTS -- Informational.
    CheckBufferUsage(
        /* max_required_buffer_size_per_device= */
        state.stats_per_host.required_buffer_size.maxCoeff() *
            options.num_sc_per_device,
        state.coo_buffer_size_per_device, state.stacked_table_name,
        options.batch_number);

    PopulateOutputStats(state, out.stats, output_mutex);
  }

  out.num_minibatches = global_minibatching_split.count() + 1;
  DCHECK(options.enable_minibatching || out.num_minibatches == 1)
      << "Minibatching is not enabled but num_minibatches is not 1.";

  return out;
}

}  // namespace jax_sc_embedding
