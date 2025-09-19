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
#include <string>
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
#include "jax_tpu_embedding/sparsecore/lib/core/abstract_input_batch.h"
#include "jax_tpu_embedding/sparsecore/lib/core/coo_format.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_threads.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "jax_tpu_embedding/sparsecore/lib/core/partitioned_coo_tensors.h"
#include "jax_tpu_embedding/sparsecore/lib/core/sort_and_group_coo_tensors_impl.h"
#include "tsl/platform/statusor.h"  // from @tsl
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/traceme.h"

namespace jax_sc_embedding {

namespace {

// Extract the COO tensors for a single feature slice.
void ExtractCooTensorsForSingleFeatureSlice(
    const StackedTableMetadata& metadata,
    absl::Span<std::unique_ptr<AbstractInputBatch>> input_batches,
    const int local_device_id, const int feature_slice_id,
    const int feature_slices_per_device,
    const PreprocessSparseDenseMatmulInputOptions& options,
    ExtractedCooTensors& extracted_coo_tensors) {
  const int feature_index = metadata.feature_index;
  const std::unique_ptr<AbstractInputBatch>& curr_batch =
      input_batches[feature_index];
  const int num_samples = curr_batch->size();

  const int batch_size_per_slice = CeilOfRatio(
      extracted_coo_tensors.batch_size_for_device, feature_slices_per_device);

  CHECK_GT(feature_slices_per_device, 0);
  CHECK_GT(options.global_device_count, 0);
  CHECK_GT(options.local_device_count, 0);

  const int row_offset_per_slice =
      metadata.row_offset /
      (options.global_device_count * feature_slices_per_device);
  const int row_offset =
      feature_slice_id * batch_size_per_slice + row_offset_per_slice;
  const int col_offset = metadata.col_offset;
  const int col_shift = metadata.col_shift;

  const int num_samples_per_split =
      num_samples / (options.local_device_count * feature_slices_per_device);
  const int start_index =
      (local_device_id * feature_slices_per_device + feature_slice_id) *
      num_samples_per_split;
  int end_index = std::min(num_samples, start_index + num_samples_per_split);

  // In the case of feature stacking, we need to group all the COO tensors
  // at this stage (i.e., before the sorting later on).
  VLOG(2) << absl::StrFormat(
      "Extracting COO Tensor from feature #%d from row %d to %d "
      "(local_device_id = %d, feature_slice_id = %d, row_offset = %d, "
      "batch_size_per_slice = %d)",
      feature_index, start_index, end_index, local_device_id, feature_slice_id,
      row_offset, batch_size_per_slice);
  curr_batch->ExtractCooTensors(
      {
          .slice_start = start_index,
          .slice_end = end_index,
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

// Holds the state for processing a single stacked table across all local
// devices. This includes extracted COO tensors, partitioned COO tensors,
// CSR arrays, and statistics.
struct TableState {
  std::string_view stacked_table_name;
  absl::Span<const StackedTableMetadata> stacked_table_metadata;
  int coo_buffer_size_per_device;
  CsrArraysPerHost csr_arrays_per_host;
  StatsPerHost stats_per_host;
  int batch_size_for_device;
  bool table_minibatching_required = false;
  MinibatchingSplit table_minibatching_split = 0;
  std::vector<ExtractedCooTensors> extracted_coo_tensors_per_device;
  std::vector<PartitionedCooTensors> partitioned_coo_tensors_per_device;

  TableState(const std::string& name,
             absl::Span<const StackedTableMetadata> metadata,
             const PreprocessSparseDenseMatmulInputOptions& options,
             int num_scs, int row_pointers_size_per_bucket)
      : stacked_table_name(name),
        stacked_table_metadata(metadata),
        coo_buffer_size_per_device(
            ComputeCooBufferSizePerDevice(num_scs, options.num_sc_per_device,
                                          metadata, options.batch_number)),
        csr_arrays_per_host(options.local_device_count,
                            row_pointers_size_per_bucket *
                                (options.enable_minibatching
                                     ? CooFormat::kMaxMinibatchingBuckets
                                     : 1) *
                                options.num_sc_per_device,
                            coo_buffer_size_per_device),
        stats_per_host(options.local_device_count, options.GetNumScs(),
                       options.num_sc_per_device),
        batch_size_for_device(0) {
    extracted_coo_tensors_per_device.reserve(options.local_device_count);
    partitioned_coo_tensors_per_device.reserve(options.local_device_count);
  }
};

// Extracts, sorts, and groups COO tensors for a single stacked table across
// all local devices. This function populates
// `state.extracted_coo_tensors_per_device` and
// `state.partitioned_coo_tensors_per_device`.
void ExtractSortAndGroupCooTensorsForTable(
    TableState& state,
    absl::Span<std::unique_ptr<AbstractInputBatch>> input_batches,
    const PreprocessSparseDenseMatmulInputOptions& options,
    uint64_t context_id) {
  tsl::profiler::TraceMeConsumer consumer(
      [&] {
        return absl::StrCat("InputPreprocessingTable-ExtractSortGroup-",
                            state.stacked_table_name);
      },
      context_id);

  for (int local_device = 0; local_device < options.local_device_count;
       ++local_device) {
    ExtractedCooTensors extracted_coo_tensors =
        internal::ExtractCooTensorsForAllFeaturesPerLocalDevice(
            state.stacked_table_metadata, input_batches, local_device, options);
    state.extracted_coo_tensors_per_device.push_back(extracted_coo_tensors);
    if (local_device == 0)
      state.batch_size_for_device = extracted_coo_tensors.batch_size_for_device;
    else
      CHECK_EQ(state.batch_size_for_device,
               extracted_coo_tensors.batch_size_for_device);

    internal::StatsPerDevice stats_per_device =
        state.stats_per_host.GetStatsPerDevice(local_device);
    const PartitionedCooTensors grouped_coo_tensors =
        SortAndGroupCooTensorsPerLocalDevice(
            extracted_coo_tensors, state.stacked_table_metadata[0], options,
            stats_per_device, state.table_minibatching_required);
    state.partitioned_coo_tensors_per_device.push_back(grouped_coo_tensors);
    state.stats_per_host.dropped_id_count += stats_per_device.dropped_id_count;
  }
}

// Creates minibatching buckets for a single stacked table across all local
// devices. This function re-sorts and groups the extracted COO tensors
// based on the minibatching split determined in a previous stage.
// `state`: The TableState holding the COO tensors and statistics.
// `options`: Preprocessing options.
// `context_id`: Profiling context ID.
void CreateMinibatchingBucketsForTable(
    TableState& state, const PreprocessSparseDenseMatmulInputOptions& options,
    uint64_t context_id) {
  tsl::profiler::TraceMeConsumer consumer(
      [&] {
        return absl::StrCat(
            "InputPreprocessingTable-CreateMinibatchingBuckets-",
            state.stacked_table_name);
      },
      context_id);
  state.stats_per_host.dropped_id_count = 0;
  for (int local_device = 0; local_device < options.local_device_count;
       ++local_device) {
    internal::StatsPerDevice stats_per_device =
        state.stats_per_host.GetStatsPerDevice(local_device);
    state.partitioned_coo_tensors_per_device[local_device] =
        SortAndGroupCooTensorsPerLocalDevice(
            state.extracted_coo_tensors_per_device[local_device],
            state.stacked_table_metadata[0], options, stats_per_device,
            state.table_minibatching_split);
    state.stats_per_host.dropped_id_count += stats_per_device.dropped_id_count;
  }
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
    const absl::Span<const StackedTableMetadata> stacked_table_metadata,
    absl::Span<std::unique_ptr<AbstractInputBatch>> input_batches,
    const int local_device_id,
    const PreprocessSparseDenseMatmulInputOptions& options) {
  int batch_size_for_device = 0;
  for (const auto& feature_metadata : stacked_table_metadata) {
    batch_size_for_device +=
        input_batches[feature_metadata.feature_index]->size() /
        options.local_device_count;
  }
  CheckDeviceBatchSize(batch_size_for_device, options.num_sc_per_device,
                       stacked_table_metadata[0].name);

  int feature_slices_per_device;
  switch (options.feature_stacking_strategy) {
    case FeatureStackingStrategy::kStackThenSplit:
      feature_slices_per_device = 1;
      break;
    case FeatureStackingStrategy::kSplitThenStack:
      feature_slices_per_device = options.num_sc_per_device;
      break;
    default:
      LOG(FATAL) << "Unsupported feature stacking strategy: "
                 << static_cast<int>(options.feature_stacking_strategy);
      break;
  }

  CHECK_GE(batch_size_for_device,
           feature_slices_per_device * stacked_table_metadata.size())
      << "Batch size must be greater or equal to the number of "
         "features stacked together (per feature slice).";

  ExtractedCooTensors extracted_coo_tensors(options.num_sc_per_device,
                                            batch_size_for_device);

  // This slices each feature into `feature_slices` partitions and then
  // interleaves them: (k=num_sc_per_device-1). For stacking strategy
  //   SC0: F1_1, F2_1, ... Fn_1,  // <- batch_size_per_slice
  //   SC1: F1_2, F2_2, ... Fn_2,  // <- batch_size_per_slice
  //   ...                         // <- batch_size_per_slice
  //   SCk: F1_k, F2_k, ..., Fn_k  // <- batch_size_per_slice
  for (int feature_slice_id = 0; feature_slice_id < feature_slices_per_device;
       ++feature_slice_id) {
    for (const auto& feature_metadata : stacked_table_metadata) {
      ExtractCooTensorsForSingleFeatureSlice(
          feature_metadata, input_batches, local_device_id, feature_slice_id,
          feature_slices_per_device, options, extracted_coo_tensors);
    }
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

    LOG_IF(WARNING, batch_number % 100 == 0) << absl::StrFormat(
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
  if (!options.enable_minibatching) {
    return false;
  }
  bool local_minibatching_required = false;
  for (const auto& state : table_states) {
    local_minibatching_required |= state.table_minibatching_required;
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
  MinibatchingSplit local_minibatching_split = 0;
  for (const auto& state : table_states) {
    local_minibatching_split |= state.table_minibatching_split;
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
void PopulateOutput(TableState& state, PreprocessSparseDenseMatmulOutput& out,
                    absl::Mutex& output_mutex) {
  state.stats_per_host.Flatten();

  absl::MutexLock mutex_lock(output_mutex);
  out.lhs_row_pointers[state.stacked_table_name] =
      std::move(state.csr_arrays_per_host.row_pointers);
  out.lhs_embedding_ids[state.stacked_table_name] =
      std::move(state.csr_arrays_per_host.embedding_ids);
  out.lhs_sample_ids[state.stacked_table_name] =
      std::move(state.csr_arrays_per_host.sample_ids);
  out.lhs_gains[state.stacked_table_name] =
      std::move(state.csr_arrays_per_host.gains);

  out.stats.max_ids_per_partition[state.stacked_table_name] =
      std::move(state.stats_per_host.max_ids_per_partition);
  out.stats.max_unique_ids_per_partition[state.stacked_table_name] =
      std::move(state.stats_per_host.max_unique_ids_per_partition);
  out.stats.required_buffer_sizes[state.stacked_table_name] =
      std::move(state.stats_per_host.required_buffer_size);
  out.stats.dropped_id_count[state.stacked_table_name] =
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
// `context_id`: Profiling context ID.
// `out`: The output structure to be populated with CSR arrays and stats.
// `output_mutex`: Mutex to protect access to `out`.
void FillDeviceBuffersForTable(
    TableState& state, const PreprocessSparseDenseMatmulInputOptions& options,
    bool global_minibatching_required,
    MinibatchingSplit global_minibatching_split,
    int row_pointers_size_per_bucket, uint64_t context_id,
    PreprocessSparseDenseMatmulOutput& out, absl::Mutex& output_mutex) {
  tsl::profiler::TraceMeConsumer consumer(
      [&] {
        return absl::StrCat("InputPreprocessingTable-FillBuffer-",
                            state.stacked_table_name);
      },
      context_id);
  int table_dropped_ids = 0;
  for (int local_device = 0; local_device < options.local_device_count;
       ++local_device) {
    PartitionedCooTensors& grouped_coo_tensors =
        state.partitioned_coo_tensors_per_device[local_device];
    if (options.enable_minibatching && global_minibatching_required) {
      grouped_coo_tensors.Merge(global_minibatching_split);
    }

    const int batch_size_per_sc =
        CeilOfRatio(state.batch_size_for_device, options.num_sc_per_device);
    const int coo_buffer_size_per_sc =
        state.coo_buffer_size_per_device / options.num_sc_per_device;
    internal::CsrArraysPerDevice csr_arrays_per_device =
        state.csr_arrays_per_host.GetCsrArraysPerDevice(local_device);
    FillLocalDeviceBuffer(grouped_coo_tensors, row_pointers_size_per_bucket,
                          coo_buffer_size_per_sc, batch_size_per_sc, options,
                          csr_arrays_per_device, table_dropped_ids);
    state.stats_per_host.dropped_id_count += table_dropped_ids;
  }
  // NOMUTANTS -- Informational.
  CheckBufferUsage(
      /* max_required_buffer_size_per_device= */
      state.stats_per_host.required_buffer_size.maxCoeff() *
          options.num_sc_per_device,
      state.coo_buffer_size_per_device, state.stacked_table_name,
      options.batch_number);

  PopulateOutput(state, out, output_mutex);
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
    const absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>&
        stacked_tables,
    const PreprocessSparseDenseMatmulInputOptions& options) {
  tsl::profiler::TraceMe t([=, &options] {
    return absl::StrCat("input_preprocessing_cc-", options.local_device_count,
                        "/", options.global_device_count);
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
      std::max(num_scs, TPU_VECTOR_REGISTER_ALIGMENT_SIZE);

  tsl::profiler::TraceMeProducer producer("InputPreprocessingMainThread");

  std::vector<TableState> table_states;
  for (const auto& [stacked_table_name, stacked_table_metadata] :
       stacked_tables) {
    table_states.emplace_back(stacked_table_name, stacked_table_metadata,
                              options, num_scs, row_pointers_size_per_bucket);
  }

  // Stage 1: COO Extraction and Initial Sort/Group
  {
    absl::BlockingCounter counter(stacked_tables.size());
    for (auto& state : table_states) {
      PreprocessingThreadPool()->Schedule(
          [&, &state = state, context_id = producer.GetContextId()] {
            ExtractSortAndGroupCooTensorsForTable(state, input_batches, options,
                                                  context_id);
            counter.DecrementCount();
          });
    }
    counter.Wait();
  }
  TF_ASSIGN_OR_RETURN(bool global_minibatching_required,
                      SyncMinibatchingRequired(options, table_states));

  // Stage 2: Optional Re-Sort/Group
  MinibatchingSplit global_minibatching_split = 0;

  if (options.enable_minibatching && global_minibatching_required) {
    absl::BlockingCounter counter(stacked_tables.size());
    for (auto& state : table_states) {
      PreprocessingThreadPool()->Schedule(
          [&, &state = state, context_id = producer.GetContextId()] {
            CreateMinibatchingBucketsForTable(state, options, context_id);
            counter.DecrementCount();
          });
    }
    counter.Wait();

    TF_ASSIGN_OR_RETURN(global_minibatching_split,
                        SyncMinibatchingSplit(options, table_states));
  }

  // Stage 3: Fill Device Buffers
  {
    absl::BlockingCounter counter(stacked_tables.size());
    for (auto& state : table_states) {
      PreprocessingThreadPool()->Schedule(
          [&, &state = state, global_minibatching_required,
           global_minibatching_split, context_id = producer.GetContextId()] {
            FillDeviceBuffersForTable(
                state, options, global_minibatching_required,
                global_minibatching_split, row_pointers_size_per_bucket,
                context_id, out, output_mutex);
            counter.DecrementCount();
          });
    }
    counter.Wait();
  }

  out.num_minibatches =
      options.enable_minibatching && global_minibatching_required
          ? global_minibatching_split.count() + 1
          : 1;
  DCHECK(options.enable_minibatching || out.num_minibatches == 1)
      << "Minibatching is not enabled but num_minibatches is not 1.";

  return out;
}

}  // namespace jax_sc_embedding
