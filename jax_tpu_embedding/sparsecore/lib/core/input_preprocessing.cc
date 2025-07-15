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
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_threads.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
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
    const int batch_size_per_slice, std::vector<CooFormat>& coo_tensors) {
  const int feature_index = metadata.feature_index;
  const std::unique_ptr<AbstractInputBatch>& curr_batch =
      input_batches[feature_index];
  const int num_samples = curr_batch->size();

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
  curr_batch->ExtractCooTensors({.slice_start = start_index,
                                 .slice_end = end_index,
                                 .row_offset = row_offset,
                                 .col_offset = col_offset,
                                 .col_shift = col_shift,
                                 .num_scs = options.GetNumScs(),
                                 .combiner = metadata.row_combiner},
                                coo_tensors);
}
}  // namespace

namespace internal {

// Extract the COO tensors for all features.
ExtractedCooTensors ExtractCooTensorsForAllFeaturesPerLocalDevice(
    const absl::Span<const StackedTableMetadata> stacked_table_metadata,
    absl::Span<std::unique_ptr<AbstractInputBatch>> input_batches,
    const int local_device_id,
    const PreprocessSparseDenseMatmulInputOptions& options) {
  ExtractedCooTensors extracted_coo_tensors;

  extracted_coo_tensors.batch_size_for_device = 0;
  for (const auto& feature_metadata : stacked_table_metadata) {
    extracted_coo_tensors.batch_size_for_device +=
        input_batches[feature_metadata.feature_index]->size() /
        options.local_device_count;
  }

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

  const int batch_size_per_slice = CeilOfRatio(
      extracted_coo_tensors.batch_size_for_device, feature_slices_per_device);

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
          feature_slices_per_device, options, batch_size_per_slice,
          extracted_coo_tensors.coo_tensors);
    }
  }
  return extracted_coo_tensors;
}

}  // namespace internal

namespace {

// Preprocess inputs for a single table. Stacked table here refers to a
// a table that has no parent in the table stacking hierarchy. So in the case
// of table stacking, the stacked table is the top level table and in the case
// where we don't have any table stacking, the table itself is top level.
void PreprocessInputForStackedTablePerLocalDevice(
    const absl::Span<const StackedTableMetadata> stacked_table_metadata,
    absl::Span<std::unique_ptr<AbstractInputBatch>> input_batches,
    const int local_device_id,
    const PreprocessSparseDenseMatmulInputOptions& options,
    const int coo_buffer_size, const int row_pointers_size_per_sc,
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

  ExtractedCooTensors extracted_coo_tensors =
      internal::ExtractCooTensorsForAllFeaturesPerLocalDevice(
          stacked_table_metadata, input_batches, local_device_id, options);

  row_pointer_buffer.setConstant(coo_buffer_size);

  //
  // Step 2: Sort the COO tensors and group them by SC.
  //

  std::vector<std::vector<CooFormat>> coo_tensors_by_id =
      SortAndGroupCooTensorsPerLocalDevice(
          extracted_coo_tensors, stacked_table_metadata[0], options,
          max_ids_buffer, max_unique_ids_buffer,
          required_buffer_size_per_sc_buffer);

  //
  // Step 3: Compute the row pointers for each group of IDs.
  //
  const int batch_size_per_sc = CeilOfRatio(
      extracted_coo_tensors.batch_size_for_device, options.num_sc_per_device);
  const int coo_buffer_size_per_sc =
      coo_buffer_size / options.num_sc_per_device;
  FillRowPointersPerLocalDevice(
      coo_tensors_by_id, row_pointers_size_per_sc, coo_buffer_size_per_sc,
      batch_size_per_sc, num_scs, options.num_sc_per_device, row_pointer_buffer,
      embedding_id_buffer, sample_id_buffer, gain_buffer);
}

// Check the buffer usage ratio and log a warning if it exceeds a certain
// threshold.
// `max_required_buffer_size_per_device`: The maximum buffer size required by
// the input for a single device.
// `coo_buffer_size_per_device`: The allocated buffer size per device.
// `stacked_table_name`: The name of the stacked table.
void CheckBufferUsage(int max_required_buffer_size_per_device,
                      int coo_buffer_size_per_device,
                      absl::string_view stacked_table_name) {
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

    LOG(WARNING) << absl::StrFormat(
        "Required usage %.2f%% (%d bytes) of computed/given buffer size "
        "(%d bytes) for stacked table %s (Wasted space: %d bytes, "
        "Buffer shortfall: %d bytes)",
        usage_ratio * 100.0f, required_buffer_bytes, coo_buffer_bytes,
        stacked_table_name, wasted_space, buffer_shortfall);
  }
}

void MergeStats(
    absl::flat_hash_map<std::string, Eigen::RowVectorXi>& current_stats,
    const absl::flat_hash_map<std::string, Eigen::RowVectorXi>& other_stats) {
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

}  // namespace

void SparseDenseMatmulInputStats::merge(
    const SparseDenseMatmulInputStats& other) {
  MergeStats(max_ids_per_partition, other.max_ids_per_partition);
  MergeStats(max_unique_ids_per_partition, other.max_unique_ids_per_partition);
  MergeStats(required_buffer_sizes, other.required_buffer_sizes);
}

PreprocessSparseDenseMatmulOutput PreprocessSparseDenseMatmulInput(
    absl::Span<std::unique_ptr<AbstractInputBatch>> input_batches,
    const absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>&
        stacked_tables,
    const PreprocessSparseDenseMatmulInputOptions& options) {
  tsl::profiler::TraceMe t([=] {
    return absl::StrCat("input_preprocessing_cc-", options.local_device_count,
                        "/", options.global_device_count);
  });
  if (options.sharding_strategy != ShardingStrategy::kMod) {
    LOG(FATAL) << "Only mod sharding is supported for now.";
  }
  CHECK_GT(options.local_device_count, 0);

  absl::Mutex mutex;
  PreprocessSparseDenseMatmulOutput out;
  const int num_scs = options.GetNumScs();
  const int row_pointers_size_per_sc =
      std::max(num_scs, TPU_VECTOR_REGISTER_ALIGMENT_SIZE);

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
              stacked_table_metadata, input_batches, local_device, options,
              coo_buffer_size_per_device, row_pointers_size_per_sc,
              row_pointer_buffer, embedding_id_buffer, sample_id_buffer,
              gain_buffer, max_ids_per_partition_per_sc_buffer,
              max_unique_ids_per_partition_per_sc_buffer,
              required_buffer_size_per_sc_buffer);
        }
        CheckBufferUsage(
            /* max_required_buffer_size_per_device= */
            required_buffer_size_per_sc.maxCoeff() * options.num_sc_per_device,
            coo_buffer_size_per_device, stacked_table_name);

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

}  // namespace jax_sc_embedding
