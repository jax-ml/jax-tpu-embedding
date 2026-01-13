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
#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_SORT_AND_GROUP_COO_TENSORS_IMPL_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_SORT_AND_GROUP_COO_TENSORS_IMPL_H_

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <utility>
#include <vector>

#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/numeric/bits.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "Eigen/Core"  // from @eigen_archive
#include "hwy/contrib/sort/order.h"  // from @highway
#include "hwy/contrib/sort/vqsort.h"  // from @highway
#include "jax_tpu_embedding/sparsecore/lib/core/coo_format.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "jax_tpu_embedding/sparsecore/lib/core/minibatching_splits_impl.h"
#include "jax_tpu_embedding/sparsecore/lib/core/partitioned_coo_tensors.h"
#include "xla/tsl/concurrency/async_value_ref.h"  // from @xla
#include "xla/util.h"  // from @xla
#include "tsl/profiler/lib/traceme.h"

namespace jax_sc_embedding {
namespace internal {

inline void ValidateMaxIdsOrDie(
    const int32_t observed_max_ids_per_partition,
    const int32_t observed_max_unique_ids_per_partition,
    const int32_t max_ids_per_partition,
    const int32_t max_unique_ids_per_partition,
    const absl::string_view stacked_table_name, const bool allow_id_dropping) {
  // If id dropping is allowed, we log a warning if the observed max ids per
  // partition is greater than the set max ids per partition.
  if (observed_max_ids_per_partition > max_ids_per_partition) {
    if (allow_id_dropping) {
      LOG(WARNING) << "Allowing ID dropping for table: " << stacked_table_name
                   << " observed max ids per partition: "
                   << observed_max_ids_per_partition
                   << " is greater than the set max ids per partition: "
                   << max_ids_per_partition;
    } else {
      LOG(FATAL) << "Observed max ids per partition: "
                 << observed_max_ids_per_partition
                 << " for table: " << stacked_table_name
                 << " is greater than the set max ids per partition: "
                 << max_ids_per_partition
                 << ". Use `allow_id_dropping` or increase the "
                    "`max_ids_per_partition`. You could also try to reduce the "
                    "batch size.";
    }
  }
  if (observed_max_unique_ids_per_partition > max_unique_ids_per_partition) {
    if (allow_id_dropping) {
      LOG(WARNING) << "Allowing ID dropping for table: " << stacked_table_name
                   << " observed max unique ids per partition: "
                   << observed_max_unique_ids_per_partition
                   << " is greater than the set max unique ids per partition: "
                   << max_unique_ids_per_partition;
    } else {
      LOG(FATAL) << "Observed max unique ids per partition: "
                 << observed_max_unique_ids_per_partition
                 << " for table: " << stacked_table_name
                 << " is greater than the set max unique ids per partition: "
                 << max_unique_ids_per_partition
                 << ". Use `allow_id_dropping` or increase the "
                    "`max_ids_per_partition`. You could also try to reduce the "
                    "batch size.";
    }
  }
}

inline void ValidateKeyCapacity(const int local_sc_id, const int key_count) {
  // Index = 0 to kDataMask giving us a count of kDataMask + 1.
  if (key_count > 1 + CooFormat::kDataMask) {
    LOG(ERROR) << absl::StrFormat(
        "Too many tensors for SparseCore #%d: got %d, limit: "
        "%d. Preprocessed output may not be reliable and cause undefined "
        "behavior.",
        local_sc_id, key_count, CooFormat::kDataMask);
  }
}

// Updates the global maximum number of ids per partition by taking the
// element-wise maximum between the current global maximum and the provided
// local maximum's sum across bucket.
inline void UpdateMaxIdsPerPartition(BlockRow<int>& global_max,
                                     const MatrixXi& local_max_per_bucket) {
  // global_max: 1 x global_sc_count
  // local_max: global_sc_count x bucket_count
  DCHECK_EQ(global_max.rows(), 1);
  DCHECK_EQ(global_max.cols(), local_max_per_bucket.rows());
  global_max =
      global_max.cwiseMax(local_max_per_bucket.rowwise().sum().transpose());
}

template <typename SplitType>
inline void UpdateMinibatchingSplit(
    MatrixXi& ids_per_sc_partition_per_bucket,
    MatrixXi& unique_ids_per_partition_per_bucket,
    const int32_t global_sc_count, const int32_t max_ids_per_partition,
    const int32_t max_unique_ids_per_partition, SplitType& minibatching_split) {
  // This works both when minibatching is required and not. In the former
  // case we have bool which tells us if minibatching is required,
  // in the latter case it is std::bitset<64> which tells us the exact
  // splits.
  for (int global_sc_id = 0; global_sc_id < global_sc_count; ++global_sc_id) {
    auto ids_per_bucket =
        ids_per_sc_partition_per_bucket.row(global_sc_id).array();
    auto unique_ids_per_bucket =
        unique_ids_per_partition_per_bucket.row(global_sc_id).array();
    if constexpr (std::is_same_v<SplitType, MinibatchingSplit>) {
      // The arrays must be mutable as ComputeMinibatchingSplit modifies them.
      // absl::Makespan works because the array would be row-major and
      // values would be contiguous in memory.
      static_assert(decltype(ids_per_bucket)::IsRowMajor);
      static_assert(decltype(unique_ids_per_bucket)::IsRowMajor);
      // NOTE: ComputeSplit modifies the span, but we have already updated
      // the output stats.
      minibatching_split |= ComputeMinibatchingSplit(
          absl::MakeSpan(ids_per_bucket.data(), ids_per_bucket.size()),
          max_ids_per_partition);
      minibatching_split |=
          ComputeMinibatchingSplit(absl::MakeSpan(unique_ids_per_bucket.data(),
                                                  unique_ids_per_bucket.size()),
                                   max_unique_ids_per_partition);
    } else {
      minibatching_split |=
          unique_ids_per_bucket.maxCoeff() > max_unique_ids_per_partition ||
          ids_per_bucket.maxCoeff() > max_ids_per_partition;
    }
  }
}

inline void LogSparseCoreStats(
    const int32_t local_sc_id, const absl::string_view stacked_table_name,
    const MatrixXi& ids_per_sc_partition_per_bucket,
    const MatrixXi& unique_ids_per_partition_per_bucket, const size_t keys_size,
    const PartitionedCooTensors& grouped_coo_tensors) {
  if (VLOG_IS_ON(2)) {
    LOG(INFO) << "For table " << stacked_table_name << " on local SparseCore "
              << local_sc_id
              << ": Observed ids per global SparseCore partition: "
              << ids_per_sc_partition_per_bucket.rowwise().sum();

    LOG(INFO) << "For table " << stacked_table_name << " on local SparseCore "
              << local_sc_id
              << ": Observed unique ids per global SparseCore partition: "
              << unique_ids_per_partition_per_bucket.rowwise().sum();

    LOG(INFO) << "For table " << stacked_table_name << " on local SparseCore "
              << local_sc_id << ": Total number of ids processed: " << keys_size
              << ", total after deduplication: "
              << ids_per_sc_partition_per_bucket.sum()
              << ", total after drop id: " << grouped_coo_tensors.Size();
  }
}

struct LocalSparseCoreTensorGroupingContext {
  absl::Span<const uint64_t> keys;
  const ExtractedCooTensorsPerSparseCore& extracted_coo_tensors;
  const StackedTableMetadata& stacked_table_metadata;
  const PreprocessSparseDenseMatmulInputOptions& options;
  const int32_t local_sc_id;
  const int32_t num_sc_bits;

  // Outputs.
  PartitionedCooTensors& grouped_coo_tensors;
  MatrixXi& ids_per_sc_partition_per_bucket;
  MatrixXi& unique_ids_per_partition_per_bucket;
  StatsPerDevice& stats;
  // These are only used for id dropping decisions and can be ignored otherwise.
  MatrixXi& kept_ids_per_sc_partition_per_bucket;
  MatrixXi& kept_unique_ids_per_partition_per_bucket;
};

template <bool kHasVariableWeights, bool kCreateBuckets>
inline void GroupAndDeduplicateCooTensorsForLocalSparseCore(
    LocalSparseCoreTensorGroupingContext context) {
  tsl::profiler::TraceMe group_traceme([&] {
    return absl::StrCat("GroupAndDeduplicateCooTensorsForLocalSparseCore/",
                        context.local_sc_id);
  });
  // Unpack context for readability.
  const PreprocessSparseDenseMatmulInputOptions& options = context.options;
  const StackedTableMetadata& stacked_table_metadata =
      context.stacked_table_metadata;
  PartitionedCooTensors& grouped_coo_tensors = context.grouped_coo_tensors;
  StatsPerDevice& stats = context.stats;
  MatrixXi& observed_ids = context.ids_per_sc_partition_per_bucket;
  MatrixXi& observed_unique_ids = context.unique_ids_per_partition_per_bucket;
  MatrixXi& kept_ids = context.kept_ids_per_sc_partition_per_bucket;
  MatrixXi& kept_unique_ids = context.kept_unique_ids_per_partition_per_bucket;

  const bool allow_id_dropping = options.allow_id_dropping;
  const uint32_t global_sc_count = options.GetNumScs();
  const int max_ids_per_partition =
      stacked_table_metadata.max_ids_per_partition;
  const int max_unique_ids_per_partition =
      stacked_table_metadata.max_unique_ids_per_partition;

  // We do NOT drop IDs when minibatching is enabled and we are in the
  // first pass (`kCreateBuckets=false`), as we need to detect limit
  // overflows to decide if minibatching is required.
  const bool can_drop_id = !options.enable_minibatching || kCreateBuckets;
  const bool perform_id_dropping = allow_id_dropping && can_drop_id;

  uint32_t prev_col_id = std::numeric_limits<uint32_t>::max();
  uint32_t prev_row_id = std::numeric_limits<uint32_t>::max();
  uint32_t prev_bucket_id = 0;
  // Tracks whether the current unique `col_id` should be dropped for exceeding
  // capacity. This decision is sticky for all tensors with the same `col_id`
  // within the same bucket.
  bool dropping_current_unique_col_id = false;
  const int num_sc_bits = context.num_sc_bits;
  for (const uint64_t key : context.keys) {
    // Step 1: Unpack key to get tensor coordinates.
    const uint32_t bucket_id =
        kCreateBuckets ? CooFormat::GetBucketIdFromKey(key) : 0;
    const uint32_t col_id =
        absl::rotl(CooFormat::GetRotatedColIdFromKey(key), num_sc_bits);
    const uint32_t global_sc_id = col_id & (global_sc_count - 1);

    const CooFormat coo_tensor =
        context.extracted_coo_tensors.GetCooFormatWithGain<kHasVariableWeights>(
            key, col_id);
    const uint32_t row_id = coo_tensor.row_id;

    // Step 2: Handle duplicates.
    // An ID that is a duplicate of a previously non-dropped ID is merged.
    // It does not count as a new ID for stats and does not go through dropping
    // logic.
    if (grouped_coo_tensors.MaybeMerge(coo_tensor)) {
      continue;
    }
    // If the ID is a duplicate of the last seen ID, it must have been dropped
    // (otherwise it would have been merged above), so drop this one too.
    if (perform_id_dropping && row_id == prev_row_id && col_id == prev_col_id) {
      ++stats.dropped_id_count;
      continue;
    }

    // Step 3: Update observed statistics for the new ID.
    // We have a new column if the bucket_id changes (we can't dedupe across
    // bucket boundaries) or if the col_id changes within the same bucket. Note
    // that multiple col_ids can map to the same bucket.
    bool is_new_col = col_id != prev_col_id;
    if constexpr (kCreateBuckets) {
      is_new_col = is_new_col || bucket_id != prev_bucket_id;
    }
    // Update observed stats. These are never decremented and are used for
    // reporting.
    observed_ids(global_sc_id, bucket_id) += 1;
    if (is_new_col) {
      observed_unique_ids(global_sc_id, bucket_id) += 1;
    }

    // Step 4: Add ID to result or drop it.
    if (!perform_id_dropping) {
      grouped_coo_tensors.Add(bucket_id, coo_tensor);
    } else {
      // Check limits.
      const bool exceeds_ids_limit =
          (kept_ids(global_sc_id, bucket_id) + 1) > max_ids_per_partition;
      if (is_new_col) {
        dropping_current_unique_col_id =
            (kept_unique_ids(global_sc_id, bucket_id) + 1) >
            max_unique_ids_per_partition;
      }

      // Drop/Keep ID.
      if (exceeds_ids_limit || dropping_current_unique_col_id) {
        // Dropped id.
        ++stats.dropped_id_count;
      } else {
        grouped_coo_tensors.Add(bucket_id, coo_tensor);
      }

      // Update kept counts.
      kept_ids(global_sc_id, bucket_id) += 1;
      if (is_new_col) {
        kept_unique_ids(global_sc_id, bucket_id) += 1;
      }
    }

    // Step 5: Update state for next iteration.
    // This must be done regardless of whether the ID was dropped to ensure
    // correct stats collection for subsequent IDs.
    prev_col_id = col_id;
    prev_row_id = row_id;
    prev_bucket_id = bucket_id;
  }
}

}  // namespace internal

// Sorts and groups the provided COO tensors in this hierarchy: Local SC ->
// Minibatching Bucket -> Global SC.
//
//
// NOTE: We use output buffers `max_ids_per_sc`, `max_unique_ids_per_sc`, and
// `required_buffer_size_per_sc` because we fill values in a loop to a bigger
// array.
//
// TODO(b/469153631): Break this function down into smaller ones. Aggregation
// could be a separate function.
template <bool kHasVariableWeights = false, bool kCreateBuckets,
          typename SplitType>
tsl::AsyncValueRef<DeviceSortingTaskResult>
SortAndGroupCooTensorsPerLocalDeviceImpl(
    const ExtractedCooTensors& extracted_coo_tensors,
    const StackedTableMetadata& stacked_table_metadata,
    const PreprocessSparseDenseMatmulInputOptions& options,
    internal::StatsPerDevice& stats, SplitType& minibatching_split) {
  tsl::profiler::TraceMe t("SortAndGroupCooTensors");
  const int num_sc_per_device = options.num_sc_per_device;
  const uint32_t global_sc_count = options.GetNumScs();
  const int num_sc_bits = absl::bit_width(global_sc_count - 1);
  const int max_ids_per_partition =
      stacked_table_metadata.max_ids_per_partition;
  const int max_unique_ids_per_partition =
      stacked_table_metadata.max_unique_ids_per_partition;
  const absl::string_view stacked_table_name = stacked_table_metadata.name;
  // This function can be called in two passes for minibatching. The logic for
  // stats collection and ID dropping depends on the pass.
  //
  // Pass 1: Check if minibatching is required (`kCreateBuckets` is false).
  // - No IDs are dropped.
  // - Stats are collected on all observed IDs to compute splits.
  //
  // Pass 2: Create buckets (`kCreateBuckets` is true).
  // - A dummy stats object is used (stats are not re-computed).
  // - IDs may be dropped if they exceed capacity.

  // Partition COO tensors among SparseCores for the local device (based on row
  // id).
  const int bucket_count =
      kCreateBuckets ? CooFormat::kMaxMinibatchingBuckets : 1;

  struct SortingTaskResult {
    PartitionedCooTensors grouped_coo_tensors;
    StatsPerHost stats_host;
    int dropped_id_count = 0;
    SplitType split_val = {};
  };

  std::vector<tsl::AsyncValueRef<SortingTaskResult>> task_results;
  task_results.reserve(num_sc_per_device);

  // Loop over sc's
  for (int local_sc_id = 0; local_sc_id < num_sc_per_device; ++local_sc_id) {
    auto result_av = tsl::MakeUnconstructedAsyncValueRef<SortingTaskResult>();
    task_results.push_back(result_av);

    tsl::AsyncValueRef<ExtractedCooTensorsPerSparseCore> sc_tensor_av =
        extracted_coo_tensors.per_sc_tensors_av[local_sc_id];
    tsl::RunWhenReady(
        {sc_tensor_av.GetAsyncValue()},
        [local_sc_id, result_av, sc_tensor_av, &stacked_table_metadata,
         &options, num_sc_bits, global_sc_count, bucket_count,
         max_ids_per_partition, max_unique_ids_per_partition,
         stacked_table_name]() mutable {
          options.async_task_scheduler([=]() mutable {
            tsl::profiler::TraceMe sort_traceme([&] {
              return absl::StrCat("SortAndGroupCooTensorsPerSparseCore/",
                                  local_sc_id);
            });

            // We need to aggregate stats and splits as well.
          int total_dropped = 0;
          SplitType task_split = {};

          StatsPerHost local_stats_host(1, global_sc_count,
                                        options.num_sc_per_device);

          const auto& extracted_coo_tensors = sc_tensor_av.get();
          // Prepare Local Result for ONE SC.
          PartitionedCooTensors grouped_coo_tensors(global_sc_count,
                                                    bucket_count);

          grouped_coo_tensors.Reserve(extracted_coo_tensors.size());

          internal::StatsPerDevice stats =
              local_stats_host.GetStatsPerDevice(0);

          std::vector<uint64_t> keys;
          keys.reserve(extracted_coo_tensors.size());
          internal::ValidateKeyCapacity(local_sc_id,
                                        extracted_coo_tensors.size());

          tsl::profiler::TraceMe generate_keys_traceme(
              [&] { return absl::StrCat("GenerateKeys/", local_sc_id); });
          for (uint32_t coo_index = 0; coo_index < extracted_coo_tensors.size();
               ++coo_index) {
            // The key here is [bucket_id(6 bits), global_sc_id(num_scs bits),
            // local_embedding_id(32-num_scs bits), index(26 bits)].
            //  Note that this assumes `num_scs` is a power of 2.
            uint32_t data = kHasVariableWeights
                                ? coo_index
                                : extracted_coo_tensors.ids[coo_index].row_id;
            keys.push_back(CooFormat::GetGroupingKey(
                extracted_coo_tensors.ids[coo_index].col_id, data, num_sc_bits,
                kCreateBuckets, options.minibatching_bucketing_hash_fn));
          }
          generate_keys_traceme.Stop();

          tsl::profiler::TraceMe vqsort_traceme(
              [&] { return absl::StrCat("VQSort/", local_sc_id); });
          hwy::VQSort(keys.data(), keys.size(), hwy::SortAscending());
          vqsort_traceme.Stop();

          // These counters track the number of IDs that are actually kept (not
          // dropped) for each partition and bucket for this device.
          MatrixXi kept_ids_per_sc_partition_per_bucket =
              MatrixXi::Zero(global_sc_count, bucket_count);
          MatrixXi kept_unique_ids_per_partition_per_bucket =
              MatrixXi::Zero(global_sc_count, bucket_count);
          MatrixXi ids_per_sc_partition_per_bucket =
              MatrixXi::Zero(global_sc_count, bucket_count);
          MatrixXi unique_ids_per_partition_per_bucket =
              MatrixXi::Zero(global_sc_count, bucket_count);

          const internal::LocalSparseCoreTensorGroupingContext context = {
              .keys = keys,
              .extracted_coo_tensors = extracted_coo_tensors,
              .stacked_table_metadata = stacked_table_metadata,
              .options = options,
              .local_sc_id = local_sc_id,
              .num_sc_bits = num_sc_bits,
              .grouped_coo_tensors = grouped_coo_tensors,
              .ids_per_sc_partition_per_bucket =
                  ids_per_sc_partition_per_bucket,
              .unique_ids_per_partition_per_bucket =
                  unique_ids_per_partition_per_bucket,
              .stats = stats,
              .kept_ids_per_sc_partition_per_bucket =
                  kept_ids_per_sc_partition_per_bucket,
              .kept_unique_ids_per_partition_per_bucket =
                  kept_unique_ids_per_partition_per_bucket,
          };

          internal::GroupAndDeduplicateCooTensorsForLocalSparseCore<
              kHasVariableWeights, kCreateBuckets>(context);

          grouped_coo_tensors.FillRemainingScBuckets();

          // Update global max using this device's values.
          internal::UpdateMaxIdsPerPartition(stats.max_ids_per_partition,
                                             ids_per_sc_partition_per_bucket);
          internal::UpdateMaxIdsPerPartition(
              stats.max_unique_ids_per_partition,
              unique_ids_per_partition_per_bucket);

          // Update required buffer size. `stats.required_buffer_size` is
          // 1xNumScPerDevice. We should index it by `local_sc_id`.
          Eigen::Array<int, Eigen::Dynamic, 1> partition_sizes =
              ids_per_sc_partition_per_bucket.rowwise().sum().array();
          stats.required_buffer_size[local_sc_id] +=
              partition_sizes
                  .unaryExpr([](int val) {
                    return xla::RoundUpTo(val,
                                          TPU_VECTOR_REGISTER_ALIGNMENT_SIZE);
                  })
                  .sum();

          internal::LogSparseCoreStats(local_sc_id, stacked_table_name,
                                       ids_per_sc_partition_per_bucket,
                                       unique_ids_per_partition_per_bucket,
                                       keys.size(), grouped_coo_tensors);

          const int32_t observed_max_ids_per_bucket =
              ids_per_sc_partition_per_bucket.maxCoeff();
          const int32_t observed_max_unique_ids_per_bucket =
              unique_ids_per_partition_per_bucket.maxCoeff();

          if (options.enable_minibatching) {
            internal::UpdateMinibatchingSplit(
                ids_per_sc_partition_per_bucket,
                unique_ids_per_partition_per_bucket, global_sc_count,
                max_ids_per_partition, max_unique_ids_per_partition,
                task_split);
          }
          // Only validate if creating minibatching buckets or when minibatching
          // is disabled, not when checking if minibatching is required.
          if (!options.enable_minibatching || kCreateBuckets) {
            internal::ValidateMaxIdsOrDie(
                observed_max_ids_per_bucket, observed_max_unique_ids_per_bucket,
                max_ids_per_partition, max_unique_ids_per_partition,
                stacked_table_name, options.allow_id_dropping);
          }

          total_dropped += stats.dropped_id_count;

          // Merge the parts for this task.
          SortingTaskResult task_result = {
              .grouped_coo_tensors = std::move(grouped_coo_tensors),
              .stats_host = std::move(local_stats_host),
              .dropped_id_count = total_dropped,
              .split_val = task_split};

          result_av.emplace(std::move(task_result));
          });
        });
  }

  auto device_result =
      tsl::MakeUnconstructedAsyncValueRef<DeviceSortingTaskResult>();

  // Aggregate results when all tasks are done.
  tsl::RunWhenReady(
      absl::MakeConstSpan(task_results),
      [task_results = std::move(task_results), device_result, stats,
       split_ptr = &minibatching_split,
       enable_minibatching = options.enable_minibatching,
       num_sc_per_device]() mutable {
        tsl::profiler::TraceMe t("MergeSortingTaskResults");
        std::vector<PartitionedCooTensors> parts;
        parts.reserve(num_sc_per_device);

        int total_dropped = 0;
        for (tsl::AsyncValueRef<SortingTaskResult>& res_av : task_results) {
          auto& res = res_av.get();
          parts.push_back(std::move(res.grouped_coo_tensors));

          total_dropped += res.dropped_id_count;

          // Merge Stats
          internal::StatsPerDevice res_stats =
              res.stats_host.GetStatsPerDevice(0);

          stats.max_ids_per_partition = stats.max_ids_per_partition.cwiseMax(
              res_stats.max_ids_per_partition);

          stats.max_unique_ids_per_partition =
              stats.max_unique_ids_per_partition.cwiseMax(
                  res_stats.max_unique_ids_per_partition);

          stats.required_buffer_size = stats.required_buffer_size.cwiseMax(
              res_stats.required_buffer_size);

          if (enable_minibatching) {
            *split_ptr |= res.split_val;
          }
        }

        device_result.emplace(DeviceSortingTaskResult{
            .grouped_coo_tensors =
                DevicePartitionedCooTensors{.grouped_coo_tensors =
                                                std::move(parts)},
            .total_dropped_id_count = total_dropped});
      });

  return device_result;
}

template <bool kHasVariableWeights = false, typename SplitType>
tsl::AsyncValueRef<DeviceSortingTaskResult>
SortAndGroupCooTensorsPerLocalDeviceAsync(
    const ExtractedCooTensors& extracted_coo_tensors,
    const StackedTableMetadata& stacked_table_metadata,
    const PreprocessSparseDenseMatmulInputOptions& options,
    internal::StatsPerDevice stats, SplitType& minibatching_split) {
  const bool create_buckets =
      options.enable_minibatching &&
      std::is_same_v<SplitType, MinibatchingSplit>;
  if (create_buckets) {
    return SortAndGroupCooTensorsPerLocalDeviceImpl<kHasVariableWeights, true>(
        extracted_coo_tensors, stacked_table_metadata, options, stats,
        minibatching_split);
  } else {
    return SortAndGroupCooTensorsPerLocalDeviceImpl<kHasVariableWeights, false>(
        extracted_coo_tensors, stacked_table_metadata, options, stats,
        minibatching_split);
  }
}

// Blocking version for test only.
template <bool kHasVariableWeights = false, typename SplitType>
DevicePartitionedCooTensors SortAndGroupCooTensorsPerLocalDevice(
    const ExtractedCooTensors& extracted_coo_tensors,
    const StackedTableMetadata& stacked_table_metadata,
    const PreprocessSparseDenseMatmulInputOptions& options,
    internal::StatsPerDevice& stats, SplitType& minibatching_split) {
  tsl::AsyncValueRef<DeviceSortingTaskResult> av =
      SortAndGroupCooTensorsPerLocalDeviceAsync<kHasVariableWeights>(
          extracted_coo_tensors, stacked_table_metadata, options, stats,
          minibatching_split);
  tsl::BlockUntilReady(av);
  DeviceSortingTaskResult result = std::move(av.get());
  stats.dropped_id_count = result.total_dropped_id_count;
  return std::move(result.grouped_coo_tensors);
}

}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_SORT_AND_GROUP_COO_TENSORS_IMPL_H_
