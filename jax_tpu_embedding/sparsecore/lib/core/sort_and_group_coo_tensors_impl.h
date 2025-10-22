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

#include <cstdint>
#include <limits>
#include <vector>

#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/numeric/bits.h"  // from @com_google_absl
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
  // Index = 0 to kIndexMask giving us a count of kIndexMask + 1.
  if (key_count > 1 + CooFormat::kIndexMask) {
    LOG(ERROR) << absl::StrFormat(
        "Too many tensors for SparseCore #%d: got %d, limit: "
        "%d. Preprocessed output may not be reliable and cause undefined "
        "behavior.",
        local_sc_id, key_count, CooFormat::kIndexMask);
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

}  // namespace internal

// Sorts and groups the provided COO tensors in this hierarchy: Local SC ->
// Minibatching Bucket -> Global SC.
//
//
// NOTE: We use output buffers `max_ids_per_sc`, `max_unique_ids_per_sc`, and
// `required_buffer_size_per_sc` because we fill values in a loop to a bigger
// array.
template <typename SplitType>
PartitionedCooTensors SortAndGroupCooTensorsPerLocalDevice(
    const ExtractedCooTensors& extracted_coo_tensors,
    const StackedTableMetadata& stacked_table_metadata,
    const PreprocessSparseDenseMatmulInputOptions& options,
    internal::StatsPerDevice& stats, SplitType& minibatching_split) {
  tsl::profiler::TraceMe t("SortAndGroupCooTensors");
  const std::vector<CooFormat>& coo_tensors = extracted_coo_tensors.coo_tensors;
  const int num_sc_per_device = options.num_sc_per_device;
  bool allow_id_dropping = options.allow_id_dropping;
  const int batch_size_per_sc = xla::CeilOfRatio(
      extracted_coo_tensors.batch_size_for_device, options.num_sc_per_device);
  const uint32_t global_sc_count = options.GetNumScs();
  const int num_sc_bits = absl::bit_width(global_sc_count - 1);
  const int max_ids_per_partition =
      stacked_table_metadata.max_ids_per_partition;
  const int max_unique_ids_per_partition =
      stacked_table_metadata.max_unique_ids_per_partition;
  const absl::string_view stacked_table_name = stacked_table_metadata.name;
  // Minibatching is enabled and we need to create buckets for minibatching.
  const bool create_buckets = options.enable_minibatching &&
                              (std::is_same_v<SplitType, MinibatchingSplit>);

  // Partition COO tensors among SparseCores for the local device (based on row
  // id).
  const int bucket_count =
      create_buckets ? CooFormat::kMaxMinibatchingBuckets : 1;
  PartitionedCooTensors grouped_coo_tensors(
      coo_tensors.size(), num_sc_per_device, global_sc_count, bucket_count);

  uint32_t coo_tensor_index = 0;

  // Loop over scs for this device.
  for (int32_t local_sc_id = 0; local_sc_id < options.num_sc_per_device;
       ++local_sc_id) {
    MatrixXi ids_per_sc_partition_per_bucket =
        MatrixXi::Zero(global_sc_count, bucket_count);
    MatrixXi unique_ids_per_partition_per_bucket =
        MatrixXi::Zero(global_sc_count, bucket_count);
    std::vector<uint64_t> keys;
    const int expected_keys_size =
        extracted_coo_tensors.coo_tensors_per_sc[local_sc_id];
    keys.reserve(expected_keys_size);
    internal::ValidateKeyCapacity(local_sc_id, expected_keys_size);
    // We take the advantage of the fact that the row_ids are already sorted
    // within each batch.
    for (; coo_tensor_index < coo_tensors.size() &&
           coo_tensors[coo_tensor_index].row_id <
               (local_sc_id + 1) * batch_size_per_sc;
         coo_tensor_index++) {
      // The key here is [bucket_id(6 bits), global_sc_id(num_scs bits),
      // local_embedding_id(32-num_scs bits), index(26 bits)].
      //  Note that this assumes `num_scs` is a power of 2.
      keys.push_back(coo_tensors[coo_tensor_index].GetGroupingKey(
          num_sc_bits, coo_tensor_index, create_buckets,
          options.minibatching_bucketing_hash_fn));
    }

    // The expected allocation size may be uninitialized.
    DCHECK(expected_keys_size == 0 || keys.size() == expected_keys_size);
    hwy::VQSort(keys.data(), keys.size(), hwy::SortAscending());

    uint32_t prev_col_id = std::numeric_limits<uint32_t>::max();
    uint32_t prev_row_id = std::numeric_limits<uint32_t>::max();
    uint32_t prev_bucket_id = 0;
    for (const uint64_t key : keys) {
      uint32_t index = key & CooFormat::kIndexMask;

      const CooFormat& coo_tensor = coo_tensors[index];
      const uint32_t col_id = coo_tensor.col_id;
      const uint32_t global_sc_id = coo_tensor.col_id & (global_sc_count - 1);
      const uint32_t bucket_id =
          create_buckets
              ? coo_tensor.GetBucketId(options.minibatching_bucketing_hash_fn)
              : 0;
      const uint32_t row_id = coo_tensor.row_id;

      if (bucket_id != prev_bucket_id || col_id != prev_col_id) {
        unique_ids_per_partition_per_bucket(global_sc_id, bucket_id) += 1;
      }

      // If the row ids and col ids are both same as the previous one,
      // dedup the id by adding the gains.
      if (col_id == prev_col_id && row_id == prev_row_id) {
        grouped_coo_tensors.MergeWithLastCoo(coo_tensor);
      } else {
        ids_per_sc_partition_per_bucket(global_sc_id, bucket_id) += 1;
        // If either max_unique_ids_per_partition or max_ids_per_partition is
        // exceeded, we drop the id. For minibatching, if even the smallest
        // bucket exceeds the capacity, we drop the id, since minibatching can't
        // help us.
        const bool over_capacity =
            unique_ids_per_partition_per_bucket(global_sc_id, bucket_id) >
                max_unique_ids_per_partition ||
            ids_per_sc_partition_per_bucket(global_sc_id, bucket_id) >
                max_ids_per_partition;
        if (over_capacity) {
          // Dropped id.
          ++stats.dropped_id_count;
          continue;
        } else {
          grouped_coo_tensors.Add(local_sc_id, bucket_id, coo_tensor);
        }
      }
      prev_col_id = col_id;
      prev_row_id = row_id;
      prev_bucket_id = bucket_id;
    }
    grouped_coo_tensors.FillRemainingScBuckets();

    // Update global max using this device's values.
    internal::UpdateMaxIdsPerPartition(stats.max_ids_per_partition,
                                       ids_per_sc_partition_per_bucket);
    internal::UpdateMaxIdsPerPartition(stats.max_unique_ids_per_partition,
                                       unique_ids_per_partition_per_bucket);
    auto partition_sizes =
        ids_per_sc_partition_per_bucket.rowwise().sum().array();
    stats.required_buffer_size[local_sc_id] +=
        partition_sizes
            .unaryExpr([](int val) {
              return xla::RoundUpTo(val, TPU_VECTOR_REGISTER_ALIGNMENT_SIZE);
            })
            .sum();

    if (VLOG_IS_ON(2)) {
      LOG(INFO) << "Observed ids per partition/sparsecore"
                << " for table " << stacked_table_name << ": "
                << ids_per_sc_partition_per_bucket.rowwise().sum();

      LOG(INFO) << "Observed unique ids per partition/sparsecore"
                << " for table " << stacked_table_name << ": "
                << unique_ids_per_partition_per_bucket.rowwise().sum();

      LOG(INFO) << "Total number of ids for table " << stacked_table_name
                << " on SparseCore" << local_sc_id << ": " << keys.size()
                << ", after deduplication: "
                << ids_per_sc_partition_per_bucket.sum()
                << ", after drop id: " << grouped_coo_tensors.Size(local_sc_id);
    }

    const int32_t observed_max_ids_per_bucket =
        ids_per_sc_partition_per_bucket.maxCoeff();
    const int32_t observed_max_unique_ids_per_bucket =
        unique_ids_per_partition_per_bucket.maxCoeff();

    if (options.enable_minibatching) {
      // This works both when minibatching is required and not. In the former
      // case we have bool which tells us if minibatching is required,
      // in the latter case it is std::bitset<64> which tells us the exact
      // splits.
      for (int global_sc_id = 0; global_sc_id < global_sc_count;
           ++global_sc_id) {
        auto ids_per_bucket =
            ids_per_sc_partition_per_bucket.row(global_sc_id).array();
        auto unique_ids_per_bucket =
            unique_ids_per_partition_per_bucket.row(global_sc_id).array();
        if constexpr (std::is_same_v<SplitType, MinibatchingSplit>) {
          // absl::Makespan works because the array would be row-major and
          // values would be contiguous in memory.
          static_assert(decltype(ids_per_bucket)::IsRowMajor);
          static_assert(decltype(unique_ids_per_bucket)::IsRowMajor);
          // NOTE: ComputeSplit modifies the span, but we have already updated
          // the output stats.
          minibatching_split |= internal::ComputeMinibatchingSplit(
              absl::MakeSpan(ids_per_bucket), max_ids_per_partition);
          minibatching_split |= internal::ComputeMinibatchingSplit(
              absl::MakeSpan(unique_ids_per_bucket),
              max_unique_ids_per_partition);
        } else {
          minibatching_split |=
              unique_ids_per_bucket.maxCoeff() > max_unique_ids_per_partition ||
              ids_per_bucket.maxCoeff() > max_ids_per_partition;
        }
      }
    }

    // Only validate if creating minibatching buckets or when minibatching is
    // disabled, not when checking if minibatching is required.
    if (!options.enable_minibatching || create_buckets)
      internal::ValidateMaxIdsOrDie(
          observed_max_ids_per_bucket, observed_max_unique_ids_per_bucket,
          max_ids_per_partition, max_unique_ids_per_partition,
          stacked_table_name, allow_id_dropping);
  }

  return grouped_coo_tensors;
}

}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_SORT_AND_GROUP_COO_TENSORS_IMPL_H_
