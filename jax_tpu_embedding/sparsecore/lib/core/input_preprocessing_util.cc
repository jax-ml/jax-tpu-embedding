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
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/numeric/bits.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "third_party/eigen3/Eigen/Core"
#include "hwy/contrib/sort/order.h"  // from @highway
#include "hwy/contrib/sort/vqsort.h"  // from @highway
#include "tsl/profiler/lib/traceme.h"

namespace jax_sc_embedding {
namespace {

void ValidateMaxIdsOrDie(const int32_t observed_max_ids_per_partition,
                         const int32_t observed_max_unique_ids_per_partition,
                         const int32_t max_ids_per_partition,
                         const int32_t max_unique_ids_per_partition,
                         const absl::string_view stacked_table_name,
                         const bool allow_id_dropping) {
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
                 << max_ids_per_partition;
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
                 << max_unique_ids_per_partition;
    }
  }
}

}  // namespace

int GetColId(const int col_id, const int col_shift, const int col_offset,
             const int num_scs_mod, const int num_scs_mod_inv) {
  // This is equivalent to:
  // (col_ids + col_shift) % num_sc_shards +
  //    (col_ids // num_sc_shards * num_sc_shards) + col_offset
  return ((col_id + col_shift) & num_scs_mod) + (col_id & num_scs_mod_inv) +
         col_offset;
}

RowCombiner GetRowCombiner(absl::string_view combiner) {
  if (combiner == "sum") {
    return RowCombiner::kSum;
  } else if (combiner == "mean") {
    return RowCombiner::kMean;
  } else if (combiner == "sqrtn") {
    return RowCombiner::kSqrtn;
  }
  return RowCombiner::kSum;
}

std::vector<std::vector<CooFormat>> SortAndGroupCooTensorsPerLocalDevice(
    absl::Span<const CooFormat> coo_tensors, const int batch_size_per_sc,
    const int global_sc_count, const int32_t batch_size_for_device,
    const int32_t max_ids_per_partition,
    const int32_t max_unique_ids_per_partition,
    const absl::string_view stacked_table_name, const bool allow_id_dropping,
    const int num_sc_per_device, const int total_num_coo_tensors,
    Eigen::Ref<VectorXi> max_ids_per_sc,
    Eigen::Ref<VectorXi> max_unique_ids_per_sc,
    Eigen::Ref<VectorXi> required_buffer_size_per_sc) {
  tsl::profiler::TraceMe t("SortAndGroupCooTensors");
  DCHECK_GT(batch_size_for_device, 0)
      << "Invalid batch size for device " << batch_size_for_device;
  const int local_sc_count = batch_size_for_device / batch_size_per_sc;
  std::vector<std::vector<CooFormat>> coo_tensors_by_id;
  coo_tensors_by_id.resize(num_sc_per_device);
  const int approximate_num_coo_tensors_per_sc =
      total_num_coo_tensors / num_sc_per_device + 1;
  for (int i = 0; i < num_sc_per_device; ++i) {
    // Roughly estimate the number of COO tensors for each SC.
    coo_tensors_by_id[i].reserve(approximate_num_coo_tensors_per_sc);
  }

  uint32_t coo_tensor_index = 0;
  const int32_t num_scs_bit = std::log2(global_sc_count);
  // Initialize the aggregated max ids and unique ids per SC to 0.
  max_ids_per_sc.fill(0);
  max_unique_ids_per_sc.fill(0);
  required_buffer_size_per_sc.fill(0);
  // Loop over scs for this device.
  for (int32_t local_sc_id = 0; local_sc_id < local_sc_count; ++local_sc_id) {
    std::vector<int32_t> ids_per_sc_partition(global_sc_count, 0);
    std::vector<int32_t> unique_ids_per_sc_partition(global_sc_count, 0);
    std::vector<uint64_t> keys;
    keys.reserve(batch_size_per_sc);
    // We take the advantage of the fact that the row_ids are already sorted
    // within each batch.
    for (; coo_tensor_index < coo_tensors.size() &&
           coo_tensors[coo_tensor_index].row_id <
               (local_sc_id + 1) * batch_size_per_sc;
         coo_tensor_index++) {
      // The key here is [col_ids % num_scs, col_ids / num_scs, index].
      // Note that this assumes `num_scs` is a power of 2.
      keys.push_back(
          (static_cast<uint64_t>(absl::rotr(
               static_cast<uint32_t>(coo_tensors[coo_tensor_index].col_id),
               num_scs_bit))
           << 32) +
          coo_tensor_index);
    }
    hwy::VQSort(keys.data(), keys.size(), hwy::SortAscending());

    uint32_t prev_col_id = std::numeric_limits<uint32_t>::max();
    uint32_t prev_row_id = std::numeric_limits<uint32_t>::max();
    for (const uint64_t key : keys) {
      const uint32_t index = static_cast<uint32_t>(key);
      const CooFormat& coo_tensor = coo_tensors[index];
      const uint32_t global_sc_id =
          num_scs_bit > 0 ? static_cast<uint32_t>(key >> (64 - num_scs_bit))
                          : 0;
      const uint32_t col_id = static_cast<uint32_t>(key >> 32);
      const uint32_t row_id = coo_tensor.row_id;

      if (col_id != prev_col_id) {
        unique_ids_per_sc_partition[global_sc_id] += 1;
      }

      // If the row ids and col ids are both same as the previous one,
      // dedup the id by adding the gains.
      if (col_id == prev_col_id && row_id == prev_row_id) {
        coo_tensors_by_id[local_sc_id].back().gain += coo_tensor.gain;
      } else {
        ids_per_sc_partition[global_sc_id] += 1;
        // If either max_unique_ids_per_partition or max_ids_per_partition is
        // exceeded, we drop the id.
        if (unique_ids_per_sc_partition[global_sc_id] <=
                max_unique_ids_per_partition &&
            ids_per_sc_partition[global_sc_id] <= max_ids_per_partition) {
          coo_tensors_by_id[local_sc_id].push_back(coo_tensor);
        }
      }
      prev_col_id = col_id;
      prev_row_id = row_id;
    }

    // Update global max using this device's values.
    for (int global_sc_id = 0; global_sc_id < global_sc_count; ++global_sc_id) {
      max_ids_per_sc[global_sc_id] = std::max(
          max_ids_per_sc[global_sc_id], ids_per_sc_partition[global_sc_id]);
      required_buffer_size_per_sc[local_sc_id] +=
          RoundUpTo(ids_per_sc_partition[global_sc_id],
                    TPU_VECTOR_REGISTER_ALIGMENT_SIZE);
      max_unique_ids_per_sc[global_sc_id] =
          std::max(max_unique_ids_per_sc[global_sc_id],
                   unique_ids_per_sc_partition[global_sc_id]);
    }
    if (VLOG_IS_ON(2)) {
      LOG(INFO) << "Observed ids per partition/sparsecore"
                << " for table " << stacked_table_name << ": ["
                << absl::StrJoin(ids_per_sc_partition, ", ") << "]";

      LOG(INFO) << "Observed unique ids per partition/sparsecore"
                << " for table " << stacked_table_name << ": ["
                << absl::StrJoin(unique_ids_per_sc_partition, ", ") << "]";

      LOG(INFO) << "Total number of ids for table " << stacked_table_name
                << " on Sparsecore" << local_sc_id << ": " << keys.size()
                << ", after deduplication: "
                << std::reduce(ids_per_sc_partition.begin(),
                               ids_per_sc_partition.end())
                << ", after drop id: " << coo_tensors_by_id[local_sc_id].size();
    }

    const int32_t observed_max_ids_per_partition =
        *absl::c_max_element(ids_per_sc_partition);
    const int32_t observed_max_unique_ids_per_partition =
        *absl::c_max_element(unique_ids_per_sc_partition);

    ValidateMaxIdsOrDie(observed_max_ids_per_partition,
                        observed_max_unique_ids_per_partition,
                        max_ids_per_partition, max_unique_ids_per_partition,
                        stacked_table_name, allow_id_dropping);
  }
  return coo_tensors_by_id;
}
int ComputeCooBufferSizePerDevice(
    const int num_scs, const int num_scs_per_device,
    absl::Span<const StackedTableMetadata> stacked_table_metadata) {
  const int max_ids_per_partition =
      MaxIdsPerPartitionForStackedTables(stacked_table_metadata);
  const std::optional<int> suggested_coo_buffer_size =
      SuggestedCooBufferSizeForStackedTables(stacked_table_metadata);

  const int64_t max_ids_rounded_up = RoundUpTo<int64_t>(
      max_ids_per_partition, TPU_VECTOR_REGISTER_ALIGMENT_SIZE);
  const int64_t theoretical_max =
      max_ids_rounded_up * num_scs_per_device * num_scs;
  VLOG(1) << "Theoretical Max: " << theoretical_max;
  int64_t result = theoretical_max;
  if (suggested_coo_buffer_size.has_value()) {
    VLOG(1) << "Suggested Coo Buffer Size: "
            << suggested_coo_buffer_size.value();
    // Since the suggested size corresponds to only current device (local SCs),
    // Buffer for each SC should be properly aligned, hence ALIGNMENT *
    // num_scs_per_device
    result = std::min<int64_t>(
        result, RoundUpTo<int64_t>(
                    suggested_coo_buffer_size.value(),
                    TPU_VECTOR_REGISTER_ALIGMENT_SIZE * num_scs_per_device));
  } else {
    LOG_EVERY_POW_2(WARNING)
        << "No Coo Buffer Size provided for table "
        << stacked_table_metadata[0].name << ", the default value ("
        << theoretical_max
        << ") may be too "
           "large and can cause OOM. Utilize the stats returned from "
           "the sparse dense matmul preprocessing API.";
  }
  VLOG(1) << "Computed Coo Buffer Size: " << result;
  // The result could be very large and cause overflow. We need to make
  // sure the result is within the range of int before using it.
  CHECK(result > 0 && result < INT_MAX);
  return static_cast<int>(result);
}

void IncrementScId(std::pair<int, int>& sc_id, const int num_scs,
                   const int num_scs_per_device) {
  CHECK(sc_id.first < num_scs_per_device)
      << "Invalid SC ID tuple increment " << sc_id.first << ", "
      << sc_id.second;
  if (sc_id.second < num_scs - 1) {
    ++sc_id.second;
    return;
  }
  ++sc_id.first;
  sc_id.second = 0;
}

int MaxIdsPerPartitionForStackedTables(
    const absl::Span<const StackedTableMetadata> stacked_table_metadata) {
  int max_ids_per_partition = stacked_table_metadata[0].max_ids_per_partition;
  DCHECK_GT(max_ids_per_partition, 0);
  return max_ids_per_partition;
}

std::optional<int> SuggestedCooBufferSizeForStackedTables(
    const absl::Span<const StackedTableMetadata> stacked_table_metadata) {
  std::optional<int> suggested_coo_buffer_size =
      stacked_table_metadata[0].suggested_coo_buffer_size;
  return suggested_coo_buffer_size;
}

void FillRowPointersPerLocalDevice(
    absl::Span<const std::vector<CooFormat>> coo_tensors_by_id,
    const int row_pointers_size_per_sc, const int coo_buffer_size_per_sc,
    const int batch_size_per_sc, const int num_scs, const int num_sc_per_device,
    Eigen::Ref<VectorXi> row_pointers, Eigen::Ref<VectorXi> embedding_ids,
    Eigen::Ref<VectorXi> sample_ids, Eigen::Ref<VectorXf> gains) {
  tsl::profiler::TraceMe t("FillRowPointers");
  for (int local_sc_id = 0; local_sc_id < num_sc_per_device; ++local_sc_id) {
    int lhs_row_index = 0;
    int padded_coo_tensor_index = 0;
    auto last_sc_id = std::make_pair(local_sc_id, 0);

    const int row_pointer_index_start = local_sc_id * row_pointers_size_per_sc;
    const int buffer_index_start = local_sc_id * coo_buffer_size_per_sc;
    const int total_coos = coo_tensors_by_id[local_sc_id].size();
    int processed = 0;
    for (const CooFormat& coo_tensor : coo_tensors_by_id[local_sc_id]) {
      ++processed;
      const auto sc_id = std::make_pair(coo_tensor.row_id / batch_size_per_sc,
                                        coo_tensor.col_id % num_scs);
      while (last_sc_id < sc_id) {
        if (lhs_row_index >= row_pointers_size_per_sc ||
            padded_coo_tensor_index >= coo_buffer_size_per_sc) {
          LOG(ERROR) << "Static buffer size maybe too small for current "
                        "batch. IDs may be dropped! Static buffer size: "
                     << coo_buffer_size_per_sc * num_sc_per_device
                     << ". Halting row pointer filling at while processing for "
                        "local sparsecore ID "
                     << local_sc_id << ". Total COOs: " << total_coos
                     << ", while currently processed only: " << processed - 1;
          break;
        }
        row_pointers[row_pointer_index_start + lhs_row_index] =
            padded_coo_tensor_index;
        ++lhs_row_index;

        while (padded_coo_tensor_index & 7 &&
               padded_coo_tensor_index < coo_buffer_size_per_sc) {
          const int current_index =
              buffer_index_start + padded_coo_tensor_index;
          ++padded_coo_tensor_index;
          embedding_ids[current_index] = INT_MAX;
          sample_ids[current_index] = INT_MAX;
          gains[current_index] = std::nanf("");
        }
        IncrementScId(last_sc_id, num_scs, num_sc_per_device);
      }
      if (coo_tensor.row_id == batch_size_per_sc * (local_sc_id + 1)) {
        break;
      }
      if (lhs_row_index >= row_pointers_size_per_sc ||
          padded_coo_tensor_index >= coo_buffer_size_per_sc) {
        LOG(ERROR) << "Static buffer size maybe too small for current "
                      "batch. IDs may be dropped! Static buffer size: "
                   << coo_buffer_size_per_sc * num_sc_per_device
                   << ". Halting row pointer filling at while processing for "
                      "local sparsecore ID "
                   << local_sc_id << ". Total COOs: " << total_coos
                   << ", while currently processed only: " << processed - 1;
        break;
      }

      const int current_index = buffer_index_start + padded_coo_tensor_index;
      ++padded_coo_tensor_index;
      embedding_ids[current_index] = coo_tensor.col_id / num_scs;
      sample_ids[current_index] = coo_tensor.row_id % batch_size_per_sc;
      gains[current_index] = coo_tensor.gain;
    }

    while (lhs_row_index < row_pointers_size_per_sc) {
      row_pointers[row_pointer_index_start + lhs_row_index] =
          padded_coo_tensor_index;
      ++lhs_row_index;
    }

    while (padded_coo_tensor_index < coo_buffer_size_per_sc) {
      const int current_index = buffer_index_start + padded_coo_tensor_index;
      ++padded_coo_tensor_index;
      embedding_ids[current_index] = INT_MAX;
      sample_ids[current_index] = INT_MAX;
      gains[current_index] = std::nanf("");
    }
  }
}

}  // namespace jax_sc_embedding
