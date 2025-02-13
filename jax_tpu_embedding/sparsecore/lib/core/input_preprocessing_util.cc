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
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/numeric/bits.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "hwy/base.h"  // from @highway
#include "hwy/contrib/sort/order.h"  // from @highway
#include "hwy/contrib/sort/vqsort.h"  // from @highway
#include "tsl/profiler/lib/traceme.h"

namespace jax_sc_embedding {

int GetColId(const int col_id, const int col_shift, const int col_offset,
             const int num_scs_mod, const int num_scs_mod_inv) {
  // This is equivalent to:
  // (col_ids + col_shift) % num_sc_shards +
  //    (col_ids // num_sc_shards * num_sc_shards) + col_offset
  return ((col_id + col_shift) & num_scs_mod) + (col_id & num_scs_mod_inv) +
         col_offset;
}

void SortAndGroupCooTensors(
    absl::Span<const CooFormat> coo_tensors, const int batch_size_per_sc,
    const int num_scs, const int32_t batch_size_for_device,
    const int32_t max_ids_per_partition,
    const int32_t max_unique_ids_per_partition,
    const absl::string_view stacked_table_name, const bool allow_id_dropping,
    std::vector<std::vector<CooFormat>>& coo_tensors_by_id,
    int* aggregated_max_ids_per_sc, int* aggregated_max_unique_ids_per_sc) {
  tsl::profiler::TraceMe t("SortAndGroupCooTensors");
  const int local_sc = batch_size_for_device / batch_size_per_sc;
  uint32_t index = 0;
  const int32_t num_scs_bit = std::log2(num_scs);
  const int total_coo_tensors = coo_tensors.size();
  // Initialize the aggregated max ids and unique ids per SC to 0.
  for (int32_t i = 0; i < num_scs; ++i) {
    aggregated_max_ids_per_sc[i] = 0;
    aggregated_max_unique_ids_per_sc[i] = 0;
  }
  for (int32_t i = 0; i < local_sc; ++i) {
    std::vector<int32_t> max_ids_per_sc(num_scs, 0);
    std::vector<int32_t> max_unique_ids_per_sc(num_scs, 0);
    std::vector<uint64_t> keys;
    keys.reserve(batch_size_per_sc);
    // We take the advantage of the fact that the row_ids are already sorted
    // within each batch.
    while (index < total_coo_tensors &&
           (unsigned)(coo_tensors[index].row_id - i * batch_size_per_sc) <
               batch_size_per_sc) {
      // The key here is [col_ids % num_scs, col_ids / num_scs, index].
      // Note that this assumes `num_scs` is a power of 2.
      keys.push_back(
          (static_cast<uint64_t>(absl::rotr(
               static_cast<uint32_t>(coo_tensors[index].col_id), num_scs_bit))
           << 32) +
          index);
      ++index;
    }
    hwy::VQSort(keys.data(), keys.size(), hwy::SortAscending());

    uint32_t prev_col_id = std::numeric_limits<uint32_t>::max();
    for (const auto key : keys) {
      uint32_t sc_id = static_cast<uint32_t>(key >> (64 - num_scs_bit));
      if (static_cast<uint32_t>(key >> 32) != prev_col_id) {
        max_unique_ids_per_sc[sc_id] += 1;
      }
      max_ids_per_sc[sc_id] += 1;
      // If either max_unique_ids_per_partition or max_ids_per_partition is
      // exceeded, we drop the id.
      if (max_unique_ids_per_sc[sc_id] > max_unique_ids_per_partition ||
          max_ids_per_sc[sc_id] > max_ids_per_partition) {
        prev_col_id = static_cast<uint32_t>(key >> 32);
        continue;
      }
      coo_tensors_by_id[i].push_back(coo_tensors[static_cast<uint32_t>(key)]);
      prev_col_id = static_cast<uint32_t>(key >> 32);
    }

    for (int j = 0; j < num_scs; ++j) {
      aggregated_max_ids_per_sc[j] =
          std::max(aggregated_max_ids_per_sc[j], max_ids_per_sc[j]);
      aggregated_max_unique_ids_per_sc[j] = std::max(
          aggregated_max_unique_ids_per_sc[j], max_unique_ids_per_sc[j]);
    }
    VLOG(2) << "Observed ids per partition/sparsecore"
            << " for table " << stacked_table_name << ": ["
            << absl::StrJoin(max_ids_per_sc, ", ") << "]";

    VLOG(2) << "Observed unique ids per partition/sparsecore"
            << " for table " << stacked_table_name << ": ["
            << absl::StrJoin(max_unique_ids_per_sc, ", ") << "]";

    const int32_t observed_max_ids_per_partition =
        *absl::c_max_element(max_ids_per_sc);
    const int32_t observed_max_unique_ids_per_partition =
        *absl::c_max_element(max_unique_ids_per_sc);
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
        LOG(WARNING)
            << "Allowing ID dropping for table: " << stacked_table_name
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
}

int ComputeCooBufferSize(
    const int num_scs, const int num_scs_per_device,
    absl::Span<const StackedTableMetadata> stacked_table_metadata,
    const int static_buffer_size_multiplier) {
  const int max_ids_per_partition =
      MaxIdsPerPartitionForStackedTables(stacked_table_metadata);
  const int max_ids_rounded_up = (max_ids_per_partition + 7) & -8;
  const int theoretical_max = max_ids_rounded_up * num_scs_per_device * num_scs;
  if (static_buffer_size_multiplier <= 0) {
    return theoretical_max;
  }
  int batch_size = 0;
  for (const auto& metadata : stacked_table_metadata) {
    batch_size += metadata.batch_size;
  }
  return std::min(static_buffer_size_multiplier * batch_size, theoretical_max);
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

void FillRowPointers(absl::Span<const std::vector<CooFormat>> coo_tensors_by_id,
                     const int row_pointers_size_per_sc,
                     const int coo_buffer_size_per_sc,
                     const int batch_size_per_sc, const int num_scs,
                     const int num_sc_per_device, int* row_pointers,
                     int* embedding_ids, int* sample_ids, float* gains) {
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
