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
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/numeric/bits.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "Eigen/Core"  // from @eigen_archive
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

struct FillRowPointersPerSparseCoreOptions {
  int local_sc_id;
  absl::Span<const CooFormat> coo_tensors;

  int lhs_row_begin;
  int lhs_row_end;

  int coo_begin;
  int coo_end;

  int batch_size_per_sc;
  int num_sc_per_device;
  int num_scs;
  int coo_buffer_size;
};

// Check if the current indexes are valid within the buffer sizes.
bool ValidIndices(int row_index, int coo_offset, int processed,
                  const FillRowPointersPerSparseCoreOptions& options) {
  if (row_index >= options.lhs_row_end || coo_offset >= options.coo_end) {
    LOG(ERROR) << "Static buffer size maybe too small for current "
                  "batch. IDs may be dropped! Static buffer size: "
               << options.coo_buffer_size
               << ". Halting row pointer filling at while processing for local "
                  "sparsecore ID "
               << options.local_sc_id
               << ". Total COOs: " << options.coo_tensors.size()
               << ", while currently processed only: " << processed - 1;
    return false;
  }
  return true;
}

// Pad the row pointers buffer to the end of the buffer.
void PadRowPointersBuffer(int& lhs_row_offset, int padding, int row_end,
                          Eigen::Ref<RowVectorXi> row_pointers) {
  while (lhs_row_offset < row_end) {
    row_pointers[lhs_row_offset++] = padding;
  }
}

// Pad the sparse core buffer.
// Args:
//   pad_to_end: whether to pad to the end of the buffer or just align to HBM
//     granularity.
void PadSparseCoreBuffer(int& coo_index, int coo_end, bool pad_to_end,
                         Eigen::Ref<RowVectorXi> embedding_ids,
                         Eigen::Ref<RowVectorXi> sample_ids,
                         Eigen::Ref<RowVectorXf> gains) {
  while ((pad_to_end || coo_index % TPU_VECTOR_REGISTER_ALIGMENT_SIZE != 0) &&
         coo_index < coo_end) {
    embedding_ids[coo_index] = INT_MAX;
    sample_ids[coo_index] = INT_MAX;
    gains[coo_index] = std::nanf("");
    ++coo_index;
  }
}

// Fill the row pointers buffer for a single sparse core from `lhs_row_begin` to
// `lhs_row_end` and coo buffer from `coo_begin` to `coo_end`. Returns the
// `coo_index` from where next the buffer can be filled.
int FillRowPointersPerSparseCore(
    const FillRowPointersPerSparseCoreOptions& options,
    Eigen::Ref<RowVectorXi> row_pointers, Eigen::Ref<RowVectorXi> embedding_ids,
    Eigen::Ref<RowVectorXi> sample_ids, Eigen::Ref<RowVectorXf> gains) {
  int lhs_row_index = options.lhs_row_begin;
  int coo_index = options.coo_begin;
  auto last_sc_id = std::make_pair(options.local_sc_id, 0);

  int processed = 0;
  for (const CooFormat& coo_tensor : options.coo_tensors) {
    ++processed;
    const auto sc_id =
        std::make_pair(coo_tensor.row_id / options.batch_size_per_sc,
                       coo_tensor.col_id % options.num_scs);
    while (last_sc_id < sc_id) {
      if (!ValidIndices(lhs_row_index, coo_index, processed, options)) {
        break;
      }
      row_pointers[lhs_row_index++] = coo_index - options.coo_begin;
      // Align partition.
      PadSparseCoreBuffer(coo_index, options.coo_end,
                          /*pad_to_end=*/false, embedding_ids, sample_ids,
                          gains);
      IncrementScId(last_sc_id, options.num_scs, options.num_sc_per_device);
    }
    // Terminate at the sentinel node.
    if (coo_tensor.row_id ==
        options.batch_size_per_sc * (options.local_sc_id + 1)) {
      DCHECK_EQ(coo_tensor.gain, 0);
      break;
    }
    if (!ValidIndices(lhs_row_index, coo_index, processed, options)) {
      break;
    }

    embedding_ids[coo_index] = coo_tensor.col_id / options.num_scs;
    sample_ids[coo_index] = coo_tensor.row_id % options.batch_size_per_sc;
    gains[coo_index] = coo_tensor.gain;
    ++coo_index;
  }

  PadRowPointersBuffer(lhs_row_index, /*padding=*/coo_index - options.coo_begin,
                       options.lhs_row_end, row_pointers);
  return coo_index;
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

// We use output buffers `max_ids_per_sc`, `max_unique_ids_per_sc`, and
// `required_buffer_size_per_sc` because we fill values in a loop to a bigger
// array.
std::vector<std::vector<CooFormat>> SortAndGroupCooTensorsPerLocalDevice(
    const ExtractedCooTensors& extracted_coo_tensors,
    const StackedTableMetadata& stacked_table_metadata,
    const PreprocessSparseDenseMatmulInputOptions& options,
    Eigen::Ref<RowVectorXi> max_ids_per_sc,
    Eigen::Ref<RowVectorXi> max_unique_ids_per_sc,
    Eigen::Ref<RowVectorXi> required_buffer_size_per_sc) {
  tsl::profiler::TraceMe t("SortAndGroupCooTensors");
  const std::vector<CooFormat>& coo_tensors = extracted_coo_tensors.coo_tensors;
  const int num_sc_per_device = options.num_sc_per_device;
  bool allow_id_dropping = options.allow_id_dropping;
  const int batch_size_per_sc = CeilOfRatio(
      extracted_coo_tensors.batch_size_for_device, options.num_sc_per_device);
  const int global_sc_count = options.GetNumScs();
  const int max_ids_per_partition =
      stacked_table_metadata.max_ids_per_partition;
  const int max_unique_ids_per_partition =
      stacked_table_metadata.max_unique_ids_per_partition;
  const absl::string_view stacked_table_name = stacked_table_metadata.name;

  // Partition COO tensors among SparseCores for the local device (based on row
  // id).
  std::vector<std::vector<CooFormat>> coo_tensors_by_sc_id;
  coo_tensors_by_sc_id.resize(num_sc_per_device);
  for (int i = 0; i < num_sc_per_device; ++i) {
    // An additional capacity for the sentinel node. Might underutilize if
    // deduplication happens.
    coo_tensors_by_sc_id[i].reserve(
        1 + extracted_coo_tensors.coo_tensors_per_sc[i]);
  }

  uint32_t coo_tensor_index = 0;
  const int32_t num_scs_bit = std::log2(global_sc_count);
  // Initialize the aggregated max ids and unique ids per SC to 0.
  max_ids_per_sc.fill(0);
  max_unique_ids_per_sc.fill(0);
  required_buffer_size_per_sc.fill(0);
  // Loop over scs for this device.
  for (int32_t local_sc_id = 0; local_sc_id < options.num_sc_per_device;
       ++local_sc_id) {
    std::vector<int32_t> ids_per_sc_partition(global_sc_count, 0);
    std::vector<int32_t> unique_ids_per_sc_partition(global_sc_count, 0);
    std::vector<uint64_t> keys;
    const int expected_keys_size =
        extracted_coo_tensors.coo_tensors_per_sc[local_sc_id];
    keys.reserve(expected_keys_size);
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

    // The expected allocation size may be uninitialized.
    DCHECK(expected_keys_size == 0 || keys.size() == expected_keys_size);
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
        coo_tensors_by_sc_id[local_sc_id].back().gain += coo_tensor.gain;
      } else {
        ids_per_sc_partition[global_sc_id] += 1;
        // If either max_unique_ids_per_partition or max_ids_per_partition is
        // exceeded, we drop the id.
        if (unique_ids_per_sc_partition[global_sc_id] <=
                max_unique_ids_per_partition &&
            ids_per_sc_partition[global_sc_id] <= max_ids_per_partition) {
          coo_tensors_by_sc_id[local_sc_id].push_back(coo_tensor);
        }
      }
      prev_col_id = col_id;
      prev_row_id = row_id;
    }
    // Sentinel node to terminate buffer filling.
    coo_tensors_by_sc_id[local_sc_id].emplace_back(
        batch_size_per_sc * (local_sc_id + 1), 0, 0.0);
    required_buffer_size_per_sc[local_sc_id]++;

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
                << ", after drop id: "
                << coo_tensors_by_sc_id[local_sc_id].size();
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
  return coo_tensors_by_sc_id;
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
  const std::string& stacked_table_name = stacked_table_metadata[0].name;
  // Without unrolling the loop across threads and stacked tables, we need to
  // only log once per table (across epochs). This function would be called for
  // every epoch and every stacked table.We use a static limit of 20 stacked
  // tables. There might be a few repeated logs, if we don't have as many
  // tables.
  static constexpr int kStackedTableLogCountLimit = 20;
  LOG_FIRST_N(INFO, kStackedTableLogCountLimit)
      << "Theoretical Max for table " << stacked_table_name << ": "
      << theoretical_max << "( max_ids_rounded_up: " << max_ids_rounded_up
      << " num_scs_per_device: " << num_scs_per_device
      << " num_scs: " << num_scs << ")";
  int64_t result = theoretical_max;
  if (suggested_coo_buffer_size.has_value()) {
    LOG_FIRST_N(INFO, kStackedTableLogCountLimit)
        << "Suggested Coo Buffer Size for table " << stacked_table_name << ": "
        << suggested_coo_buffer_size.value();
    // Since the suggested size corresponds to only current device (local SCs),
    // Buffer for each SC should be properly aligned, hence ALIGNMENT *
    // num_scs_per_device
    result = RoundUpTo<int64_t>(
        suggested_coo_buffer_size.value(),
        TPU_VECTOR_REGISTER_ALIGMENT_SIZE * num_scs_per_device);
  } else {
    LOG_FIRST_N(WARNING, kStackedTableLogCountLimit)
        << "No Coo Buffer Size provided for table " << stacked_table_name
        << ", the default value (" << theoretical_max
        << ") may be too "
           "large and can cause OOM. Utilize the stats returned from "
           "the sparse dense matmul preprocessing API.";
  }
  LOG_FIRST_N(INFO, kStackedTableLogCountLimit)
      << "Computed Coo Buffer Size for table " << stacked_table_name << ": "
      << result;
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

// We use output buffers `row_pointers`, `embedding_ids`, `sample_ids`, and
// `gains` because we fill values in a loop to a bigger array.
void FillRowPointersPerLocalDevice(
    absl::Span<const std::vector<CooFormat>> coo_tensors_by_id,
    const int row_pointers_size_per_sc, const int coo_buffer_size_per_sc,
    const int batch_size_per_sc,
    const PreprocessSparseDenseMatmulInputOptions& options,
    Eigen::Ref<RowVectorXi> row_pointers, Eigen::Ref<RowVectorXi> embedding_ids,
    Eigen::Ref<RowVectorXi> sample_ids, Eigen::Ref<RowVectorXf> gains) {
  tsl::profiler::TraceMe t("FillRowPointers");
  const int num_sc_per_device = options.num_sc_per_device;
  const int num_scs = options.GetNumScs();
  const int coo_buffer_size = coo_buffer_size_per_sc * num_sc_per_device;
  DCHECK_GT(batch_size_per_sc, 0);
  int coo_begin = 0;
  for (int local_sc_id = 0; local_sc_id < num_sc_per_device; ++local_sc_id) {
    const int lhs_row_begin = local_sc_id * row_pointers_size_per_sc;
    const int lhs_row_end = lhs_row_begin + row_pointers_size_per_sc;
    const int coo_end =
        options.enable_minibatching
            ? coo_buffer_size                      // use whole buffer
            : coo_begin + coo_buffer_size_per_sc;  // partition coo buffer
    // TODO: b/428790659 - Loop over minibatches here.
    int coo_index = FillRowPointersPerSparseCore(
        {
            .local_sc_id = local_sc_id,
            .coo_tensors = coo_tensors_by_id[local_sc_id],
            .lhs_row_begin = lhs_row_begin,
            .lhs_row_end = lhs_row_end,
            .coo_begin = coo_begin,
            .coo_end = coo_end,
            .batch_size_per_sc = batch_size_per_sc,
            .num_sc_per_device = num_sc_per_device,
            .num_scs = num_scs,
            .coo_buffer_size = coo_buffer_size,
        },
        row_pointers, embedding_ids, sample_ids, gains);

    // Only pad (to end of COO buffer) between SparseCores for
    // non-mini-batching. Always pad after the last SparseCore. The end could be
    // different per SparseCore for non-minibatching.
    if (!options.enable_minibatching || local_sc_id == num_sc_per_device - 1) {
      PadSparseCoreBuffer(coo_index, coo_end,
                          /*pad_to_end=*/true, embedding_ids, sample_ids,
                          gains);
      CHECK_EQ(coo_index, coo_end);
    }
    coo_begin = coo_index;
  }
}

}  // namespace jax_sc_embedding
