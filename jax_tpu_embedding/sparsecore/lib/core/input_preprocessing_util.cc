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

#include <climits>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "Eigen/Core"  // from @eigen_archive
#include "jax_tpu_embedding/sparsecore/lib/core/coo_format.h"
#include "jax_tpu_embedding/sparsecore/lib/core/partitioned_coo_tensors.h"
#include "tsl/profiler/lib/traceme.h"

namespace jax_sc_embedding {
namespace {

struct BufferFillingOptions {
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
                  const BufferFillingOptions& options) {
  if (row_index >= options.lhs_row_end || coo_offset >= options.coo_end) {
    LOG(ERROR) << "The static buffer size might be too small for the current "
                  "batch. IDs may be dropped! "
               << "Stopping row pointer filling for local SparseCore ID "
               << options.local_sc_id << " at row index: " << row_index
               << " and coo offset: " << coo_offset
               << " because it reached the end of the buffer. "
               << "The buffer size is: " << options.coo_buffer_size
               << ". The coo_end is: " << options.coo_end
               << ". Total number of input COOs: " << options.coo_tensors.size()
               << ", while currently processed only: " << processed;
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

// Pad the COO buffer.
// Args:
//   pad_to_end: whether to pad to the end of the buffer or just align to HBM
//     granularity.
void PadCooBuffer(int& coo_index, int coo_end, bool pad_to_end,
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

// Fill the row pointers buffer from `lhs_row_begin` to `lhs_row_end` and COO
// buffer from `coo_begin` to `coo_end`. Returns the `coo_index` from where next
// the COO buffer can be filled.
int FillMinibatchBuffer(const BufferFillingOptions& options,
                        Eigen::Ref<RowVectorXi> row_pointers,
                        Eigen::Ref<RowVectorXi> embedding_ids,
                        Eigen::Ref<RowVectorXi> sample_ids,
                        Eigen::Ref<RowVectorXf> gains) {
  int lhs_row_index = options.lhs_row_begin;
  int coo_index = options.coo_begin;
  auto last_sc_id = std::make_pair(options.local_sc_id, 0);

  int processed = 0;
  for (const CooFormat& coo_tensor : options.coo_tensors) {
    const bool is_sentinel = coo_tensor.row_id == options.batch_size_per_sc *
                                                      (options.local_sc_id + 1);
    if (!is_sentinel) ++processed;
    const auto sc_id = std::make_pair(
        /* local_sc_id= */ coo_tensor.row_id / options.batch_size_per_sc,
        /* global_sc_id= */ coo_tensor.col_id % options.num_scs);
    DCHECK_GE(sc_id, last_sc_id) << absl::StrFormat(
        "Failed: sc_id@(%d, %d) >= last_sc_id@(%d, %d)", sc_id.first,
        sc_id.second, last_sc_id.first, last_sc_id.second);
    // Advance last_sc_id to current sc_id.
    while (last_sc_id < sc_id) {
      if (!ValidIndices(lhs_row_index, coo_index, processed, options)) {
        break;
      }
      row_pointers[lhs_row_index++] = coo_index - options.coo_begin;
      // Align partition.
      PadCooBuffer(coo_index, options.coo_end,
                   /*pad_to_end=*/false, embedding_ids, sample_ids, gains);
      IncrementScId(last_sc_id, options.num_scs, options.num_sc_per_device);
    }
    // Terminate at the sentinel node.
    if (is_sentinel) {
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

int ComputeCooBufferSizePerDevice(
    const int num_scs, const int num_scs_per_device,
    absl::Span<const StackedTableMetadata> stacked_table_metadata,
    const int batch_number) {
  const int max_ids_per_partition =
      MaxIdsPerPartitionForStackedTables(stacked_table_metadata);
  const std::optional<int> suggested_coo_buffer_size_per_device =
      SuggestedCooBufferSizeForStackedTables(stacked_table_metadata);

  const int64_t max_ids_rounded_up = RoundUpTo<int64_t>(
      max_ids_per_partition, TPU_VECTOR_REGISTER_ALIGMENT_SIZE);
  const int64_t theoretical_max =
      max_ids_rounded_up * num_scs_per_device * num_scs;
  const std::string& stacked_table_name = stacked_table_metadata[0].name;
  LOG_IF(INFO, batch_number % 100 == 0)
      << "Theoretical Max for table " << stacked_table_name << ": "
      << theoretical_max << "( max_ids_rounded_up: " << max_ids_rounded_up
      << " num_scs_per_device: " << num_scs_per_device
      << " num_scs: " << num_scs << ")";
  int64_t result = theoretical_max;
  if (suggested_coo_buffer_size_per_device.has_value()) {
    LOG_IF(INFO, batch_number % 100 == 0)
        << "Suggested Coo Buffer Size for table " << stacked_table_name << ": "
        << suggested_coo_buffer_size_per_device.value();
    // Since the suggested size corresponds to only current device (local SCs),
    // Buffer for each SC should be properly aligned, hence ALIGNMENT *
    // num_scs_per_device
    result = RoundUpTo<int64_t>(
        suggested_coo_buffer_size_per_device.value(),
        TPU_VECTOR_REGISTER_ALIGMENT_SIZE * num_scs_per_device);
  } else {
    LOG_IF(WARNING, batch_number % 100 == 0)
        << "No Coo Buffer Size provided for table " << stacked_table_name
        << ", the default value (" << theoretical_max
        << ") may be too "
           "large and can cause OOM. Utilize the stats returned from "
           "the sparse dense matmul preprocessing API.";
  }
  LOG_IF(INFO, batch_number % 100 == 0) << "Computed Coo Buffer Size for table "
                                        << stacked_table_name << ": " << result;
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
  std::optional<int> suggested_coo_buffer_size_per_device =
      stacked_table_metadata[0].suggested_coo_buffer_size_per_device;
  return suggested_coo_buffer_size_per_device;
}

// We use output buffers `row_pointers`, `embedding_ids`, `sample_ids`, and
// `gains` because we fill values in a loop to a bigger array.
void FillLocalDeviceBuffer(
    const PartitionedCooTensors& grouped_coo_tensors,
    const int row_pointers_size_per_bucket, const int coo_buffer_size_per_sc,
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
  int lhs_row_begin = 0;
  for (int local_sc_id = 0; local_sc_id < num_sc_per_device; ++local_sc_id) {
    for (int minibatch_id = 0;
         minibatch_id < grouped_coo_tensors.GetNumMinibatches();
         ++minibatch_id) {
      const int lhs_row_end = lhs_row_begin + row_pointers_size_per_bucket;
      const int coo_end =
          options.enable_minibatching
              ? coo_buffer_size                      // use whole buffer
              : coo_begin + coo_buffer_size_per_sc;  // partition coo buffer
      coo_begin = FillMinibatchBuffer(
          {
              .local_sc_id = local_sc_id,
              .coo_tensors = grouped_coo_tensors(local_sc_id, minibatch_id),
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
      lhs_row_begin = lhs_row_end;
      if (options.enable_minibatching) {
        // Align minibatch buffer
        PadCooBuffer(coo_begin, coo_buffer_size,
                     /*pad_to_end=*/false, embedding_ids, sample_ids, gains);
      }
    }  // end minibatch loop
    if (!options.enable_minibatching) {
      const int sc_end = (local_sc_id + 1) * coo_buffer_size_per_sc;
      // Pad to end of SparseCore buffer.
      PadCooBuffer(coo_begin, sc_end,
                   /*pad_to_end=*/true, embedding_ids, sample_ids, gains);
    }
  }  // end SparseCore loop
  // Pad to end of device buffer.
  PadCooBuffer(coo_begin, coo_buffer_size,
               /*pad_to_end=*/true, embedding_ids, sample_ids, gains);
}

}  // namespace jax_sc_embedding
