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
#include <optional>
#include <string>

#include "absl/base/attributes.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "Eigen/Core"  // from @eigen_archive
#include "jax_tpu_embedding/sparsecore/lib/core/coo_format.h"
#include "jax_tpu_embedding/sparsecore/lib/core/partitioned_coo_tensors.h"
#include "xla/util.h"  // from @xla
#include "tsl/profiler/lib/traceme.h"

namespace jax_sc_embedding {
namespace {

// Options for filling the SparseCore input buffers.
struct BufferFillingOptions {
  // The local SparseCore ID on the current device.
  int local_sc_id ABSL_REQUIRE_EXPLICIT_INIT;
  // The input COO tensors for the current local_sc_id and minibatch.
  absl::Span<const CooFormat> coo_tensors ABSL_REQUIRE_EXPLICIT_INIT;

  // The beginning index (inclusive) in the row_pointers buffer for this
  // segment.
  int lhs_row_begin ABSL_REQUIRE_EXPLICIT_INIT;
  // The ending index (exclusive) in the row_pointers buffer for this segment.
  int lhs_row_end ABSL_REQUIRE_EXPLICIT_INIT;

  // The beginning index (inclusive) in the COO buffers (embedding_ids,
  // sample_ids, gains) for this segment.
  int coo_begin ABSL_REQUIRE_EXPLICIT_INIT;
  // The ending index (exclusive) in the COO buffers for this segment.
  int coo_end ABSL_REQUIRE_EXPLICIT_INIT;

  // The batch size handled by each SparseCore.
  int batch_size_per_sc ABSL_REQUIRE_EXPLICIT_INIT;
  // The number of SparseCores per device.
  int num_sc_per_device ABSL_REQUIRE_EXPLICIT_INIT;
  // The total number of SparseCores across all devices.
  int num_scs ABSL_REQUIRE_EXPLICIT_INIT;
  // The total size of the COO buffer for the current device.
  int coo_buffer_size ABSL_REQUIRE_EXPLICIT_INIT;
  // Whether minibatching is enabled.
  bool enable_minibatching ABSL_REQUIRE_EXPLICIT_INIT;
};

// Check if the current indexes are valid within the buffer sizes.
bool ValidIndices(int row_index, int coo_offset, int processed,
                  const BufferFillingOptions& options) {
  if (row_index >= options.lhs_row_end || coo_offset >= options.coo_end) {
    LOG_EVERY_N(WARNING, 100)
        << "The static buffer size might be too small for the current "
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

// Enum to specify the padding behavior for `PadCooBuffer`.
enum class PadType {
  // Pads to the next HBM alignment boundary.
  kAlignOnly,
  // Pads to the end of the provided buffer segment (`coo_end`).
  kPadToEnd
};

// Pads the COO buffer.
//
// Args:
//   `coo_index`: Current index into the COO buffer.
//   `coo_end`: End index of the COO buffer segment to consider for padding.
//   `pad_type`: Specifies the padding behavior.
//   `csr`: CSR arrays to be padded.
void PadCooBuffer(int& coo_index, int coo_end, PadType pad_type,
                  internal::CsrArraysPerDevice& csr) {
  if (pad_type == PadType::kPadToEnd) {
    coo_index = coo_end;
    return;
  }
  while (coo_index % TPU_VECTOR_REGISTER_ALIGNMENT_SIZE != 0 &&
         coo_index < coo_end) {
    csr.embedding_ids[coo_index] = INT_MAX;
    csr.sample_ids[coo_index] = INT_MAX;
    csr.gains[coo_index] = std::nanf("");
    ++coo_index;
  }
}

// Returns the row pointer value for a given `coo_index`.
// If minibatching is enabled, it points to the global index in the COO buffer.
// Otherwise, it's the `coo_index` relative to the beginning of the current
// SparseCore's COO buffer segment.
int GetRowPointer(int coo_index, const BufferFillingOptions& options) {
  if (options.enable_minibatching) return coo_index;
  return coo_index - options.coo_begin;
}

// Advance last_partition_id to target_partition_id, filling row pointers and
// padding COO buffer for each partition.
void AdvanceAndPadPartitions(int& current_partition_id,
                             const int target_partition_id, int& lhs_row_index,
                             int& coo_index, int processed,
                             const BufferFillingOptions& options,
                             internal::CsrArraysPerDevice& csr_arrays) {
  DCHECK_LE(current_partition_id, target_partition_id);
  while (current_partition_id < target_partition_id) {
    if (!ValidIndices(lhs_row_index, coo_index, processed, options)) {
      return;
    }
    csr_arrays.row_pointers[lhs_row_index++] =
        GetRowPointer(coo_index, options);
    // Align partition.
    PadCooBuffer(coo_index, options.coo_end, PadType::kAlignOnly, csr_arrays);
    ++current_partition_id;
  }
}

// Fill the row pointers buffer from `lhs_row_begin` to `lhs_row_end` and COO
// buffer from `coo_begin` to `coo_end`. Returns the `coo_index` from where next
// the COO buffer can be filled.
int FillBufferSegment(const BufferFillingOptions& options,
                      internal::CsrArraysPerDevice& csr_arrays,
                      int& dropped_id_count_static_bound) {
  int lhs_row_index = options.lhs_row_begin;
  int coo_index = options.coo_begin;
  int current_partition_id = 0;
  int dropped_in_call = 0;

  int processed = 0;
  for (const CooFormat& coo_tensor : options.coo_tensors) {
    DCHECK_EQ(coo_tensor.row_id / options.batch_size_per_sc,
              options.local_sc_id);
    const int target_partition_id = coo_tensor.col_id % options.num_scs;
    AdvanceAndPadPartitions(current_partition_id, target_partition_id,
                            lhs_row_index, coo_index, processed, options,
                            csr_arrays);
    if (!ValidIndices(lhs_row_index, coo_index, processed, options)) {
      dropped_in_call =
          std::max<int>(0, options.coo_tensors.size() - processed);
      dropped_id_count_static_bound += dropped_in_call;
      break;
    }

    csr_arrays.embedding_ids[coo_index] = coo_tensor.col_id / options.num_scs;
    csr_arrays.sample_ids[coo_index] =
        coo_tensor.row_id % options.batch_size_per_sc;
    csr_arrays.gains[coo_index] = coo_tensor.gain;
    ++coo_index;
    ++processed;
  }
  // Fill remaining partitions for this SparseCore.
  AdvanceAndPadPartitions(
      current_partition_id, /*target_partition_id=*/options.num_scs,
      lhs_row_index, coo_index, processed, options, csr_arrays);

  PadRowPointersBuffer(lhs_row_index,
                       /*padding=*/GetRowPointer(coo_index, options),
                       options.lhs_row_end, csr_arrays.row_pointers);

  DCHECK_EQ(dropped_in_call + processed, options.coo_tensors.size());
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

int64_t MayBeUpdateBufferSize(int64_t theoretical_max,
                              int64_t suggested_coo_buffer_size_per_device,
                              int num_scs_per_device,
                              absl::string_view stacked_table_name) {
  // Since the suggested size corresponds to only current device (local SCs),
  // Buffer for each SC should be properly aligned, hence ALIGNMENT *
  // num_scs_per_device
  int64_t suggested_value = xla::RoundUpTo<int64_t>(
      suggested_coo_buffer_size_per_device,
      TPU_VECTOR_REGISTER_ALIGNMENT_SIZE * num_scs_per_device);
  CHECK(suggested_value <= theoretical_max)
      << "Suggested Coo Buffer Size is larger than the theoretical "
         "max for table "
      << stacked_table_name << ": " << suggested_value << " vs "
      << theoretical_max
      << ". Adjust the suggested size or the max_ids_per_partition values.";
  return suggested_value;
}

int ComputeCooBufferSizePerDevice(
    const int num_scs, const int num_scs_per_device,
    absl::Span<const StackedTableMetadata> stacked_table_metadata,
    const int batch_number, bool use_minibatching) {
  const int max_ids_per_partition =
      MaxIdsPerPartitionForStackedTables(stacked_table_metadata);
  const std::optional<int> suggested_coo_buffer_size_per_device =
      SuggestedCooBufferSizeForStackedTables(stacked_table_metadata);

  const int64_t max_ids_rounded_up = xla::RoundUpTo<int64_t>(
      max_ids_per_partition, TPU_VECTOR_REGISTER_ALIGNMENT_SIZE);
  // If minibatching is enabled, `theoretical_max` is multiplied by
  // `kMaxMinibatchingBuckets` because all minibatches for a given SparseCore
  // core are packed into a single buffer.
  const int64_t theoretical_max =
      max_ids_rounded_up * num_scs_per_device * num_scs *
      (use_minibatching ? CooFormat::kMaxMinibatchingBuckets : 1);
  const std::string& stacked_table_name = stacked_table_metadata[0].name;
  VLOG_EVERY_N(2, 10007) << "Theoretical Max for table " << stacked_table_name
                       << ": " << theoretical_max
                       << " (max_ids_rounded_up: " << max_ids_rounded_up
                       << " num_scs_per_device: " << num_scs_per_device
                       << " num_scs: " << num_scs << ")";
  // We do not take the min of `suggested_coo_buffer_size_per_device` and
  // `theoretical_max` because `theoretical_max` is dynamic and depends on
  // `max_ids_per_partition`. Taking the min could cause unexpected changes in
  // buffer sizes if `max_ids_per_partition` is changed by the user. Instead, we
  // throw an error in `MayBeUpdateBufferSize` if
  // `suggested_coo_buffer_size_per_device` is larger than `theoretical_max`.
  int64_t computed_coo_buffer_size_per_device = theoretical_max;
  if (suggested_coo_buffer_size_per_device.has_value()) {
    VLOG_EVERY_N(2, 10007) << "Suggested Coo Buffer Size for table "
                         << stacked_table_name << ": "
                         << suggested_coo_buffer_size_per_device.value();
    computed_coo_buffer_size_per_device = MayBeUpdateBufferSize(
        theoretical_max, suggested_coo_buffer_size_per_device.value(),
        num_scs_per_device, stacked_table_name);
  } else {
    LOG_IF(WARNING, batch_number % 10000 == 0)
        << "No Coo Buffer Size provided for table " << stacked_table_name
        << ", the default value (" << theoretical_max
        << ") may be too "
           "large and can cause OOM. Utilize the stats returned from "
           "the sparse dense matmul preprocessing API and update using "
           "`embedding.update_preprocessing_parameters`. ";
  }
  VLOG_EVERY_N(2, 10007) << "Computed Coo Buffer Size for table "
                       << stacked_table_name << ": "
                       << computed_coo_buffer_size_per_device;
  // The result could be very large and cause overflow. We need to make
  // sure the result is within the range of int before using it.
  CHECK(computed_coo_buffer_size_per_device > 0 &&
        computed_coo_buffer_size_per_device < INT_MAX)
      << "Computed Coo Buffer Size (" << computed_coo_buffer_size_per_device
      << ") for table " << stacked_table_name
      << " is out of the valid range (0, INT_MAX).";
  return static_cast<int>(computed_coo_buffer_size_per_device);
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
    internal::CsrArraysPerDevice& csr_arrays,
    int& dropped_id_count_static_bound) {
  tsl::profiler::TraceMe t("FillLocalDeviceBuffer");
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
      // Fill Minibatch or SparseCore slice.
      coo_begin = FillBufferSegment(
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
              .enable_minibatching = options.enable_minibatching,
          },

          csr_arrays, dropped_id_count_static_bound);
      lhs_row_begin = lhs_row_end;
      if (options.enable_minibatching) {
        // Align minibatch buffer
        PadCooBuffer(coo_begin, coo_buffer_size, PadType::kAlignOnly,
                     csr_arrays);
      }
    }  // end minibatch loop
    if (!options.enable_minibatching) {
      const int sc_end = (local_sc_id + 1) * coo_buffer_size_per_sc;
      // Pad to end of SparseCore buffer.
      PadCooBuffer(coo_begin, sc_end, PadType::kPadToEnd, csr_arrays);
    }
  }  // end SparseCore loop
  // Pad to end of device buffer.
  PadCooBuffer(coo_begin, coo_buffer_size, PadType::kPadToEnd, csr_arrays);
}

}  // namespace jax_sc_embedding
