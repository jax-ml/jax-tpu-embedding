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
#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_INPUT_PREPROCESSING_UTIL_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_INPUT_PREPROCESSING_UTIL_H_

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace jax_sc_embedding {

// Different combiners that are supported for samples with multiple ids.
// By default, we use kSum (add the embeddings for all ids in the sample).
enum class RowCombiner {
  kSum = 0,
  kMean = 1,
  kSqrtn = 2,
};

RowCombiner GetRowCombiner(absl::string_view combiner);

struct CooFormat {
  CooFormat(int row_id, int col_id, float gain)
      : row_id(row_id), col_id(col_id), gain(gain) {}
  int row_id;
  int col_id;
  float gain;

  bool operator==(const CooFormat& other) const {
    return row_id == other.row_id && col_id == other.col_id &&
           gain == other.gain;
  }
};

// Get adjusted col_id based on shift and offset.
int GetColId(int col_id, int col_shift, int col_offset, int num_scs_mod,
             int num_scs_mod_inv);

inline unsigned int CeilOfRatio(unsigned int numerator,
                                unsigned int denominator) {
  // Note: Unsigned values allow better compiler optimizations.  This precise
  // form is used because it cannot overflow.
  return numerator == 0 ? 0 : (numerator - 1) / denominator + 1;
}

// TODO(b/357664103): Converge towards a more compatible interface between the
// python representation and the c++ representation.
struct StackedTableMetadata {
  StackedTableMetadata() = delete;
  StackedTableMetadata(int feature_index, int max_ids_per_partition,
                       int max_unique_ids_per_partition, int row_offset,
                       int col_offset, int col_shift, int batch_size,
                       RowCombiner row_combiner = RowCombiner::kSum,
                       int max_col_id = std::numeric_limits<int>::max())
      : feature_index(feature_index),
        max_ids_per_partition(max_ids_per_partition),
        max_unique_ids_per_partition(max_unique_ids_per_partition),
        row_offset(row_offset),
        col_offset(col_offset),
        col_shift(col_shift),
        batch_size(batch_size),
        row_combiner(row_combiner),
        max_col_id(max_col_id) {}
  // The batch is given as a list of features (numpy arrays). `feature_index`
  // represents the index of the feature in the list.
  int feature_index;

  int max_ids_per_partition;
  int max_unique_ids_per_partition;
  int row_offset;
  int col_offset;
  int col_shift;

  // Process local batch size of the feature.
  int batch_size;

  RowCombiner row_combiner;

  // The vocabulary size of the table. Any embedding IDs that are larger than
  // this value are considered invalid.
  int max_col_id;
};

void SortAndGroupCooTensors(
    absl::Span<const CooFormat> coo_tensors, int batch_size_per_sc,
    int num_scs,  // Number of total sparsecores, across all devices.
    int32_t batch_size_for_device,  // Batch size for the local device.
    int32_t max_ids_per_partition, int32_t max_unique_ids_per_partition,
    absl::string_view stacked_table_name, bool allow_id_dropping,
    std::vector<std::vector<CooFormat>>& coo_tensors_by_id,
    int* aggregated_max_ids_per_sc, int* aggregated_max_unique_ids_per_sc);

int ComputeCooBufferSize(
    int num_scs, int num_scs_per_device,
    absl::Span<const StackedTableMetadata> stacked_table_metadata,
    int static_buffer_size_multiplier);

void IncrementScId(std::pair<int, int>& sc_id, int num_scs,
                   int num_scs_per_device);

int MaxIdsPerPartitionForStackedTables(
    absl::Span<const StackedTableMetadata> stacked_table_metadata);

void FillRowPointers(absl::Span<const std::vector<CooFormat>> coo_tensors_by_id,
                     int row_pointers_size_per_sc, int coo_buffer_size_per_sc,
                     int batch_size_per_sc, int num_scs, int num_sc_per_device,
                     int* row_pointers, int* embedding_ids, int* sample_ids,
                     float* gains);

}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_INPUT_PREPROCESSING_UTIL_H_
