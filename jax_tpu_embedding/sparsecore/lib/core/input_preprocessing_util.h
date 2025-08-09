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

#include <bitset>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "Eigen/Core"  // from @eigen_archive
#include "jax_tpu_embedding/sparsecore/lib/core/coo_format.h"
#include "jax_tpu_embedding/sparsecore/lib/core/partitioned_coo_tensors.h"

namespace jax_sc_embedding {

// TPU_VECTOR_REGISTER_ALIGMENT_SIZE represents the required alignment for data
// loaded into TPU vector registers, which are typically 8 sublanes x 128 lanes.
// Data dimensions, specially the second most minor, must be padded to be
// multiples of this value to ensure efficient TPU processing and avoid memory
// inefficiency. This alignment is enforced by XLA. This applies to most current
// generations of TPUs (v2, v3, v4, v5, v6).
inline constexpr int TPU_VECTOR_REGISTER_ALIGMENT_SIZE = 8;

// numpy uses row major order, while eigen defaults to column major.
using MatrixXi =
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXf =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
// pybind11 converts this to a 1D numpy array when returning the value.
using RowVectorXi = Eigen::Matrix<int, 1, Eigen::Dynamic, Eigen::RowMajor>;
using RowVectorXf = Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>;

using MinibatchingSplit = std::bitset<CooFormat::kMaxMinibatchingBuckets - 1>;

enum class FeatureStackingStrategy {
  // Stack all features into one large tensor, then split it across SparseCores.
  // Simpler data layout but can cause load imbalance if features have different
  // computational costs.
  kStackThenSplit = 0,
  // Split each feature individually, then stack the corresponding shards on
  // each SparseCore. Generally provides better load balancing, as each
  // SparseCore processes an equal portion of every feature.
  kSplitThenStack = 1
};

enum class ShardingStrategy : int { kMod = 1 };

struct PreprocessSparseDenseMatmulInputOptions {
  const int local_device_count;
  const int global_device_count;
  const int num_sc_per_device;
  const ShardingStrategy sharding_strategy = ShardingStrategy::kMod;
  const bool allow_id_dropping = true;
  FeatureStackingStrategy feature_stacking_strategy =
      FeatureStackingStrategy::kSplitThenStack;
  const bool enable_minibatching = false;

  // The batch number should be a sequential counter that is unique for each
  // batch. It is safe to reset this counter to 0 on restart. The number should
  // be unique to identify the batch for collective operations during
  // mini-batching. The number should be sequential to help limit logging
  // (e.g., LOG_IF(INFO, batch_number_ % 100 == 0)).
  const int batch_number = 0;

  uint32_t GetNumScs() const { return num_sc_per_device * global_device_count; }
};

// Different combiners that are supported for samples with multiple ids.
// By default, we use kSum (add the embeddings for all ids in the sample).
enum class RowCombiner {
  kSum = 0,
  kMean = 1,
  kSqrtn = 2,
};

RowCombiner GetRowCombiner(absl::string_view combiner);

struct ExtractedCooTensors {
  std::vector<CooFormat> coo_tensors;
  // Number of samples these coo_tensors are extracted from.
  int batch_size_for_device;
  // Count coo tensors per SC for efficient allocation of vector for sorting and
  // grouping them. Might be lower after deduplication.
  std::vector<int> coo_tensors_per_sc;

  ExtractedCooTensors(int num_sc_per_device, int batch_size_for_device)
      : batch_size_for_device(batch_size_for_device),
        coo_tensors_per_sc(num_sc_per_device, 0) {}
};

// Rounds up the given value to the next multiple of the given alignment.
// This is equivalent to ceil(value / align) * align, but implemented in an
// integer-safe way.
template <typename T>
static inline T RoundUpTo(T value, T align) {
  return (value + align - 1) / align * align;
};

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
  StackedTableMetadata(
      std::string_view name, int feature_index, int max_ids_per_partition,
      int max_unique_ids_per_partition, int row_offset, int col_offset,
      int col_shift, int batch_size,
      std::optional<int> suggested_coo_buffer_size_per_device = std::nullopt,
      RowCombiner row_combiner = RowCombiner::kSum,
      int max_col_id = std::numeric_limits<int>::max())
      : name(name),
        feature_index(feature_index),
        max_ids_per_partition(max_ids_per_partition),
        max_unique_ids_per_partition(max_unique_ids_per_partition),
        suggested_coo_buffer_size_per_device(
            suggested_coo_buffer_size_per_device),
        row_offset(row_offset),
        col_offset(col_offset),
        col_shift(col_shift),
        batch_size(batch_size),
        row_combiner(row_combiner),
        max_col_id(max_col_id) {}

  std::string name;

  // The batch is given as a list of features (numpy arrays). `feature_index`
  // represents the index of the feature in the list.
  int feature_index;

  int max_ids_per_partition;
  int max_unique_ids_per_partition;
  std::optional<int> suggested_coo_buffer_size_per_device;
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

PartitionedCooTensors SortAndGroupCooTensorsPerLocalDevice(
    const ExtractedCooTensors& extracted_coo_tensors,
    const StackedTableMetadata& stacked_table_metadata,
    const PreprocessSparseDenseMatmulInputOptions& options,
    Eigen::Ref<RowVectorXi> max_ids_per_sc,
    Eigen::Ref<RowVectorXi> max_unique_ids_per_sc,
    Eigen::Ref<RowVectorXi> required_buffer_size_per_sc,
    int& dropped_id_counter, MinibatchingSplit& minibatching_split);

int ComputeCooBufferSizePerDevice(
    int num_scs, int num_scs_per_device,
    absl::Span<const StackedTableMetadata> stacked_table_metadata,
    int batch_number = 0);

void IncrementScId(std::pair<int, int>& sc_id, int num_scs,
                   int num_scs_per_device);

int MaxIdsPerPartitionForStackedTables(
    absl::Span<const StackedTableMetadata> stacked_table_metadata);

std::optional<int> SuggestedCooBufferSizeForStackedTables(
    absl::Span<const StackedTableMetadata> stacked_table_metadata);

void FillLocalDeviceBuffer(
    const PartitionedCooTensors& grouped_coo_tensors,
    int row_pointers_size_per_bucket, int coo_buffer_size_per_sc,
    int batch_size_per_sc,
    const PreprocessSparseDenseMatmulInputOptions& options,
    Eigen::Ref<RowVectorXi> row_pointers, Eigen::Ref<RowVectorXi> embedding_ids,
    Eigen::Ref<RowVectorXi> sample_ids, Eigen::Ref<RowVectorXf> gains);

}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_INPUT_PREPROCESSING_UTIL_H_
