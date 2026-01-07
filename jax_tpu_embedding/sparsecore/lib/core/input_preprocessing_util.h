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
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "Eigen/Core"  // from @eigen_archive
#include "jax_tpu_embedding/sparsecore/lib/core/all_reduce_interface.h"
#include "jax_tpu_embedding/sparsecore/lib/core/coo_format.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_threads.h"
#include "jax_tpu_embedding/sparsecore/lib/core/partitioned_coo_tensors.h"

namespace jax_sc_embedding {

// TPU_VECTOR_REGISTER_ALIGNMENT_SIZE represents the required alignment for data
// loaded into TPU vector registers, which are typically 8 sublanes x 128 lanes.
// Data dimensions, specially the second most minor, must be padded to be
// multiples of this value to ensure efficient TPU processing and avoid memory
// inefficiency. This alignment is enforced by XLA. This applies to most current
// generations of TPUs (v2, v3, v4, v5, v6).
inline constexpr int TPU_VECTOR_REGISTER_ALIGNMENT_SIZE = 8;

// numpy uses row major order, while eigen defaults to column major.
template <typename T>
using MatrixX =
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

using MatrixXi = MatrixX<int>;
using MatrixXf = MatrixX<float>;
// pybind11 converts this to a 1D numpy array when returning the value.
template <typename T>
using RowVectorX = Eigen::Matrix<T, 1, Eigen::Dynamic, Eigen::RowMajor>;

using RowVectorXi = RowVectorX<int>;
using RowVectorXf = RowVectorX<float>;

template <typename T>
using BlockRow = Eigen::Block<MatrixX<T>, 1, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using StackedTableMap = absl::flat_hash_map<std::string, T>;

// Container for output CSR arrays for multiple stacked tables.
// Allows pre-allocated buffers to be passed in, avoiding data copies.
struct OutputCsrArrays {
  StackedTableMap<Eigen::Map<MatrixXi>> lhs_row_pointers;
  StackedTableMap<Eigen::Map<MatrixXi>> lhs_embedding_ids;
  StackedTableMap<Eigen::Map<MatrixXi>> lhs_sample_ids;
  StackedTableMap<Eigen::Map<MatrixXf>> lhs_gains;
};

// Different combiners that are supported for samples with multiple ids.
// By default, we use kSum (add the embeddings for all ids in the sample).
enum class RowCombiner {
  kSum = 0,
  kMean = 1,
  kSqrtn = 2,
};

RowCombiner GetRowCombiner(absl::string_view combiner);

struct IdPair {
  int32_t col_id;
  int32_t row_id;
};

struct ExtractedCooTensors {
  std::vector<IdPair> ids;
  // For storing gains, we use different strategies to optimize for memory and
  // performance:
  // 1. If weights are non-uniform (`has_variable_weights_ == true`), we store
  //    per-tensor gains in the `gains` vector.
  // 2. If weights are uniform (`has_variable_weights_ == false`):
  //    a. For 'sum' combiner: Gains are always 1.0, so we don't store them and
  //       instead rematerialize during grouping.
  //    b. For 'mean' combiner: We store token counts per row in
  //       `row_token_counts` and compute gains (1.0 / token_count) during
  //       grouping. This is ~2.5% faster than storing pre-computed gains per
  //       row.
  //    c. For 'sqrtn' combiner: We pre-calculate and store gains per row in
  //       `row_gains`, because sqrt() is more expensive to compute on-the-fly
  //       during grouping compared to division for 'mean'.
  // This approach minimizes memory footprint for uniform weights by storing
  // data per-row for 'mean' and 'sqrtn', rather than per-tensor. (We only
  // perform 1 push_back per row, rather than per tensor.)
  std::vector<float> gains;
  // Number of samples these coo_tensors are extracted from.
  int batch_size_for_device;
  std::vector<int> row_token_counts;
  std::vector<float> row_gains;
  // Count coo tensors per SC for efficient allocation of vector for sorting and
  // grouping them. Might be lower after deduplication.
  std::vector<int> coo_tensors_per_sc;
  bool has_variable_weights_;

  ExtractedCooTensors() : ExtractedCooTensors(0, 0) {}
  ExtractedCooTensors(int num_sc_per_device, int batch_size_for_device,
                      bool has_variable_weights = true,
                      RowCombiner combiner = RowCombiner::kSum)
      : batch_size_for_device(batch_size_for_device),
        coo_tensors_per_sc(num_sc_per_device, 0),
        has_variable_weights_(has_variable_weights) {
    if (!has_variable_weights_) {
      switch (combiner) {
        case RowCombiner::kMean:
          row_token_counts.resize(batch_size_for_device);
          break;
        case RowCombiner::kSqrtn:
          row_gains.resize(batch_size_for_device);
          break;
        case RowCombiner::kSum:
          break;
      }
    }
  }

  // Test only constructor.
  ExtractedCooTensors(int num_sc_per_device, int batch_size_for_device,
                      absl::Span<const CooFormat> coos)
      : batch_size_for_device(batch_size_for_device),
        coo_tensors_per_sc(num_sc_per_device, 0),
        has_variable_weights_(false) {
    ids.reserve(coos.size());

    // Check if any of the gains are not 1.0.
    for (const auto& coo : coos) {
      if (coo.gain != 1.0f) {
        has_variable_weights_ = true;
        break;
      }
    }

    // If we have variable weights, we need to store the gains.
    if (has_variable_weights_) {
      gains.reserve(coos.size());
    }

    // Add all the coos to the extracted coo tensors.
    for (const auto& coo : coos) {
      ids.push_back({.col_id = coo.col_id, .row_id = coo.row_id});
      if (has_variable_weights_) {
        gains.push_back(coo.gain);
      }
    }

    // Populate the coo_tensors_per_sc vector.
    DCHECK_GT(num_sc_per_device, 0);
    DCHECK_EQ(batch_size_for_device % num_sc_per_device, 0);
    const int batch_size_per_sc = batch_size_for_device / num_sc_per_device;
    for (const auto& id_pair : ids) {
      coo_tensors_per_sc[id_pair.row_id / batch_size_per_sc]++;
    }
  }

  bool has_variable_weights() const { return has_variable_weights_; }

  void emplace_back(int row_id, int embedding_id, float gain, int col_shift,
                    int col_offset, int num_scs_mod) {
    this->ids.push_back({.col_id = CooFormat::GetColId(embedding_id, col_shift,
                                                       col_offset, num_scs_mod),
                         .row_id = row_id});
    if (has_variable_weights_) {
      this->gains.push_back(gain);
    }
  }

  size_t size() const { return ids.size(); }

  void reserve(size_t n) {
    ids.reserve(n);
    if (has_variable_weights_) {
      gains.reserve(n);
    }
  }

  CooFormat get(int i) const {
    return CooFormat(ids[i].row_id, ids[i].col_id,
                     has_variable_weights_ ? gains[i] : 1.0f);
  }

  template <bool kHasVariableWeights>
  CooFormat GetCooFormatWithGain(uint64_t key, uint32_t col_id,
                                 RowCombiner combiner) const {
    float gain = 1.0f;
    uint32_t row_id;
    if constexpr (kHasVariableWeights) {
      const uint32_t index = CooFormat::GetDataFromKey(key);
      row_id = ids[index].row_id;
      gain = gains[index];
    } else {
      row_id = CooFormat::GetDataFromKey(key);
      switch (combiner) {
        case RowCombiner::kMean:
          gain = 1.0f / static_cast<float>(row_token_counts[row_id]);
          break;
        case RowCombiner::kSum:
          gain = 1.0f;
          break;
        case RowCombiner::kSqrtn:
          gain = row_gains[row_id];
          break;
      }
    }
    return CooFormat(row_id, col_id, gain);
  }

  // For test only.
  std::vector<CooFormat> ToCooVector() const {
    std::vector<CooFormat> coos;
    coos.reserve(size());
    for (int i = 0; i < size(); ++i) {
      coos.push_back(get(i));
    }
    return coos;
  }
};

struct DeviceSortingTaskResult {
  DevicePartitionedCooTensors grouped_coo_tensors;
  int total_dropped_id_count = 0;
};

namespace internal {

struct CsrArraysPerDevice {
  Eigen::Ref<RowVectorXi> row_pointers;
  Eigen::Ref<RowVectorXi> embedding_ids;
  Eigen::Ref<RowVectorXi> sample_ids;
  Eigen::Ref<RowVectorXf> gains;
};

struct StatsPerDevice {
  BlockRow<int> max_ids_per_partition;
  BlockRow<int> max_unique_ids_per_partition;
  BlockRow<int> required_buffer_size;
  int dropped_id_count;
};

}  // namespace internal

struct CsrArraysPerHost {
  Eigen::Map<MatrixXi> row_pointers;
  Eigen::Map<MatrixXi> embedding_ids;
  Eigen::Map<MatrixXi> sample_ids;
  Eigen::Map<MatrixXf> gains;

  CsrArraysPerHost(Eigen::Ref<MatrixXi> row_pointers,
                   Eigen::Ref<MatrixXi> embedding_ids,
                   Eigen::Ref<MatrixXi> sample_ids, Eigen::Ref<MatrixXf> gains)
      : row_pointers(row_pointers.data(), row_pointers.rows(),
                     row_pointers.cols()),
        embedding_ids(embedding_ids.data(), embedding_ids.rows(),
                      embedding_ids.cols()),
        sample_ids(sample_ids.data(), sample_ids.rows(), sample_ids.cols()),
        gains(gains.data(), gains.rows(), gains.cols()) {}

  internal::CsrArraysPerDevice GetCsrArraysPerDevice(int local_device_id)
      ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return internal::CsrArraysPerDevice{
        .row_pointers = row_pointers.row(local_device_id),
        .embedding_ids = embedding_ids.row(local_device_id),
        .sample_ids = sample_ids.row(local_device_id),
        .gains = gains.row(local_device_id),
    };
  }
};

struct StatsPerHost {
  // NOTE: max ids and max unique ids are {global_sc_count *
  //   num_devices}, where they are then aggregated(max) along device
  //   dimension to get {global_sc_count} (i.e. max [unique] ids for each
  //   sc), which can be further aggregated(max) for a single value for
  //   all SCs.
  // NOTE: required buffer size is {local_sc_count * num_devices}, which
  //   is same as {global_sc_count}, and can be further aggregated to get
  //   the maximum size of any SC buffer shard.
  MatrixXi max_ids_per_partition;
  MatrixXi max_unique_ids_per_partition;
  MatrixXi required_buffer_size;
  int dropped_id_count;

  StatsPerHost(int local_device_count, int global_sc_count,
               int num_sc_per_device)
      : max_ids_per_partition(local_device_count, global_sc_count),
        max_unique_ids_per_partition(local_device_count, global_sc_count),
        required_buffer_size(local_device_count, num_sc_per_device),
        dropped_id_count(0) {
    max_ids_per_partition.setZero();
    max_unique_ids_per_partition.setZero();
    required_buffer_size.setZero();
  }

  void Merge(const StatsPerHost& other) {
    max_ids_per_partition =
        max_ids_per_partition.cwiseMax(other.max_ids_per_partition);
    max_unique_ids_per_partition = max_unique_ids_per_partition.cwiseMax(
        other.max_unique_ids_per_partition);
    required_buffer_size =
        required_buffer_size.cwiseMax(other.required_buffer_size);
    dropped_id_count += other.dropped_id_count;
  }

  void Flatten() {
    max_ids_per_partition.resize(1, max_ids_per_partition.size());
    max_unique_ids_per_partition.resize(1, max_unique_ids_per_partition.size());
    required_buffer_size.resize(1, required_buffer_size.size());
  }

  internal::StatsPerDevice GetStatsPerDevice(int local_device_id)
      ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return internal::StatsPerDevice{
        .max_ids_per_partition = max_ids_per_partition.row(local_device_id),
        .max_unique_ids_per_partition =
            max_unique_ids_per_partition.row(local_device_id),
        .required_buffer_size = required_buffer_size.row(local_device_id),
        .dropped_id_count = 0,
    };
  }
};

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

inline void PreprocessingThreadPoolSchedule(std::function<void()> callback) {
  PreprocessingThreadPool()->Schedule(std::move(callback));
}

struct PreprocessSparseDenseMatmulInputOptions {
  // The number of TPU devices on the current host.
  const int local_device_count ABSL_REQUIRE_EXPLICIT_INIT;
  // The total number of TPU devices across all hosts.
  const int global_device_count ABSL_REQUIRE_EXPLICIT_INIT;
  // The number of SparseCores per TPU device.
  const int num_sc_per_device ABSL_REQUIRE_EXPLICIT_INIT;
  // The sharding strategy used to distribute embedding IDs across SparseCores.
  const ShardingStrategy sharding_strategy = ShardingStrategy::kMod;
  // Whether to allow dropping embedding IDs if the buffer size is exceeded.
  const bool allow_id_dropping = true;
  // The strategy used for stacking features from different tables.
  FeatureStackingStrategy feature_stacking_strategy =
      FeatureStackingStrategy::kSplitThenStack;
  // Whether mini-batching is enabled.
  const bool enable_minibatching = false;

  // The batch number should be a sequential counter that is unique for each
  // batch. It is safe to reset this counter to 0 on restart. The number should
  // be unique to identify the batch for collective operations during
  // mini-batching. The number should be sequential to help limit logging
  // (e.g., LOG_IF(INFO, batch_number_ % 100 == 0)).
  const int batch_number = 0;

  // (Non-owning) Interface for performing all-reduce operations, used during
  // mini-batching to synchronize state across different hosts.
  AllReduceInterface* absl_nullable all_reduce_interface;

  // Hash function used for creating minibatching buckets.
  CooFormat::HashFn minibatching_bucketing_hash_fn = HighwayHash;

  // Callback to schedule async work. Since `absl::FunctionRef` is non-owning,
  // any custom callable provided must outlive the
  // `PreprocessSparseDenseMatmulInputOptions` instance.
  absl::FunctionRef<void(std::function<void()>)> async_task_scheduler =
      PreprocessingThreadPoolSchedule;

  // Number of SparseCores to group into a single sorting task.
  // This controls the parallelism of the sorting phase. Defaults to
  // num_sc_per_device.
  // TODO(b/469153631): Figure out heuristic to avoid over-parallelization
  // of small tables (which incur significant overhead).
  int num_sc_per_sorting_task = num_sc_per_device;

  // Returns the total number of SparseCores across all devices and hosts.
  uint32_t GetNumScs() const { return num_sc_per_device * global_device_count; }
};

static_assert(
    std::is_trivially_copyable<PreprocessSparseDenseMatmulInputOptions>());

struct StackedTableMetadata {
  StackedTableMetadata() = delete;
  StackedTableMetadata(
      absl::string_view name, int feature_index, int max_ids_per_partition,
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
  // Global row offset for this feature across all devices.
  int row_offset;
  // Global column offset for this feature across all devices.
  int col_offset;
  int col_shift;

  // Process local batch size of the feature.
  int batch_size;

  RowCombiner row_combiner;

  // The maximum valid physical row index in the stacked table on a SparseCore
  // shard.
  int max_col_id;

  bool operator==(const StackedTableMetadata& other) const = default;
};

int ComputeCooBufferSizePerDevice(
    int num_scs, int num_scs_per_device,
    absl::Span<const StackedTableMetadata> stacked_table_metadata,
    int batch_number = 0, bool use_minibatching = false);

int MaxIdsPerPartitionForStackedTables(
    absl::Span<const StackedTableMetadata> stacked_table_metadata);

std::optional<int> SuggestedCooBufferSizeForStackedTables(
    absl::Span<const StackedTableMetadata> stacked_table_metadata);

void FillLocalDeviceBuffer(
    const DevicePartitionedCooTensors& grouped_coo_tensors,
    int row_pointers_size_per_bucket, int coo_buffer_size_per_sc,
    int batch_size_per_sc,
    const PreprocessSparseDenseMatmulInputOptions& options,
    internal::CsrArraysPerDevice& csr, int& dropped_id_count_static_bound);

}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_INPUT_PREPROCESSING_UTIL_H_
