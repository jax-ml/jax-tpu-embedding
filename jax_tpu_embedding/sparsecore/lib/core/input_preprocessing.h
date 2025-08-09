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
#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_INPUT_PREPROCESSING_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_INPUT_PREPROCESSING_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/abstract_input_batch.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"

namespace jax_sc_embedding {

template <typename T>
using StackedTableMap = absl::flat_hash_map<std::string, T>;

struct SparseDenseMatmulInputStats {
  StackedTableMap<RowVectorXi> max_ids_per_partition;
  StackedTableMap<RowVectorXi> max_unique_ids_per_partition;
  StackedTableMap<RowVectorXi> required_buffer_sizes;
  StackedTableMap<int> dropped_id_count;

  int TotalDroppedIdCount() const {
    int sum = 0;
    for (const auto& [_, v] : dropped_id_count) sum += v;
    return sum;
  }
  // Merge another SparseDenseMatmulInputStats object into the current one.
  void merge(const SparseDenseMatmulInputStats& other);
};

namespace internal {
ExtractedCooTensors ExtractCooTensorsForAllFeaturesPerLocalDevice(
    absl::Span<const StackedTableMetadata> stacked_table_metadata,
    absl::Span<std::unique_ptr<AbstractInputBatch>> input_batches,
    int local_device_id,
    const PreprocessSparseDenseMatmulInputOptions& options);
}  // namespace internal

struct PreprocessSparseDenseMatmulOutput {
  StackedTableMap<MatrixXi> lhs_row_pointers;
  StackedTableMap<MatrixXi> lhs_embedding_ids;
  StackedTableMap<MatrixXi> lhs_sample_ids;
  StackedTableMap<MatrixXf> lhs_gains;
  // TODO: b/428790659 - Populate this value based on computed split.
  int num_minibatches = 1;
  SparseDenseMatmulInputStats stats;
};

PreprocessSparseDenseMatmulOutput PreprocessSparseDenseMatmulInput(
    absl::Span<std::unique_ptr<AbstractInputBatch>> input_batches,
    const absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>&
        stacked_tables,
    const PreprocessSparseDenseMatmulInputOptions& options);

}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_INPUT_PREPROCESSING_H_
