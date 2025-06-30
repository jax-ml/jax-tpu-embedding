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
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/abstract_input_batch.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"

namespace jax_sc_embedding {

struct PreprocessSparseDenseMatmulInputOptions {
  int local_device_count;
  int global_device_count;
  int num_sc_per_device;
  int sharding_strategy = 1;
  bool allow_id_dropping = true;

  int GetNumScs() const { return num_sc_per_device * global_device_count; }
};

template <typename T>
using StackedTableMap = absl::flat_hash_map<std::string, T>;

struct SparseDenseMatmulInputStats {
  StackedTableMap<RowVectorXi> max_ids_per_partition;
  StackedTableMap<RowVectorXi> max_unique_ids_per_partition;
  StackedTableMap<RowVectorXi> required_buffer_sizes;
};

struct ExtractedCooTensors {
  std::vector<CooFormat> coo_tensors;
  int batch_size_for_device = 0;
};

struct PreprocessSparseDenseMatmulOutput {
  StackedTableMap<MatrixXi> lhs_row_pointers;
  StackedTableMap<MatrixXi> lhs_embedding_ids;
  StackedTableMap<MatrixXi> lhs_sample_ids;
  StackedTableMap<MatrixXf> lhs_gains;
  SparseDenseMatmulInputStats stats;
};

PreprocessSparseDenseMatmulOutput PreprocessSparseDenseMatmulInput(
    absl::Span<std::unique_ptr<AbstractInputBatch>> input_batches,
    const absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>&
        stacked_tables,
    const PreprocessSparseDenseMatmulInputOptions& options);

// Template instantiation and linkage require us to define the following in this
// file (Template definition not visible at point of instantiation).

template <typename WeightsStreamT>
float ComputeWeightDivisor(RowCombiner combiner,
                           WeightsStreamT& weights_stream) {
  switch (combiner) {
    case RowCombiner::kSum:
      return 1.0f;
    case RowCombiner::kMean: {
      // Sum of elements.
      float sum = 0.0f;
      for (; weights_stream.col() < weights_stream.cols();
           weights_stream.next_col()) {
        sum += weights_stream.get();
      }
      return sum;
    }
    case RowCombiner::kSqrtn: {
      // Sqrt of sum of squares.
      float sum = 0.0f;
      for (; weights_stream.col() < weights_stream.cols();
           weights_stream.next_col()) {
        sum += std::pow(weights_stream.get(), 2);
      }
      return std::sqrt(sum);
    }
  }
}

template <typename ValuesStreamT, typename WeightsStreamT>
void ProcessCooTensors(int start_index, int end_index, int row_offset,
                       int col_offset, int col_shift, int num_scs,
                       int global_device_count, RowCombiner combiner,
                       ValuesStreamT& values_stream,
                       WeightsStreamT& weights_stream,
                       std::vector<CooFormat>& coo_tensors) {
  CHECK(num_scs > 0 && (num_scs & (num_scs - 1)) == 0);
  DCHECK_GT(global_device_count, 0);
  const int num_scs_bit = std::log2(num_scs);
  const int num_scs_mod = (1 << num_scs_bit) - 1;
  const int num_scs_mod_inv = ~num_scs_mod;

  coo_tensors.reserve(values_stream.size());

  const int row_offset_per_device = row_offset / global_device_count;

  DCHECK_EQ(values_stream.size(), weights_stream.size());

  for (; values_stream.row() < end_index && weights_stream.row() < end_index;
       values_stream.next_row(), weights_stream.next_row()) {
    DCHECK_EQ(values_stream.cols(), weights_stream.cols());
    DCHECK_EQ(values_stream.row(), weights_stream.row());
    DCHECK_EQ(values_stream.col(), weights_stream.col());
    DCHECK_EQ(values_stream.col(), 0);

    const int sample_id =
        values_stream.row() - start_index + row_offset_per_device;
    const float divisor = ComputeWeightDivisor(combiner, weights_stream);

    for (weights_stream.seek_col(0); values_stream.col() < values_stream.cols();
         values_stream.next_col(), weights_stream.next_col()) {
      const int embedding_id = values_stream.get();
      const float gain = weights_stream.get() / divisor;
      DCHECK_GE(embedding_id, 0);

      coo_tensors.emplace_back(sample_id,
                               GetColId(embedding_id, col_shift, col_offset,
                                        num_scs_mod, num_scs_mod_inv),
                               gain);
    }
  }
}

}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_INPUT_PREPROCESSING_H_
