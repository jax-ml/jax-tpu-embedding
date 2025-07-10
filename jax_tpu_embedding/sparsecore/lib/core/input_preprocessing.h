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
#include <cstdint>
#include <limits>
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
           weights_stream.NextCol()) {
        sum += weights_stream.get();
      }
      return sum;
    }
    case RowCombiner::kSqrtn: {
      // Sqrt of sum of squares.
      float sum = 0.0f;
      for (; weights_stream.col() < weights_stream.cols();
           weights_stream.NextCol()) {
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
       values_stream.NextRow(), weights_stream.NextRow()) {
    DCHECK_EQ(values_stream.cols(), weights_stream.cols());
    DCHECK_EQ(values_stream.row(), weights_stream.row());
    DCHECK_EQ(values_stream.col(), weights_stream.col());
    DCHECK_EQ(values_stream.col(), 0);

    const int sample_id =
        values_stream.row() - start_index + row_offset_per_device;
    const float divisor = ComputeWeightDivisor(combiner, weights_stream);

    for (weights_stream.SeekCol(0); values_stream.col() < values_stream.cols();
         values_stream.NextCol(), weights_stream.NextCol()) {
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

// Class to iterate over a sparse CSR array.
// Example:
//   values = [1, 2, 3, 4, 5, 6]
//   row_pointers = [0, 2, 5, 6]
//   This represents a sparse matrix with 3 rows:
//     Row 0: [1, 2]
//     Row 1: [3, 4, 5]
//     Row 2: [6]
template <typename T, typename ValuesView, typename RowPointersView>
class SparseCsrInputBatchStream {
 public:
  SparseCsrInputBatchStream(ValuesView values, RowPointersView row_pointers,
                            int row_start, int row_end,
                            T max_vocab_id = std::numeric_limits<T>::max())
      : values_ref_(values),
        row_pointers_(row_pointers),
        curr_row_(row_start),
        row_end_(row_end),
        curr_idx_(row_pointers[row_start]),
        max_vocab_id_(max_vocab_id) {}

  int size() const { return row_pointers_[row_end_] - row_pointers_[0]; }

  int cols() const {
    return row_pointers_[curr_row_ + 1] - row_pointers_[curr_row_];
  }

  void NextRow() {
    ++curr_row_;
    if (curr_row_ < row_end_) {
      curr_idx_ = row_pointers_[curr_row_];
    }
  }

  void NextCol() { ++curr_idx_; }

  void SeekCol(int col) { curr_idx_ = row_pointers_[curr_row_] + col; }

  int row() const { return curr_row_; }

  int col() const { return curr_idx_ - row_pointers_[curr_row_]; }

  T get() const {
    DCHECK_LT(curr_idx_, row_pointers_[curr_row_ + 1]);
    T embedding_id = values_ref_[curr_idx_];
    CHECK(embedding_id >= 0 && embedding_id <= max_vocab_id_)
        << "Invalid vocabulary id: " << embedding_id
        << " for table vocabulary size: " << max_vocab_id_;
    return embedding_id;
  }

 private:
  ValuesView values_ref_;
  RowPointersView row_pointers_;
  int curr_row_;
  int row_end_;
  int curr_idx_;
  T max_vocab_id_;
};

template <typename T, typename U1, typename U2>
SparseCsrInputBatchStream(U1, U2, int, int, T)
    -> SparseCsrInputBatchStream<T, U1, U2>;

template <typename U1, typename U2>
SparseCsrInputBatchStream(U1, U2, int, int)
    -> SparseCsrInputBatchStream<int64_t, U1, U2>;

// Class to iterate over a sparse CSR array, providing unity weights for each
// value. This class takes an existing `ValuesStream` (e.g.,
// `SparseCsrInputBatchStream`) and provides an interface to iterate over the
// same structure, but returning a weight of 1.0 for each value instead of the
// actual value. This is useful when the input does not have associated weights
// but the processing logic expects a weight stream.
template <typename ValuesStream>
class UnityWeightsStream {
 public:
  UnityWeightsStream(const ValuesStream& value_stream)
      : value_stream_(value_stream), curr_col_(0) {}

  int size() const { return value_stream_.size(); }

  int cols() const { return value_stream_.cols(); }

  void NextRow() { curr_col_ = 0; }

  void NextCol() { ++curr_col_; }

  void SeekCol(int col) { curr_col_ = col; }

  int row() const { return value_stream_.row(); }

  int col() const { return curr_col_; }

  float get() const { return 1.0f; }

 private:
  const ValuesStream& value_stream_;
  int curr_col_;
};
template <typename T>
UnityWeightsStream(T) -> UnityWeightsStream<T>;

}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_INPUT_PREPROCESSING_H_
