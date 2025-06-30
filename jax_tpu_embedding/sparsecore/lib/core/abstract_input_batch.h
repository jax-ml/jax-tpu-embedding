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
#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_FEATURE_INPUT_WRAPPER_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_FEATURE_INPUT_WRAPPER_H_
#include <sys/types.h>

#include <cmath>
#include <vector>

#include "absl/log/check.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"

namespace jax_sc_embedding {
// NOTE: Converting input data to a C++ native type can be expensive. Therefore,
// we define a read-only wrapper to abstract the input data.
class AbstractInputBatch {
 public:
  // Return the batch size or the number of samples in this input batch.
  virtual ssize_t size() const = 0;

  // Extract COO Tensors between given row indexes (slice) accumulating them in
  // `coo_tensors`.
  virtual void ExtractCooTensors(int slice_start, int slice_end, int row_offset,
                                 int col_offset, int col_shift, int num_scs,
                                 int global_device_count, RowCombiner combiner,
                                 std::vector<CooFormat>& coo_tensors) = 0;

  virtual ~AbstractInputBatch() = default;
};

/**
 * Static interface for iterating over batches of embedding IDs.
 *
 * AbstractInputBatchStream uses the Curiously Recurring Template Pattern (CRTP)
 * to provide a compile-time interface for various stream implementations.
 * It defines a standard set of operations for navigating a 2D-like data
 * structure of samples (rows) and embedding IDs (columns).
 *
 * This pattern avoids the overhead of virtual functions by resolving method
 * calls at compile time.
 *
 * T: The data type of the embedding IDs (e.g., int, long long).
 * Derived: The concrete stream implementation class that inherits from
 * this one.
 */
template <typename T, typename Derived>
class AbstractInputBatchStream {
 public:
  // Total number of embedding IDs (could be a lower bound).
  int size() const { return derived()->size(); }
  // Number of embedding IDs in the current sample/example.
  int cols() const { return derived()->cols(); }
  // Proceed to the next sample.
  void next_row() { derived()->next_row(); }
  // Proceed to the next embedding ID.
  void next_col() { derived()->next_col(); }
  // Seek to the specified column.
  void seek_col(int col) { derived()->seek_col(col); }
  // Get the sample/example index.
  int row() const { return derived()->row(); }
  // Get the embedding ID index.
  int col() const { return derived()->col(); }
  // Get the embedding ID.
  T get() const { return derived()->get(); }
  // for compile time checks.
  using value_type = T;

 private:
  Derived* derived() { return static_cast<Derived*>(this); }
  const Derived* derived() const { return static_cast<const Derived*>(this); }
};

// Computes the weight divisor based on the specified row combiner.
//
// This function calculates the divisor used for normalizing embedding weights
// according to the given `combiner` method (sum, mean, or sqrtn). It iterates
// through the provided `weights_stream` to compute the necessary sum or
// sum-of-squares for normalization.
//
// Args:
//   combiner: The method used to combine embedding vectors (e.g., sum, mean,
//     sqrtn).
//   weights_stream: An input stream of embedding weights.
//
// Returns:
//   The computed weight divisor.
template <typename WeightsStreamT>
float ComputeWeightDivisor(RowCombiner combiner,
                           WeightsStreamT& weights_stream) {
  static_assert(
      std::is_base_of_v<AbstractInputBatchStream<float, WeightsStreamT>,
                        WeightsStreamT>,
      "WeightsStreamT must be a derivative of AbstractInputBatchStream.");
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

// Processes input streams of embedding values and weights to generate COO
// tensors.
//
// This function iterates through the provided `values_stream` and
// `weights_stream`, extracting embedding IDs and their corresponding weights.
// It applies a specified `combiner` to normalize weights and constructs
// `CooFormat` objects, which are then added to the `coo_tensors` vector.
//
// Args:
//   start_index: The starting index of the current slice/batch.
//   end_index: The end index of the current slice/batch.
//   row_offset: The offset to apply to the row index.
//   col_offset: The offset to apply to the column index.
//   col_shift: The shift to apply to the column index.
//   num_scs: The number of SparseCore devices.
//   global_device_count: The total number of devices across all hosts.
//   combiner: The method used to combine embedding vectors (e.g., sum, mean,
//     sqrtn).
//   values_stream: An input stream of embedding IDs.
//   weights_stream: An input stream of embedding weights.
//   coo_tensors: A vector to store the generated `CooFormat` objects.
template <typename ValuesStreamT, typename WeightsStreamT>
void ProcessCooTensors(int start_index, int end_index, int row_offset,
                       int col_offset, int col_shift, int num_scs,
                       int global_device_count, RowCombiner combiner,
                       ValuesStreamT& values_stream,
                       WeightsStreamT& weights_stream,
                       std::vector<CooFormat>& coo_tensors) {
  using T = typename ValuesStreamT::value_type;

  static_assert(
      std::is_base_of_v<AbstractInputBatchStream<T, ValuesStreamT>,
                        ValuesStreamT>,
      "ValuesStreamT must be a derivative of AbstractInputBatchStream.");

  static_assert(
      std::is_base_of_v<AbstractInputBatchStream<float, WeightsStreamT>,
                        WeightsStreamT>,
      "WeightsStreamT must be a derivative of AbstractInputBatchStream.");

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

    // The weights stream is iterated twice, once for compute weight divisor,
    // and second time for accumulating COO tensors. Hence, we manually seek to
    // col 0.
    for (weights_stream.seek_col(0); values_stream.col() < values_stream.cols();
         values_stream.next_col(), weights_stream.next_col()) {
      const T embedding_id = values_stream.get();
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

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_FEATURE_INPUT_WRAPPER_H_
