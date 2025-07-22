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
#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_RAGGED_TENSOR_INPUT_BATCH_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_RAGGED_TENSOR_INPUT_BATCH_H_
#include <cstdint>
#include <vector>

#include "absl/log/check.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/abstract_input_batch.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "jax_tpu_embedding/sparsecore/lib/core/process_coo_tensors_impl.h"
#include "jax_tpu_embedding/sparsecore/lib/core/sparse_csr_input_stream_impl.h"
#include "jax_tpu_embedding/sparsecore/lib/core/unity_weights_stream_impl.h"

namespace jax_sc_embedding {

// Valency: Dimension along which Embeddings for an input are added up.

// This struct represents row offsets for a fixed valency input batch.
// It calculates the offset for a given row index based on the batch size and
// valency.
// For example, if batch_size = 4 and valency = 2, then:
//   FixedValencyRowOffsets offsets(4, 2);
//   EXPECT_EQ(offsets[0], 0);
//   EXPECT_EQ(offsets[1], 2);
//   EXPECT_EQ(offsets[2], 4);
//   EXPECT_EQ(offsets[3], 6);
//   EXPECT_EQ(offsets[4], 8);
class FixedValencyRowOffsets {
 public:
  FixedValencyRowOffsets(int batch_size, int valency)
      : batch_size_(batch_size), valency_(valency) {
    CHECK_GT(batch_size, 0);
    CHECK_GT(valency, 0);
  }
  int64_t operator[](int64_t index) const { return index * valency_; }
  int64_t size() const { return batch_size_ + 1; }

 private:
  int batch_size_;
  int valency_;
};

// EmbeddingIdsView and RowOffsetsView are template parameters that represent a
// view into the underlying data (or even the actual data itself):
//   - EmbeddingIdsView is required to support `operator[]`.
//   - RowOffsetsView is required to support `operator[]` and `size() const`.
// This allows the class to be used with different types of data sources, such
// as vectors, arrays, or other data structures.
template <typename EmbeddingIdsView, typename RowOffsetsView>
class RaggedTensorInputBatch : public AbstractInputBatch {
 public:
  // This class represents a batch of input data encoded using row offsets,
  // similar to how RaggedTensor uses row offsets as described in
  // https://www.tensorflow.org/guide/ragged_tensor#tfraggedtensorfrom_row_splits.
  RaggedTensorInputBatch(int batch_number, EmbeddingIdsView embedding_ids,
                         RowOffsetsView row_offsets)
      : batch_number_(batch_number),
        embedding_ids_(embedding_ids),
        row_offsets_(row_offsets) {}

  int64_t size() const override { return row_offsets_.size() - 1; }
  void ExtractCooTensors(const ExtractCooTensorsOptions& options,
                         std::vector<CooFormat>& coo_tensors) override {
    SparseCsrInputBatchStream<int64_t, EmbeddingIdsView, RowOffsetsView>
        values_stream(embedding_ids_, row_offsets_, options.slice_start,
                      options.slice_end);
    UnityWeightsStream weights_stream(values_stream);

    ProcessCooTensors(options, values_stream, weights_stream, coo_tensors);
  }

  virtual int batch_number() const override { return batch_number_; }

 private:
  int batch_number_;
  EmbeddingIdsView embedding_ids_;
  RowOffsetsView row_offsets_;
};

// deduction guide for compiler
template <typename T1, typename T2>
RaggedTensorInputBatch(T1, T2) -> RaggedTensorInputBatch<T1, T2>;
}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_RAGGED_TENSOR_INPUT_BATCH_H_
