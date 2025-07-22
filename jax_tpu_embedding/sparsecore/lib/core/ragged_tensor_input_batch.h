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

#include "jax_tpu_embedding/sparsecore/lib/core/abstract_input_batch.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "jax_tpu_embedding/sparsecore/lib/core/process_coo_tensors_impl.h"
#include "jax_tpu_embedding/sparsecore/lib/core/sparse_csr_input_stream_impl.h"
#include "jax_tpu_embedding/sparsecore/lib/core/unity_weights_stream_impl.h"

namespace jax_sc_embedding {

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
