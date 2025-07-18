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
#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_ABSTRACT_INPUT_BATCH_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_ABSTRACT_INPUT_BATCH_H_
#include <sys/types.h>

#include <vector>

#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"

namespace jax_sc_embedding {
// NOTE: Converting input data to a C++ native type can be expensive. Therefore,
// we define a read-only wrapper to abstract the input data.
class AbstractInputBatch {
 public:
  struct ExtractCooTensorsOptions {
    // Start index of the slice to be processed (inclusive).
    int slice_start;
    // End index of the slice to be processed (exclusive).
    int slice_end;
    // Row offset to be added to the sample id.
    int row_offset;
    // Column offset to be added to the embedding id.
    int col_offset;
    // Number of bits to shift the embedding id.
    int col_shift;
    // Number of sparse cores.
    int num_scs;
    // Combiner to be used for the row.
    RowCombiner combiner;
  };

  // Return the batch size or the number of samples in this input batch.
  virtual ssize_t size() const = 0;

  // Return the batch number.
  // The batch number should be a sequential counter that is unique for each
  // batch. It is safe to reset this counter to 0 on restart. The number should
  // be unique to identify the batch for collective operations during
  // mini-batching. The number should be sequential to help limit logging
  // (e.g., LOG_IF(INFO, batch_number_ % 100 == 0)).
  virtual int batch_number() const = 0;

  // Extract COO Tensors.
  virtual void ExtractCooTensors(const ExtractCooTensorsOptions& options,
                                 std::vector<CooFormat>& coo_tensors) = 0;

  virtual ~AbstractInputBatch() = default;
};

}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_ABSTRACT_INPUT_BATCH_H_
