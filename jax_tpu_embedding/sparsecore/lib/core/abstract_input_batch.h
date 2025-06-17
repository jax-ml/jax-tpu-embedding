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

#include <vector>

#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"

namespace jax_sc_embedding {
// NOTE: Converting input data to a C++ native type can be expensive. Therefore,
// we define a read-only wrapper to abstract the input data.
class AbstractInputBatch {
 public:
  // Return the batch size or the number of samples in this input batch.
  virtual ssize_t size() const = 0;

  // Extract COO Tensors between given row indexes (slice). The flow should
  // mostly be similar and we could make this thinner, operating on lower level
  // abstraction for inputs. coo_tensors accumulates the extracted tensors.
  virtual void ExtractCooTensors(int slice_start, int slice_end, int row_offset,
                                 int col_offset, int col_shift, int num_scs,
                                 int global_device_count, RowCombiner combiner,
                                 std::vector<CooFormat>& coo_tensors) = 0;

  virtual ~AbstractInputBatch() = default;
};

}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_FEATURE_INPUT_WRAPPER_H_
