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

#include <cstdint>

#include "absl/base/attributes.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"

namespace jax_sc_embedding {
// NOTE: Converting input data to a C++ native type can be expensive. Therefore,
// we define a read-only wrapper to abstract the input data.
// Represents a batch of inputs for a single Feature and corresponding weights.
class AbstractInputBatch {
 public:
  struct ExtractCooTensorsOptions {
    // Start index of the slice to be processed (inclusive).
    const int slice_start ABSL_REQUIRE_EXPLICIT_INIT;
    // End index of the slice to be processed (exclusive).
    const int slice_end ABSL_REQUIRE_EXPLICIT_INIT;
    // Row offset to be added to the sample id.
    const int row_offset ABSL_REQUIRE_EXPLICIT_INIT;
    // Column offset to be added to the embedding id.
    const int col_offset ABSL_REQUIRE_EXPLICIT_INIT;
    // Number of bits to shift the embedding id.
    const int col_shift ABSL_REQUIRE_EXPLICIT_INIT;
    // Number of sparse cores per device. Used to compute COO tensor counts per
    // SC.
    const int num_sc_per_device ABSL_REQUIRE_EXPLICIT_INIT;
    // Number of sparse cores.
    const uint32_t num_scs ABSL_REQUIRE_EXPLICIT_INIT;
    // Combiner to be used for the row.
    const RowCombiner combiner ABSL_REQUIRE_EXPLICIT_INIT;
  };

  // Return the batch size or the number of samples in this input batch.
  virtual ssize_t size() const = 0;

  // Extract COO Tensors.
  virtual void ExtractCooTensors(
      const ExtractCooTensorsOptions& options,
      ExtractedCooTensors& extracted_coo_tensors) = 0;

  virtual ~AbstractInputBatch() = default;
};

}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_ABSTRACT_INPUT_BATCH_H_
