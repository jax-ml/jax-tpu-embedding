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
#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_INPUT_NUMPY_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_INPUT_NUMPY_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "pybind11/numpy.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11

namespace jax_sc_embedding {

namespace py = ::pybind11;

// This struct will be eventually moved to generic input abstract base class

struct ExtractedCooTensors {
  std::vector<CooFormat> coo_tensors;
  int batch_size_for_device = 0;
};

// These functions will be eventually moved to a NumpyArrayInput class

absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>
GetStackedTableMetadata(py::list& feature_specs, py::list& features);

ExtractedCooTensors ExtractCooTensorsForAllFeatures(
    absl::Span<const StackedTableMetadata> stacked_table_metadata,
    py::list& features, py::list& feature_weights, int local_device_id,
    int local_device_count, int num_scs, int num_global_devices);

}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_INPUT_NUMPY_H_
