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
#include "jax_tpu_embedding/sparsecore/lib/core/numpy_input_batch.h"

#include <memory>
#include <vector>

#include "absl/log/check.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/abstract_input_batch.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"

namespace jax_sc_embedding {

namespace py = ::pybind11;

AbstractInputBatch* NumpySparseInputBatch::Slice(int start_index,
                                                 int end_index) const {
  DCHECK(!PyGILState_Check());  // Does not require external GIL
  py::gil_scoped_acquire _;
  py::slice slice = py::slice(start_index, end_index, 1);
  return new NumpySparseInputBatch(feature_[slice], weights_[slice]);
}

void NumpySparseInputBatch::ExtractCooTensors(
    int row_offset, int col_offset, int col_shift, int num_scs,
    int global_device_count, RowCombiner combiner,
    std::vector<CooFormat>& coo_tensors) const {
  DCHECK(!PyGILState_Check());  // Does not require external GIL
  py::gil_scoped_acquire _;
  // Temporary: This function will be moved into this file.
  jax_sc_embedding::ExtractCooTensors(
      feature_, weights_, row_offset, col_offset, col_shift, num_scs,
      global_device_count, combiner, coo_tensors);
}

}  // namespace jax_sc_embedding
