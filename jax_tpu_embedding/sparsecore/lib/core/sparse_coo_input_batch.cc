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
#include "jax_tpu_embedding/sparsecore/lib/core/sparse_coo_input_batch.h"

#include <Python.h>

#include <cstdint>
#include <optional>

#include "absl/log/check.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/abstract_input_batch.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "jax_tpu_embedding/sparsecore/lib/core/process_coo_tensors_impl.h"
#include "jax_tpu_embedding/sparsecore/lib/core/sparse_coo_input_stream_impl.h"
#include "jax_tpu_embedding/sparsecore/lib/core/unity_weights_stream_impl.h"
#include "pybind11/numpy.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "tsl/profiler/lib/traceme.h"

namespace jax_sc_embedding {

std::optional<int64_t> PySparseCooInputBatch::GetIdsCountInSlice(
    int start_row, int end_row) const {
  if (start_row >= end_row) {
    return 0;
  }
  auto indices_array = indices_.unchecked<2>();
  const int64_t num_elements = indices_array.shape(0);
  if (num_elements == 0) {
    return 0;
  }
  const int64_t start_idx =
      LowerBoundRow(indices_array, /*start_idx=*/0, num_elements, start_row);
  const int64_t end_idx =
      LowerBoundRow(indices_array, start_idx, num_elements, end_row);
  return end_idx - start_idx;
}

namespace {

template <typename ValueT, typename ColIdT>
void ExtractSparseCooTensorsWithValueType(
    const py::array_t<int64_t>& indices, const py::array& values,
    const AbstractInputBatch::ExtractCooTensorsOptions& options,
    ExtractedCooTensorsPerSparseCoreT<ColIdT>& coo_tensors) {
  SparseCooInputBatchStream<ValueT,
                            pybind11::detail::unchecked_reference<int64_t, 2>,
                            pybind11::detail::unchecked_reference<ValueT, 1>>
      values_stream(indices.unchecked<2>(), values.unchecked<ValueT, 1>(),
                    options.slice_start, options.slice_end);
  UnityWeightsStream weights_stream(values_stream);
  ProcessCooTensors(options, values_stream, weights_stream, coo_tensors);
}

}  // namespace

template <typename ColIdT>
void PySparseCooInputBatch::ExtractCooTensorsImpl(
    const ExtractCooTensorsOptions& options,
    ExtractedCooTensorsPerSparseCoreT<ColIdT>& coo_tensors) {
  DCHECK(!PyGILState_Check());  // Does not require external GIL.
  tsl::profiler::TraceMe t(
      [&] { return absl::StrCat("ExtractCooTensors/", table_name_); });

  if (is_int64_) {
    ExtractSparseCooTensorsWithValueType<int64_t>(indices_, values_, options,
                                                  coo_tensors);
  } else {
    ExtractSparseCooTensorsWithValueType<int32_t>(indices_, values_, options,
                                                  coo_tensors);
  }
}

void PySparseCooInputBatch::ExtractCooTensors(
    const ExtractCooTensorsOptions& options,
    ExtractedCooTensorsPerSparseCore& coo_tensors) {
  ExtractCooTensorsImpl(options, coo_tensors);
}

void PySparseCooInputBatch::ExtractCooTensors64(
    const ExtractCooTensorsOptions& options,
    ExtractedCooTensorsPerSparseCoreT<int64_t>& coo_tensors) {
  ExtractCooTensorsImpl(options, coo_tensors);
}

}  // namespace jax_sc_embedding
