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
#include <vector>

#include "absl/base/call_once.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "jax_tpu_embedding/sparsecore/lib/core/process_coo_tensors_impl.h"
#include "jax_tpu_embedding/sparsecore/lib/core/sparse_csr_input_stream_impl.h"
#include "jax_tpu_embedding/sparsecore/lib/core/unity_weights_stream_impl.h"
#include "pybind11/gil.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "tsl/profiler/lib/traceme.h"

namespace jax_sc_embedding {

void PySparseCooInputBatch::ConstructRowPointers() const {
  if (!row_pointers_.empty()) {
    return;
  }
  auto indices_array = indices_.unchecked<2>();
  auto values_array = values_.unchecked<1>();
  // Precompute indexes for row starts. Add a sentinel node for last row.
  row_pointers_.reserve(batch_size_ + 1);
  int row_pointers_index = 0;
  int last_row_id = -1;  // Only for DCHECK.
  int last_col_id = -1;  // Only for DCHECK.
  int last_val = -1;     // Only for DCHECK.
  for (int i = 0; i < indices_array.shape(0); ++i) {
    const int row_id = indices_array(i, 0), col_id = indices_array(i, 1),
              val = values_array(i);
    DCHECK_GE(row_id, last_row_id) << "Decreasing row id values for row-major.";
    while (row_pointers_index <= row_id) {
      // Increment index until we reach the current row. Keep storing the row
      // pointers.
      row_pointers_.push_back(i);
      ++row_pointers_index;
    }

    // Loop Invariant: The index should point to one beyond the current row id.
    DCHECK_EQ(row_pointers_index, row_id + 1);

    if (row_id == last_row_id) {  // Same Row should have increasing col values.
      DCHECK_GT(col_id, last_col_id)
          << "Non-increasing col id values for row-major.";
    }

    last_row_id = row_id;  // NOMUTANTS - debugging.
    last_col_id = col_id;  // NOMUTANTS - debugging.
    last_val = val;        // NOMUTANTS - debugging.
  }
  while (row_pointers_index <= batch_size_) {
    row_pointers_.push_back(indices_array.shape(0));
    row_pointers_index++;
  }

  DCHECK_EQ(row_pointers_.size(), batch_size_ + 1);
}

void PySparseCooInputBatch::ConstructRowPointersIfRequired() const {
  absl::call_once(row_pointer_construction_flag_,
                  &PySparseCooInputBatch::ConstructRowPointers, this);
}

void PySparseCooInputBatch::ExtractCooTensors(
    const ExtractCooTensorsOptions& options, ExtractedCooTensors& coo_tensors) {
  DCHECK(!PyGILState_Check());  // Does not require external GIL.
  tsl::profiler::TraceMe t([] { return "ExtractCooTensors"; });

  ConstructRowPointersIfRequired();

  SparseCsrInputBatchStream<int64_t,
                            pybind11::detail::unchecked_reference<int, 1>,
                            absl::Span<const int64_t>>
      values_stream(values_.unchecked<1>(), absl::MakeConstSpan(row_pointers_),
                    options.slice_start, options.slice_end, table_name_,
                    max_vocab_id_);
  UnityWeightsStream weights_stream(values_stream);

  ProcessCooTensors(options, values_stream, weights_stream, coo_tensors);
}
}  // namespace jax_sc_embedding
