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
#include "jax_tpu_embedding/sparsecore/lib/core/tf_sparse_tensor_input_batch.h"

#include <Python.h>

#include <cmath>
#include <vector>

#include "absl/log/check.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/abstract_input_batch.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "pybind11/gil.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "tsl/profiler/lib/traceme.h"

namespace jax_sc_embedding {

namespace {
float ComputeGain(RowCombiner combiner, int elements_in_row) {
  switch (combiner) {
    case RowCombiner::kSum:
      return 1.0;
    case RowCombiner::kMean:
      return 1.0 / elements_in_row;
    case RowCombiner::kSqrtn:
      return 1.0 / std::sqrt<float>(elements_in_row);
  }
}
}  // namespace

AbstractInputBatch* TfSparseTensorInputBatch::Slice(
    int row_start, int row_end_exclusive) const {
  DCHECK(!PyGILState_Check());  // Does not require external GIL
  // We do not slice the indices/values, since row_pointers need to be updated
  // too. We just copy a reference of them with updated row_start and row_end
  // values.
  py::gil_scoped_acquire _;
  return new TfSparseTensorInputBatch(indices_, values_, row_pointers_,
                                      row_start, row_end_exclusive,
                                      max_col_id_);
}

void TfSparseTensorInputBatch::ExtractCooTensors(
    int row_offset, int col_offset, int col_shift, int num_scs,
    int global_device_count, RowCombiner combiner,
    std::vector<CooFormat>& coo_tensors) const {
  DCHECK(!PyGILState_Check());  // Does not require external GIL
  tsl::profiler::TraceMe t([] { return "ExtractCooTensors"; });

  CHECK(num_scs > 0 && (num_scs & (num_scs - 1)) == 0);
  DCHECK_GT(global_device_count, 0);
  const int num_scs_bit = std::log2(num_scs);
  const int num_scs_mod = (1 << num_scs_bit) - 1;
  const int num_scs_mod_inv = ~num_scs_mod;
  const int row_offset_per_device = row_offset / global_device_count;
  auto values_array = values_.unchecked<1>();
  for (int row_id = row_start_; row_id < row_end_; ++row_id) {
    const int adjusted_row = row_id - row_start_ + row_offset_per_device;
    const int elements_in_row =
        row_pointers_[row_id + 1] - row_pointers_[row_id];
    const float gain = ComputeGain(combiner, elements_in_row);
    for (int i = row_pointers_[row_id]; i < row_pointers_[row_id + 1]; ++i) {
      const int col_id = values_array(i);
      DCHECK(col_id >= 0 && col_id <= max_col_id_)
          << "Invalid col id: " << col_id
          << " for table vocabulary size: " << max_col_id_;
      coo_tensors.emplace_back(
          adjusted_row,
          GetColId(col_id, col_shift, col_offset, num_scs_mod, num_scs_mod_inv),
          gain);
    }
  }
}
}  // namespace jax_sc_embedding
