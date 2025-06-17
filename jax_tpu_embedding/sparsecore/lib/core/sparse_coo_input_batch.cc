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

#include <cmath>
#include <vector>

#include "absl/base/call_once.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "pybind11/gil.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "tsl/profiler/lib/traceme.h"

namespace jax_sc_embedding {

namespace {
float ComputeGain(RowCombiner combiner, int elements_in_row) {
  switch (combiner) {
    case RowCombiner::kSum:
      return 1.0f;
    case RowCombiner::kMean:
      return 1.0f / elements_in_row;
    case RowCombiner::kSqrtn:
      return 1.0f / std::sqrt(static_cast<float>(elements_in_row));
  }
}
}  // namespace

void PySparseCooInputBatch::ConstructRowPointers() {
  if (!row_pointers_.empty()) {
    return;
  }
  auto indices_array = indices_.unchecked<2>();
  // Precompute indexes for row starts. Add a sentinel node for last row.
  const int total_rows = row_end_ - row_start_;
  // We use relative indexing to the first row id. This prevents issue with
  // memory usage where row_start is really large.
  row_pointers_.reserve(1 + total_rows);
  int next_row_id = row_start_;
  int row_id = -1;
  int col_id = -1;  // only for DCHECK.
  for (int i = 0; i < indices_array.shape(0); ++i) {
    DCHECK_GE(indices_array(i, 0), row_id)
        << "Invalid row ordering for row-major.";
    row_id = indices_array(i, 0);
    if (row_id == next_row_id) {  // New Row
      row_pointers_.push_back(i);
      ++next_row_id;
    } else {  // Same Row
      DCHECK_GT(indices_array(i, 1), col_id)
          << "Invalid col ordering for row-major.";
    }
    col_id = indices_array(i, 1);
  }
  // sentinel node for the last row.
  row_pointers_.push_back(indices_array.shape(0));
}

void PySparseCooInputBatch::ConstructRowPointersIfRequired() {
  absl::call_once(row_pointer_construction_flag_,
                  &PySparseCooInputBatch::ConstructRowPointers, this);
}

void PySparseCooInputBatch::ExtractCooTensors(
    int row_start, int row_end, int row_offset, int col_offset, int col_shift,
    int num_scs, int global_device_count, RowCombiner combiner,
    std::vector<CooFormat>& coo_tensors) {
  DCHECK(!PyGILState_Check());  // Does not require external GIL.
  tsl::profiler::TraceMe t([] { return "ExtractCooTensors"; });

  CHECK(num_scs > 0 && (num_scs & (num_scs - 1)) == 0);
  DCHECK_GT(global_device_count, 0);

  ConstructRowPointersIfRequired();

  const int num_scs_bit = std::log2(num_scs);
  const int num_scs_mod = (1 << num_scs_bit) - 1;
  const int num_scs_mod_inv = ~num_scs_mod;
  const int row_offset_per_device = row_offset / global_device_count;
  auto values_array = values_.unchecked<1>();
  for (int row_id = row_start; row_id < row_end; ++row_id) {
    const int adjusted_row = row_id - row_start + row_offset_per_device;
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
