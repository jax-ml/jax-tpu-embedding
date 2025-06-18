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
  row_pointers_.reserve(batch_size_ + 1);
  int row_pointers_index = 0;
  int last_row_id = -1;  // Only for DCHECK.
  int last_col_id = -1;  // Only for DCHECK.
  int last_val = -1;     // Only for DCHECK.
  for (int i = 0; i < indices_array.shape(0); ++i) {
    const int row_id = indices_array(i, 0), col_id = indices_array(i, 1),
              val = values_.at(i);
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
      DCHECK(col_id >= 0 && col_id <= max_vocab_id_)
          << "Invalid col id: " << col_id
          << " for table vocabulary size: " << max_vocab_id_;
      coo_tensors.emplace_back(
          adjusted_row,
          GetColId(col_id, col_shift, col_offset, num_scs_mod, num_scs_mod_inv),
          gain);
    }
  }
}
}  // namespace jax_sc_embedding
