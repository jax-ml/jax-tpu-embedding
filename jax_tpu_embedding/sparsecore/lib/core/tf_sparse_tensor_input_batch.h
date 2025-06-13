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
#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_TF_SPARSE_TENSOR_INPUT_BATCH_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_TF_SPARSE_TENSOR_INPUT_BATCH_H_

#include <Python.h>

#include <cstdint>
#include <vector>

#include "absl/log/check.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/abstract_input_batch.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/gil.h"  // from @pybind11
#include "pybind11/numpy.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11

namespace jax_sc_embedding {

namespace py = ::pybind11;

class TfSparseTensorInputBatch : public AbstractInputBatch {
 public:
  TfSparseTensorInputBatch(const py::array& indices, const py::array& values,
                           const int64_t max_col_id)
      : indices_(indices), values_(values), max_col_id_(max_col_id) {
    DCHECK(PyGILState_Check())
        << "Need GIL to create references to features and weights.";
    py::gil_scoped_release _;
    auto indices_array = indices_.unchecked<2>();
    // Precompute indexes for row starts. Add a sentinel node for last row.
    row_start_ = indices_array(0, 0);
    row_end_ = indices_array(indices_array.shape(0) - 1, 0) + 1;
    const int total_rows = row_end_ - row_start_;
    // We use relative indexing to the first row id. This prevents issue with
    // memory usage where row_start is really large.
    row_pointers_storage_.reserve(1 + total_rows);
    int row_id = row_start_;
    for (int i = 0; i < indices_array.shape(0); ++i) {
      if (indices_array(i, 0) == row_id) {
        row_pointers_storage_.push_back(i);
        ++row_id;
      }
    }
    // sentinel node for the last row.
    row_pointers_storage_.push_back(indices_array.shape(0));
    row_pointers_ = absl::MakeConstSpan(row_pointers_storage_);
  }

  TfSparseTensorInputBatch(const py::array& indices, const py::array& values,
                           absl::Span<const int64_t> row_pointers,
                           const int row_start, const int row_end,
                           const int64_t max_col_id)
      : indices_(indices),
        values_(values),
        max_col_id_(max_col_id),
        row_pointers_(row_pointers),
        row_start_(row_start),
        row_end_(row_end) {}

  py::ssize_t size() const override { return row_end_ - row_start_; }

  AbstractInputBatch* Slice(int row_start,
                            int row_end_exclusive) const override;

  void ExtractCooTensors(int row_offset, int col_offset, int col_shift,
                         int num_scs, int global_device_count,
                         RowCombiner combiner,
                         std::vector<CooFormat>& coo_tensors) const override;

 private:
  const py::array_t<int64_t> indices_;
  const py::array_t<int32_t> values_;
  const int64_t max_col_id_;
  std::vector<int64_t> row_pointers_storage_;
  absl::Span<const int64_t> row_pointers_;
  int64_t row_start_;
  int64_t row_end_;  // exclusive
};

}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_TF_SPARSE_TENSOR_INPUT_BATCH_H_
