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
#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_SPARSE_COO_INPUT_BATCH_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_SPARSE_COO_INPUT_BATCH_H_

#include <Python.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/abstract_input_batch.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/gil.h"  // from @pybind11
#include "pybind11/numpy.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11

namespace jax_sc_embedding {

namespace py = ::pybind11;

// This class represents a sparse input batch in COO format.
// - `indices` is a 2D array where each row represents a (row_id, col_id) pair.
//   It is assumed that `indices` is sorted in row-major order.
// - `values` is a 1D array where each element represents the value associated
//   with the corresponding (row_id, col_id) pair in `indices`.
class PySparseCooInputBatch : public AbstractInputBatch {
 public:
  PySparseCooInputBatch(const int batch_number,
                        const py::array_t<int64_t>& indices,
                        const py::array_t<int32_t>& values,
                        const py::array_t<int64_t>& dense_shape,
                        const int64_t max_vocab_id,
                        const std::string table_name)
      : batch_number_(batch_number),
        indices_(indices),
        values_(values),
        max_vocab_id_(max_vocab_id),
        batch_size_(dense_shape.at(0)),
        table_name_(std::move(table_name)) {
    DCHECK(PyGILState_Check())
        << "Need GIL to create references to indices and values.";
  }

  // Returns the number of rows in the current slice.
  int64_t size() const override { return batch_size_; }

  // Extracts COO tensors for each SparseCore.
  void ExtractCooTensors(const ExtractCooTensorsOptions& options,
                         std::vector<CooFormat>& coo_tensors) override;

  virtual int batch_number() const override { return batch_number_; }

 private:
  int batch_number_;
  // (N,2) array, sorted by row_id.
  const py::array_t<int64_t> indices_;
  const py::array_t<int32_t> values_;
  const int64_t max_vocab_id_;
  const int64_t batch_size_;
  const std::string table_name_;

  std::vector<int64_t> row_pointers_;
  absl::once_flag row_pointer_construction_flag_;

  // Converts this to a CSR format. A refactor could return an object of type
  // SparseCsrInputBatch after Slicing, and ExtractCooTensors can call
  // the same function on a temporary object of SparseCsrInputBatch type.
  void ConstructRowPointersIfRequired();

  // Internal function called by `ConstructRowPointersIfRequired`.
  void ConstructRowPointers();
};
}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_SPARSE_COO_INPUT_BATCH_H_
