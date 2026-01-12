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
#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_NUMPY_INPUT_BATCH_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_NUMPY_INPUT_BATCH_H_

#include <cstdint>
#include <optional>

#include "absl/log/check.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/abstract_input_batch.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/gil.h"  // from @pybind11
#include "pybind11/numpy.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11

namespace jax_sc_embedding {

namespace py = ::pybind11;

class NumpySparseInputBatch : public AbstractInputBatch {
 public:
  NumpySparseInputBatch(const py::array& feature)
      : NumpySparseInputBatch(feature, std::nullopt) {}

  NumpySparseInputBatch(const py::array& feature,
                        std::optional<py::array> weights)
      : feature_(feature), weights_(weights) {
    DCHECK(PyGILState_Check())
        << "Need GIL to create references to features and weights.";
    if (weights_.has_value()) {
      CHECK_EQ(feature_.shape(0), weights_->shape(0))
          << "Batch size mismatch for features and weights.";
      CHECK_EQ(feature_.ndim(), weights_->ndim())
          << "Dimension mismatch for features and weights";
    }
    CHECK(feature_.ndim() == 1 || feature_.ndim() == 2)
        << "Only 1D and 2D numpy arrays supported as inputs.";

    if (feature_.ndim() == 1) {
      // Iterating over every row to sum up the number of IDs negates the
      // performance benefit of reserving memory for them, so we underestimate
      // the number of IDs as 1 per sample.
      id_count_ = feature_.shape(0);
    } else {
      id_count_ = feature_.shape(0) * feature_.shape(1);
    }
  }

  // Returns the number of samples in this input batch.
  int64_t size() const override { return feature_.shape(0); }

  // Returns the total number of embedding IDs across all samples.
  int64_t id_count() const override { return id_count_; }

  std::optional<int64_t> GetIdsCountInSlice(int start_row,
                                            int end_row) const override {
    if (feature_.ndim() == 2) {
      return (end_row - start_row) * feature_.shape(1);
    }
    return std::nullopt;
  }

  bool HasVariableWeights() const override { return weights_.has_value(); }

  void ExtractCooTensors(
      const ExtractCooTensorsOptions& options,
      ExtractedCooTensorsPerSparseCore& coo_tensors) override;

 private:
  const py::array feature_;
  // NOTE: stored as optional<py::array> instead of py::object to avoid GIL
  // locking to cast to array every time.
  const std::optional<const py::array> weights_;
  int64_t id_count_;
};

}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_NUMPY_INPUT_BATCH_H_
