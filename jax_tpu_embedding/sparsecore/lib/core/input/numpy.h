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

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/input/input_formats.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "pybind11/gil.h"  // from @pybind11
#include "pybind11/numpy.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "tsl/profiler/lib/traceme.h"

namespace jax_sc_embedding {

namespace py = ::pybind11;

// TODO: There is scope for not using python types here at all (or at least to a
// greater extent).
class NumpyArrayInput : public SparseInput {
  py::list feature_specs;

  py::list features;
  py::list feature_weights;

 private:
  // TODO: abseil.io/tips/176 - move coo_tensors to output (we can use
  // coo_tensors.size())
  //
  // `features` and `feature_weights` are 2D arrays, which
  // means they are rectangular shaped arrays of dtype int (features) and float
  // (feature_weights).
  int ExtractCooTensorsFrom2dArray(const py::array& features,
                                   const py::array& feature_weights,
                                   int row_offset, int col_offset,
                                   int col_shift, int num_scs_mod,
                                   int num_scs_mod_inv, int global_device_count,
                                   RowCombiner combiner,
                                   std::vector<CooFormat>& coo_tensors) const;

  float ComputeWeightDivisor(RowCombiner combiner, const float* gains_buffer,
                             py::ssize_t stride, py::ssize_t size) const;

  // TODO: abseil.io/tips/176 - move coo_tensors to output (we can use
  // coo_tensors.size())
  //
  // `features` and `feature_weights` are 1D arrays of arrays. That is, they
  // are numpy arrays with dtype=object where the object is a 1D array of ints
  // (features) and floats (feature_weights). When looping over the inner
  // arrays, we have to cast the object to a py::array_t<T> and then access the
  // inner arrays.
  int ExtractCooTensorsFrom1dArray(const py::array& features,
                                   const py::array& feature_weights,
                                   int row_offset, int col_offset,
                                   int col_shift, int num_scs_mod,
                                   int num_scs_mod_inv, int global_device_count,
                                   RowCombiner combiner,
                                   std::vector<CooFormat>& coo_tensors) const;

  // TODO: abseil.io/tips/176 - move coo_tensors to output (we can use
  // coo_tensors.size())
  int ExtractCooTensors(py::array features_split,
                        py::array feature_weights_split, int row_offset,
                        int col_offset, int col_shift, int num_scs,
                        int global_device_count, RowCombiner combiner,
                        std::vector<CooFormat>& coo_tensors) const;

 public:
  NumpyArrayInput(py::list features, py::list feature_weights,
                  py::list feature_specs)
      : feature_specs(feature_specs),
        features(features),
        feature_weights(feature_weights) {
    CHECK(features.size() == feature_weights.size());
    CHECK(features.size() == feature_specs.size());
  }
  absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>
  GetStackedTableMetadata() const override;

  // Note that the stacked_table_metadata list is sorted by row offsets of the
  // features.
  std::vector<CooFormat> ExtractCooTensorsForAllFeatures(
      absl::Span<const StackedTableMetadata> stacked_table_metadata,
      int local_device_id, int local_device_count, int num_scs,
      int num_global_devices, int& batch_size_for_device,
      int& total_num_coo_tensors) const override;
};

}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_INPUT_NUMPY_H_
