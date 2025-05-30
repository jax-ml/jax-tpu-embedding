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
#include <memory>

#include "jax_tpu_embedding/sparsecore/lib/core/input/numpy.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing.h"
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/eigen.h"  // from @pybind11
#include "pybind11/gil.h"  // from @pybind11
#include "pybind11/numpy.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "third_party/pybind11_abseil/absl_casters.h"

namespace jax_sc_embedding {

namespace {

namespace py = ::pybind11;

py::tuple PyNumpyPreprocessSparseDenseMatmulInput(
    py::list features, py::list feature_weights, py::list feature_specs,
    int local_device_count, int global_device_count, int num_sc_per_device,
    int sharding_strategy, bool has_leading_dimension, bool allow_id_dropping) {
  // We cannot pass a reference to a temporary object, hence create this on
  // the heap.
  std::unique_ptr<NumpyArrayInput> sparse_input =
      std::make_unique<NumpyArrayInput>(features, feature_weights,
                                        feature_specs);
  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = local_device_count,
      .global_device_count = global_device_count,
      .num_sc_per_device = num_sc_per_device,
      .sharding_strategy = sharding_strategy,
      .allow_id_dropping = allow_id_dropping,
  };
  PreprocessSparseDenseMatmulOutput out;
  {
    // We release the lock by default and acquire it in sparse input
    // transformation functions (and not the core preprocessing) where we need
    // it.
    py::gil_scoped_release release;
    out = PreprocessSparseDenseMatmulInput(*sparse_input, options);
  }
  // need the GIL back to create tuple.
  py::tuple ret_object =
      py::make_tuple(out.lhs_row_pointers, out.lhs_embedding_ids,
                     out.lhs_sample_ids, out.lhs_gains, out.stats);
  // TODO: better way than to have a magic number 4.
  for (int i = 0; i < 4; i++) {
    if (!has_leading_dimension) {
      for (auto& iterator : ret_object[i].cast<py::dict>()) {
        ret_object[i][iterator.first] =
            py::cast<py::array>(iterator.second).reshape({-1});
      }
    }
  }
  return ret_object;
}
}  // namespace

PYBIND11_MODULE(input_preprocessing_cc, m) {
  // TODO: Similarly define one function for tf.sparse.tensor.
  m.def("PreprocessSparseDenseMatmulInput",
        &PyNumpyPreprocessSparseDenseMatmulInput, pybind11::arg("features"),
        pybind11::arg("feature_weights"), pybind11::arg("feature_specs"),
        pybind11::arg("local_device_count"),
        pybind11::arg("global_device_count"),
        pybind11::arg("num_sc_per_device"), pybind11::arg("sharding_strategy"),
        pybind11::arg("has_leading_dimension"),
        pybind11::arg("allow_id_dropping"));
  py::class_<PreprocessSparseDenseMatmulStats>(
      m, "PreprocessSparseDenseMatmulStats")
      .def(py::init<>())
      .def_readonly("max_ids_per_partition",
                    &PreprocessSparseDenseMatmulStats::max_ids_per_partition)
      .def_readonly(
          "max_unique_ids_per_partition",
          &PreprocessSparseDenseMatmulStats::max_unique_ids_per_partition)
      .def_readonly("required_buffer_sizes",
                    &PreprocessSparseDenseMatmulStats::required_buffer_sizes);
}
}  // namespace jax_sc_embedding
