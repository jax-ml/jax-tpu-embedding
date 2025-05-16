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
#include "jax_tpu_embedding/sparsecore/lib/core/fdo_types.h"

#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/numpy.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "third_party/pybind11_abseil/absl_casters.h"

namespace jax_sc_embedding {

namespace py = ::pybind11;

PYBIND11_MODULE(fdo_types_cc, m) {
  py::class_<FdoStats>(m, "FdoStats")
      .def_readonly("max_ids_per_partition", &FdoStats::max_ids_per_partition)
      .def_readonly("max_unique_ids_per_partition",
                    &FdoStats::max_unique_ids_per_partition)
      .def_readonly("id_drop_counters", &FdoStats::id_drop_counters)
      .def_readonly("required_buffer_sizes", &FdoStats::required_buffer_sizes);
}

}  // namespace jax_sc_embedding
