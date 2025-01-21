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
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_py_util.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/gil.h"  // from @pybind11
#include "pybind11/numpy.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "tsl/profiler/lib/traceme.h"  // from @tsl

namespace jax_sc_embedding {

namespace py = ::pybind11;

absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>
GetStackedTableMetadata(const py::list& feature_specs, const int batch_size) {
  tsl::profiler::TraceMe t([] { return "GetStackedTableMetadata"; });
  absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>
      stacked_table_metadata;
  for (int i = 0; i < feature_specs.size(); ++i) {
    const py::object& feature_spec = feature_specs[i];

    const py::object& feature_transformation =
        feature_spec.attr("_id_transformation");
    const py::object& table_spec = feature_spec.attr("table_spec");
    const py::object& stacked_table_spec =
        table_spec.attr("stacked_table_spec");
    const std::string stacked_table_name = py::cast<std::string>(
        table_spec.attr("_setting_in_stack").attr("stack_name"));
    int col_shift = 0;
    int col_offset = 0;
    int row_offset = 0;
    const int max_ids_per_partition =
        py::cast<int>(stacked_table_spec.attr("max_ids_per_partition"));
    const int max_unique_ids_per_partition =
        py::cast<int>(stacked_table_spec.attr("max_unique_ids_per_partition"));
    const int vocab_size =
        py::cast<int>(stacked_table_spec.attr("stack_vocab_size"));
    if (!feature_transformation.is_none()) {
      row_offset = py::cast<int>(feature_transformation.attr("row_offset"));
      col_shift = py::cast<int>(feature_transformation.attr("col_shift"));
      col_offset = py::cast<int>(feature_transformation.attr("col_offset"));
    }
    stacked_table_metadata[stacked_table_name].emplace_back(
        i, max_ids_per_partition, max_unique_ids_per_partition, row_offset,
        col_offset, col_shift,
        /*batch_size=*/batch_size, vocab_size);
  }
  // Sort the stacked tables by row_offset.
  for (auto& [_, t] : stacked_table_metadata) {
    std::sort(t.begin(), t.end(),
              [](const StackedTableMetadata& a, const StackedTableMetadata& b) {
                return a.row_offset < b.row_offset;
              });
  }
  return stacked_table_metadata;
}

absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>
GetStackedTableMetadata(const py::list& feature_specs,
                        const py::list& features) {
  tsl::profiler::TraceMe t([] { return "GetStackedTableMetadata"; });
  int batch_size = features[0].cast<py::array>().shape(0);
  return GetStackedTableMetadata(feature_specs, batch_size);
}

}  // namespace jax_sc_embedding
