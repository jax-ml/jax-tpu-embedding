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
#include <Python.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/abstract_input_batch.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "jax_tpu_embedding/sparsecore/lib/core/numpy_input_batch.h"
#include "jax_tpu_embedding/sparsecore/lib/core/sparse_coo_input_batch.h"
#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/eigen.h"  // from @pybind11
#include "pybind11/gil.h"  // from @pybind11
#include "pybind11/numpy.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil  // IWYU pragma: keep
#include "tsl/profiler/lib/traceme.h"

namespace jax_sc_embedding {

namespace py = ::pybind11;

namespace {
absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>
GetStackedTableMetadata(py::list& feature_specs) {
  CHECK(PyGILState_Check());  // Requires GIL
  tsl::profiler::TraceMe t([] { return "GetStackedTableMetadata"; });
  absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>
      stacked_table_metadata;
  for (int i = 0; i < feature_specs.size(); ++i) {
    const py::object& feature_spec = feature_specs[i];
    const py::object& feature_transformation =
        feature_spec.attr("_id_transformation");
    const py::object& table_spec = feature_spec.attr("table_spec");
    const int64_t max_col_id =
        py::cast<int64_t>(table_spec.attr("vocabulary_size"));
    const py::list& input_shape =
        feature_spec.attr("input_shape").cast<py::list>();
    const int batch_size = py::cast<int>(input_shape[0]);
    const py::object& stacked_table_spec =
        table_spec.attr("stacked_table_spec");
    if (stacked_table_spec.is_none()) {
      LOG(ERROR) << "stacked_table_spec is none for table"
                 << py::cast<std::string>(table_spec.attr("name"));
    }
    const std::string stacked_table_name = py::cast<std::string>(
        table_spec.attr("setting_in_stack").attr("stack_name"));
    int col_shift = 0;
    int col_offset = 0;
    int row_offset = 0;
    const int max_ids_per_partition =
        py::cast<int>(stacked_table_spec.attr("max_ids_per_partition"));
    const int max_unique_ids_per_partition =
        py::cast<int>(stacked_table_spec.attr("max_unique_ids_per_partition"));
    std::optional<int> suggested_coo_buffer_size;
    py::object suggested_coo_buffer_size_attr =
        stacked_table_spec.attr("suggested_coo_buffer_size");
    if (!suggested_coo_buffer_size_attr.is_none()) {
      suggested_coo_buffer_size = py::cast<int>(suggested_coo_buffer_size_attr);
    }
    const std::string row_combiner =
        py::cast<std::string>(stacked_table_spec.attr("combiner"));
    if (!feature_transformation.is_none()) {
      row_offset = py::cast<int>(feature_transformation.attr("row_offset"));
      col_shift = py::cast<int>(feature_transformation.attr("col_shift"));
      col_offset = py::cast<int>(feature_transformation.attr("col_offset"));
    }
    stacked_table_metadata[stacked_table_name].emplace_back(
        stacked_table_name, i, max_ids_per_partition,
        max_unique_ids_per_partition, row_offset, col_offset, col_shift,
        /*batch_size=*/batch_size, suggested_coo_buffer_size,
        GetRowCombiner(row_combiner), max_col_id);
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

py::tuple PyPreprocessSparseDenseMatmulInput(
    absl::Span<std::unique_ptr<AbstractInputBatch>> input_batches,
    py::list feature_specs, int local_device_count, int global_device_count,
    int num_sc_per_device, int sharding_strategy, bool has_leading_dimension,
    bool allow_id_dropping) {
  CHECK_EQ(input_batches.size(), feature_specs.size());
  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = local_device_count,
      .global_device_count = global_device_count,
      .num_sc_per_device = num_sc_per_device,
      .sharding_strategy = sharding_strategy,
      .allow_id_dropping = allow_id_dropping,
  };
  // Get the stacked table metadata for each top level table.
  // The keys are stacked table names (or the table itself if not stacked) and
  // the values are a vector of StackedTableMetadata for each feature that is
  // mapped to the table.
  const absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>
      stacked_tables = GetStackedTableMetadata(feature_specs);
  PreprocessSparseDenseMatmulOutput out;
  {
    // We release the lock by default and acquire it when we deal with python
    // objects (features, specs and weights).
    py::gil_scoped_release release;

    out = PreprocessSparseDenseMatmulInput(absl::MakeSpan(input_batches),
                                           stacked_tables, options);
  }
  // We need the GIL back to create the output tuple. The tuple creation
  // implicitly wraps Eigen matrices into numpy arrays (without copying), which
  // makes it easier to flatten them using `reshape(-1)`.
  py::tuple ret_object = py::make_tuple(
      std::move(out.lhs_row_pointers), std::move(out.lhs_embedding_ids),
      std::move(out.lhs_sample_ids), std::move(out.lhs_gains),
      std::move(out.stats));
  // Skip the last element (stats).
  for (size_t i = 0; i < ret_object.size() - 1; ++i) {
    if (!has_leading_dimension) {
      for (auto& iterator : ret_object[i].cast<py::dict>()) {
        ret_object[i][iterator.first] =
            py::cast<py::array>(iterator.second).reshape({-1});
      }
    }
  }
  return ret_object;
}

py::tuple PyNumpyPreprocessSparseDenseMatmulInput(
    py::list features, py::list feature_weights, py::list feature_specs,
    int local_device_count, int global_device_count, int num_sc_per_device,
    int sharding_strategy, bool has_leading_dimension, bool allow_id_dropping) {
  CHECK_EQ(features.size(), feature_weights.size());
  std::vector<std::unique_ptr<AbstractInputBatch>> input_batches;
  input_batches.reserve(features.size());
  for (int i = 0; i < features.size(); ++i) {
    input_batches.push_back(std::make_unique<NumpySparseInputBatch>(
        features[i], feature_weights[i]));
  }
  return PyPreprocessSparseDenseMatmulInput(
      absl::MakeSpan(input_batches), feature_specs, local_device_count,
      global_device_count, num_sc_per_device, sharding_strategy,
      has_leading_dimension, allow_id_dropping);
}

py::tuple PySparseCooPreprocessSparseDenseMatmulInput(
    py::list indices, py::list values, py::list dense_shapes,
    py::list feature_specs, int local_device_count, int global_device_count,
    int num_sc_per_device, int sharding_strategy, bool has_leading_dimension,
    bool allow_id_dropping) {
  CHECK(indices.size() == values.size());
  // NOTE: dense_shapes is unused and can be removed.
  CHECK(indices.size() == dense_shapes.size());
  std::vector<std::unique_ptr<AbstractInputBatch>> input_batches(
      indices.size());

  for (int i = 0; i < indices.size(); ++i) {
    const int64_t max_col_id =
        feature_specs[i].attr("table_spec").attr("vocabulary_size").cast<int>();
    input_batches[i] = std::make_unique<PySparseCooInputBatch>(
        indices[i].cast<py::array_t<int64_t>>(),
        values[i].cast<py::array_t<int32_t>>(), max_col_id);
  }
  return PyPreprocessSparseDenseMatmulInput(
      absl::MakeSpan(input_batches), feature_specs, local_device_count,
      global_device_count, num_sc_per_device, sharding_strategy,
      has_leading_dimension, allow_id_dropping);
}
}  // namespace

PYBIND11_MODULE(pybind_input_preprocessing, m) {
  m.def("PreprocessSparseDenseMatmulInput",
        &PyNumpyPreprocessSparseDenseMatmulInput, pybind11::arg("features"),
        pybind11::arg("feature_weights"), pybind11::arg("feature_specs"),
        pybind11::arg("local_device_count"),
        pybind11::arg("global_device_count"),
        pybind11::arg("num_sc_per_device"), pybind11::arg("sharding_strategy"),
        pybind11::arg("has_leading_dimension"),
        pybind11::arg("allow_id_dropping"));
  m.def("PreprocessSparseDenseMatmulSparseCooInput",
        &PySparseCooPreprocessSparseDenseMatmulInput, pybind11::arg("indices"),
        pybind11::arg("values"), pybind11::arg("dense_shapes"),
        pybind11::arg("feature_specs"), pybind11::arg("local_device_count"),
        pybind11::arg("global_device_count"),
        pybind11::arg("num_sc_per_device"), pybind11::arg("sharding_strategy"),
        pybind11::arg("has_leading_dimension"),
        pybind11::arg("allow_id_dropping"));
  py::class_<SparseDenseMatmulInputStats>(m, "SparseDenseMatmulInputStats")
      .def(py::init<>())
      .def_readonly("max_ids_per_partition",
                    &SparseDenseMatmulInputStats::max_ids_per_partition)
      .def_readonly("max_unique_ids_per_partition",
                    &SparseDenseMatmulInputStats::max_unique_ids_per_partition)
      .def_readonly("required_buffer_sizes",
                    &SparseDenseMatmulInputStats::required_buffer_sizes);
}
}  // namespace jax_sc_embedding
