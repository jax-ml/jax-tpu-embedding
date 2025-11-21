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
#include <random>
#include <vector>

#include "testing/base/public/benchmark.h"
#include "jax_tpu_embedding/sparsecore/lib/core/coo_format.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "jax_tpu_embedding/sparsecore/lib/core/sort_and_group_coo_tensors_impl.h"

namespace jax_sc_embedding {
namespace {

void BM_SortAndGroup_Phase1(benchmark::State& state) {
  const int num_sc_per_device = 4;
  const int batch_size_per_sc = 4096;
  const int batch_size_for_device = num_sc_per_device * batch_size_per_sc;

  std::mt19937 gen(31337);  // seed for reproducibility
  std::uniform_int_distribution<> id_distrib(0, 40000000);
  std::uniform_int_distribution<> sample_size_distrib(1, 32);

  ExtractedCooTensors extracted_coo_tensors(num_sc_per_device,
                                            batch_size_for_device);
  for (int row = 0; row < batch_size_for_device; ++row) {
    int sample_size = sample_size_distrib(gen);
    int sc_id = row / batch_size_per_sc;
    extracted_coo_tensors.coo_tensors_per_sc[sc_id] += sample_size;
    for (int i = 0; i < sample_size; ++i) {
      extracted_coo_tensors.coo_tensors.push_back(
          CooFormat(row, id_distrib(gen), 1.0));
    }
  }

  StackedTableMetadata stacked_table_metadata(
      "stacked_table", /*feature_index=*/0, /*max_ids_per_partition=*/40000,
      /*max_unique_ids_per_partition=*/40000, /*row_offset=*/0,
      /*col_offset=*/0,
      /*col_shift=*/0, /*batch_size=*/0);
  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 1,
      .global_device_count = 1,
      .num_sc_per_device = num_sc_per_device,
      .allow_id_dropping = false,
  };
  bool minibatching_required = false;
  StatsPerHost stats_per_host(/*local_device_count=*/1, /*global_sc_count=*/4,
                              /*num_sc_per_device=*/num_sc_per_device);
  internal::StatsPerDevice stats_per_device =
      stats_per_host.GetStatsPerDevice(0);

  for (auto s : state) {
    SortAndGroupCooTensorsPerLocalDevice(
        extracted_coo_tensors, stacked_table_metadata, options,
        stats_per_device, minibatching_required);
  }
}
BENCHMARK(BM_SortAndGroup_Phase1);

}  // namespace
}  // namespace jax_sc_embedding
