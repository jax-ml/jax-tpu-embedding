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
#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_INPUT_PREPROCESSING_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_INPUT_PREPROCESSING_H_
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"

namespace jax_sc_embedding {

struct PreprocessSparseDenseMatmulInputOptions {
  int local_device_count;
  int global_device_count;
  int num_sc_per_device;
  int sharding_strategy;
  bool allow_id_dropping;

  int GetNumScs() const { return num_sc_per_device * global_device_count; }
};

template <typename T>
using StackedTableMap = absl::flat_hash_map<std::string, T>;

struct SparseDenseMatmulInputStats {
  StackedTableMap<RowVectorXi> max_ids_per_partition;
  StackedTableMap<RowVectorXi> max_unique_ids_per_partition;
  StackedTableMap<RowVectorXi> required_buffer_sizes;
};

struct ExtractedCooTensors {
  std::vector<CooFormat> coo_tensors;
  int batch_size_for_device = 0;
};

struct PreprocessSparseDenseMatmulOutput {
  StackedTableMap<MatrixXi> lhs_row_pointers;
  StackedTableMap<MatrixXi> lhs_embedding_ids;
  StackedTableMap<MatrixXi> lhs_sample_ids;
  StackedTableMap<MatrixXf> lhs_gains;
  SparseDenseMatmulInputStats stats;
};

}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_INPUT_PREPROCESSING_H_
