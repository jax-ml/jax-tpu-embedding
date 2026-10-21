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
#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_INPUT_PREPROCESSING_MEGACORE_UTIL_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_INPUT_PREPROCESSING_MEGACORE_UTIL_H_

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "jax_tpu_embedding/sparsecore/lib/core/partitioned_coo_tensors.h"
#include "xla/tsl/concurrency/async_value_ref.h"  // from @xla

namespace jax_sc_embedding {

// Megacore counterpart of `FillLocalDeviceBufferAsync`. Fills a single
// chip-level CSR buffer for a stacked table and returns the number of dropped
// IDs. Megacore CSR does not support host minibatching, so there must be a
// single minibatch.
tsl::AsyncValueRef<int> FillMegacoreLocalDeviceBufferAsync(
    const DevicePartitionedCooTensors& grouped_coo_tensors,
    int batch_size_per_sc,
    const PreprocessSparseDenseMatmulInputOptions& options,
    absl::string_view stacked_table_name, MegacoreCsrOptions megacore_csr,
    internal::CsrArraysRefPerDevice csr_arrays);

}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_INPUT_PREPROCESSING_MEGACORE_UTIL_H_
