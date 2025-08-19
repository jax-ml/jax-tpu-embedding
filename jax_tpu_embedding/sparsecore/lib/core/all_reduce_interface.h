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
#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_ALL_REDUCE_INTERFACE_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_ALL_REDUCE_INTERFACE_H_

#include <cstdint>

#include "absl/status/statusor.h"  // from @com_google_absl

namespace jax_sc_embedding {

// Interface for performing all-reduce operations across multiple participants.
// Implementations of this interface are used to synchronize and aggregate
// values (e.g., boolean flags or uint64_t masks) across different hosts or
// devices using a shared `sync_key`.
class AllReduceInterface {
 public:
  virtual ~AllReduceInterface() = default;
  // Performs a blocking all-reduce operation for a boolean value.
  // The result is typically the logical OR of all `minibatching_required`
  // values from participants sharing the same `sync_key`.
  virtual absl::StatusOr<bool> BlockingAllReduce(
      int sync_key, bool minibatching_required) = 0;

  // Performs a blocking all-reduce operation for a uint64_t value.
  // The result is typically the logical OR of all `minibatching_split`
  // values from participants sharing the same `sync_key`.
  virtual absl::StatusOr<uint64_t> BlockingAllReduce(
      int sync_key, uint64_t minibatching_split) = 0;
};
}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_ALL_REDUCE_INTERFACE_H_
