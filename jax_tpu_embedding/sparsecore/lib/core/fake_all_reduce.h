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
#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_FAKE_ALL_REDUCE_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_FAKE_ALL_REDUCE_H_

#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"

namespace jax_sc_embedding {

namespace testing_utils {

// Returns a callback function that performs an AllReduce operation using
// bitwise OR. Boolean values should be passed as 0ULL (false) or 1ULL (true).
// The callback is thread-safe and reusable across multiple reduction rounds.
AllReduceCallback CreateFakeAllReduceCallback(int host_count);

}  // namespace testing_utils
}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_FAKE_ALL_REDUCE_H_
