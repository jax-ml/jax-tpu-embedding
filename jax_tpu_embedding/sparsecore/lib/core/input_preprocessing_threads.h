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
#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_INPUT_PREPROCESSING_THREADS_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_INPUT_PREPROCESSING_THREADS_H_

#include "tsl/platform/threadpool.h"  // from @tsl

namespace jax_sc_embedding {

// We use two separate thread pools to handle nested parallelism in input
// preprocessing. Table-level tasks are scheduled onto
// TableProcessingThreadPool, and each of these tasks may schedule multiple
// device-level tasks onto DeviceProcessingThreadPool. If a single pool were
// used, it could lead to deadlock: if all threads in the pool were occupied by
// table-level tasks blocked waiting for device-level tasks to complete, no
// threads would be available to run the device-level tasks, and the system
// would hang. Using separate pools prevents this issue.

// Thread pool for device-level computations in input preprocessing.
tsl::thread::ThreadPool* DeviceProcessingThreadPool();

// Thread pool for table-level computations in input preprocessing.
tsl::thread::ThreadPool* TableProcessingThreadPool();

}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_INPUT_PREPROCESSING_THREADS_H_
