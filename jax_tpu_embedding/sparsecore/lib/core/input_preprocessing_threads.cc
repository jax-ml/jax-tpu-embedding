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
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_threads.h"

#include <algorithm>
#include <cstdlib>
#include <new>

#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "third_party/tensorflow/compiler/xla/tsl/platform/env.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/threadpool.h"
#include "tsl/platform/cpu_info.h"  // from @tsl

namespace jax_sc_embedding {

namespace {

constexpr char kScEnv[] = "SPARSECORE_INPUT_PREPROCESSING_THREADS";
constexpr char kScPool[] = "SparseCoreInputPreprocessingThreadPool";

// Returns at least one but the minimum of NumSchedulableCPUs() and the value
// specified by the environment variable
// `SPARSECORE_INPUT_PREPROCESSING_THREADS`.
int GetThreadPoolSize() {
  int num_threads = tsl::port::NumSchedulableCPUs();
  if (const char* env = std::getenv(kScEnv); env != nullptr) {
    int n;
    if (absl::SimpleAtoi(env, &n) && 0 < n && n < num_threads) {
      num_threads = n;
    }
  }
  return std::max(1, num_threads);
}

}  // namespace

tsl::thread::ThreadPool* PreprocessingThreadPool() {
  static tsl::thread::ThreadPool* pool = []() {
    const int num_threads = GetThreadPoolSize();
    DCHECK_GE(num_threads, 1);
    LOG(INFO) << "Creating thread pool for SparseCore input preprocessing: "
              << num_threads << " threads";
    auto thread_pool =
        new tsl::thread::ThreadPool(tsl::Env::Default(), kScPool, num_threads);
    return thread_pool;
  }();
  return pool;
}

}  // namespace jax_sc_embedding
