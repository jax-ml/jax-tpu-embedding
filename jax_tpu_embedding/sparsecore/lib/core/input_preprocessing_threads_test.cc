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

#include <gtest/gtest.h>
#include "tsl/platform/threadpool.h"  // from @tsl

namespace jax_sc_embedding {
namespace {

TEST(InputPreprocessingThreadsTest, CreateThreadPool) {
  tsl::thread::ThreadPool* pool = PreprocessingThreadPool();
  EXPECT_NE(pool, nullptr);
}

}  // namespace
}  // namespace jax_sc_embedding
