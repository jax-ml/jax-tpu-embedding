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
#include "jax_tpu_embedding/sparsecore/lib/core/fake_all_reduce.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "tsl/platform/env.h"  // from @tsl
#include "tsl/platform/threadpool.h"  // from @tsl

namespace jax_sc_embedding {
namespace testing_utils {
namespace {

using ::testing::Each;

TEST(FakeAllReduceTest, SingleHostSingleGroup) {
  // Arrange
  AllReduceCallback all_reduce = CreateFakeAllReduceCallback(1);
  const int sync_key = 0;

  // Act & Assert
  EXPECT_EQ(all_reduce(sync_key, 1ULL), 1ULL);
  EXPECT_EQ(all_reduce(sync_key, 0ULL), 0ULL);
}

TEST(FakeAllReduceTest, SingleHostMultiGroup) {
  // Arrange
  AllReduceCallback all_reduce = CreateFakeAllReduceCallback(1);

  // Act & Assert
  EXPECT_EQ(all_reduce(0, 1ULL), 1ULL);
  EXPECT_EQ(all_reduce(1, 0ULL), 0ULL);
  EXPECT_EQ(all_reduce(2, 1ULL), 1ULL);
}

TEST(FakeAllReduceTest, MultiHostAllOnes) {
  // Arrange
  const int host_count = 4;
  AllReduceCallback all_reduce = CreateFakeAllReduceCallback(host_count);
  const int sync_key = 0;
  std::vector<uint64_t> results(host_count);

  // Act
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "test_pool", host_count);
    for (int i = 0; i < host_count; ++i) {
      pool.Schedule([&, i]() { results[i] = all_reduce(sync_key, 1ULL); });
    }
    // Destructor waits for completion.
  }

  // Assert
  EXPECT_THAT(results, Each(1ULL));
}

TEST(FakeAllReduceTest, MultiHostMixedValues) {
  // Arrange
  const int host_count = 4;
  AllReduceCallback all_reduce = CreateFakeAllReduceCallback(host_count);
  const int sync_key = 0;
  std::vector<uint64_t> results(host_count);

  // Act
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "test_pool", host_count);
    for (int i = 0; i < host_count; ++i) {
      pool.Schedule([&, i]() {
        results[i] = all_reduce(sync_key, i % 2 == 0 ? 1ULL : 0ULL);
      });
    }
    // Destructor waits for completion.
  }

  // Assert
  EXPECT_THAT(results, Each(1ULL));
}

TEST(FakeAllReduceTest, MultiHostAllZeros) {
  // Arrange
  const int host_count = 4;
  AllReduceCallback all_reduce = CreateFakeAllReduceCallback(host_count);
  const int sync_key = 0;
  std::vector<uint64_t> results(host_count);

  // Act
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "test_pool", host_count);
    for (int i = 0; i < host_count; ++i) {
      pool.Schedule([&, i]() { results[i] = all_reduce(sync_key, 0ULL); });
    }
    // Destructor waits for completion.
  }

  // Assert
  EXPECT_THAT(results, Each(0ULL));
}

TEST(FakeAllReduceTest, MultiHostMultiGroup) {
  // Arrange
  const int host_count = 2;
  AllReduceCallback all_reduce = CreateFakeAllReduceCallback(host_count);
  std::vector<uint64_t> results(host_count * 2);

  // Act
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "test_pool",
                                 host_count * 2);

    // Group 0
    pool.Schedule([&]() { results[0] = all_reduce(0, 1ULL); });
    pool.Schedule([&]() { results[1] = all_reduce(0, 0ULL); });

    // Group 1
    pool.Schedule([&]() { results[2] = all_reduce(1, 0ULL); });
    pool.Schedule([&]() { results[3] = all_reduce(1, 0ULL); });
    // Destructor waits for completion.
  }

  // Assert
  EXPECT_EQ(results[0], 1ULL);
  EXPECT_EQ(results[1], 1ULL);
  EXPECT_EQ(results[2], 0ULL);
  EXPECT_EQ(results[3], 0ULL);
}

}  // namespace
}  // namespace testing_utils
}  // namespace jax_sc_embedding
