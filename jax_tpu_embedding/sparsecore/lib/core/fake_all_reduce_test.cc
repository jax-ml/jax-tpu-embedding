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
#include <functional>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/all_reduce_interface.h"
#include "tsl/platform/env.h"  // from @tsl
#include "tsl/platform/status_matchers.h"  // from @tsl  // IWYU pragma: keep (OSS, b/438618768)
#include "tsl/platform/statusor.h"  // from @tsl
#include "tsl/platform/threadpool.h"  // from @tsl

namespace jax_sc_embedding {
namespace testing_utils {
namespace {

using ::testing::Each;
using ::tsl::testing::IsOkAndHolds;

// Helper function to run BlockingAllReduce across multiple simulated hosts.
// The `host_values` array provides the value passed to BlockingAllReduce for
// each host index.
std::vector<uint64_t> RunBlockingAllReduceOnMultiHost(
    int sync_key, absl::Span<const uint64_t> host_values) {
  int host_count = host_values.size();
  std::unique_ptr<AllReduceInterface> all_reduce =
      std::make_unique<FakeAllReduce>(host_count);
  std::vector<uint64_t> results(host_count);

  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "test_pool", host_count);
    for (int i = 0; i < host_count; ++i) {
      pool.Schedule([&, i]() {
        TF_ASSERT_OK_AND_ASSIGN(results[i], all_reduce->BlockingAllReduce(
                                                sync_key, host_values[i]));
      });
    }
    // Destructor waits for completion.
  }
  return results;
}

TEST(FakeAllReduceTest, SingleHostSingleGroup) {
  // Arrange
  FakeAllReduce all_reduce(1);
  const int sync_key = 0;

  // Act & Assert
  EXPECT_THAT(all_reduce.BlockingAllReduce(sync_key, uint64_t{1}),
              IsOkAndHolds(uint64_t{1}));
  EXPECT_THAT(all_reduce.BlockingAllReduce(sync_key, uint64_t{0}),
              IsOkAndHolds(uint64_t{0}));
}

TEST(FakeAllReduceTest, SingleHostMultiGroup) {
  // Arrange
  FakeAllReduce all_reduce(1);

  // Act & Assert
  EXPECT_THAT(all_reduce.BlockingAllReduce(0, uint64_t{1}),
              IsOkAndHolds(uint64_t{1}));
  EXPECT_THAT(all_reduce.BlockingAllReduce(1, uint64_t{0}),
              IsOkAndHolds(uint64_t{0}));
  EXPECT_THAT(all_reduce.BlockingAllReduce(2, uint64_t{1}),
              IsOkAndHolds(uint64_t{1}));
}

TEST(FakeAllReduceTest, MultiHostAllOnes) {
  // Arrange
  const int sync_key = 0;
  const std::vector<uint64_t> host_values = {1, 1, 1, 1};

  // Act
  std::vector<uint64_t> results =
      RunBlockingAllReduceOnMultiHost(sync_key, host_values);

  // Assert
  EXPECT_THAT(results, Each(uint64_t{1}));
}

TEST(FakeAllReduceTest, MultiHostMixedValues) {
  // Arrange
  const int sync_key = 0;
  const std::vector<uint64_t> host_values = {1, 0, 1, 0};

  // Act
  std::vector<uint64_t> results =
      RunBlockingAllReduceOnMultiHost(sync_key, host_values);

  // Assert
  EXPECT_THAT(results, Each(uint64_t{1}));
}

TEST(FakeAllReduceTest, MultiHostAllZeros) {
  // Arrange
  const int sync_key = 0;
  const std::vector<uint64_t> host_values = {0, 0, 0, 0};

  // Act
  std::vector<uint64_t> results =
      RunBlockingAllReduceOnMultiHost(sync_key, host_values);

  // Assert
  EXPECT_THAT(results, Each(uint64_t{0}));
}

TEST(FakeAllReduceTest, MultiHostMultiGroup) {
  // Arrange
  const int host_count = 2;
  std::unique_ptr<AllReduceInterface> all_reduce =
      std::make_unique<FakeAllReduce>(host_count);
  std::vector<uint64_t> results(host_count * 2);

  // Act
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "test_pool",
                                 host_count * 2);

    // Group 0
    pool.Schedule([&]() {
      TF_ASSERT_OK_AND_ASSIGN(results[0],
                              all_reduce->BlockingAllReduce(0, uint64_t{1}));
    });
    pool.Schedule([&]() {
      TF_ASSERT_OK_AND_ASSIGN(results[1],
                              all_reduce->BlockingAllReduce(0, uint64_t{0}));
    });

    // Group 1
    pool.Schedule([&]() {
      TF_ASSERT_OK_AND_ASSIGN(results[2],
                              all_reduce->BlockingAllReduce(1, uint64_t{0}));
    });
    pool.Schedule([&]() {
      TF_ASSERT_OK_AND_ASSIGN(results[3],
                              all_reduce->BlockingAllReduce(1, uint64_t{0}));
    });
    // Destructor waits for completion.
  }

  // Assert
  EXPECT_THAT(results[0], uint64_t{1});
  EXPECT_THAT(results[1], uint64_t{1});
  EXPECT_THAT(results[2], uint64_t{0});
  EXPECT_THAT(results[3], uint64_t{0});
}

TEST(FakeAllReduceTest, TestBoolValue) {
  // Arrange
  const int host_count = 2;
  std::unique_ptr<AllReduceInterface> all_reduce =
      std::make_unique<FakeAllReduce>(host_count);
  bool result[host_count];

  // Act
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "test_pool", host_count);

    // Group 0
    pool.Schedule([&]() {
      TF_ASSERT_OK_AND_ASSIGN(result[0],
                              all_reduce->BlockingAllReduce(0, true));
    });
    pool.Schedule([&]() {
      TF_ASSERT_OK_AND_ASSIGN(result[1],
                              all_reduce->BlockingAllReduce(0, false));
    });
    // Destructor waits for completion.
  }

  // Assert
  EXPECT_THAT(result, Each(true));
}

}  // namespace
}  // namespace testing_utils
}  // namespace jax_sc_embedding
