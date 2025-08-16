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
#include "jax_tpu_embedding/sparsecore/lib/core/minibatching_sync_service.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/synchronization/blocking_counter.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/fake_all_reduce.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_threads.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "tsl/platform/env.h"  // from @tsl
#include "tsl/platform/threadpool.h"  // from @tsl

namespace jax_sc_embedding {
namespace {

// Helper to get a static thread pool shared across tests.
static tsl::thread::ThreadPool* MultiHostPool() {
  static tsl::thread::ThreadPool* pool = []() {
    auto thread_pool = new tsl::thread::ThreadPool(
        tsl::Env::Default(), "MultiHostPool", /*num_threads=*/8);
    return thread_pool;
  }();
  return pool;
}

struct SyncRequiredTestCase {
  std::string test_name;
  int num_hosts;
  int num_tables;
  std::vector<std::vector<bool>> host_table_requires;
  bool expected_required;
};

class SyncRequiredTest : public ::testing::TestWithParam<SyncRequiredTestCase> {
};

TEST_P(SyncRequiredTest, SyncMinibatchingRequired) {
  const auto& test_case = GetParam();
  const int num_hosts = test_case.num_hosts;
  const int num_tables = test_case.num_tables;

  auto all_reduce_callback =
      testing_utils::CreateFakeAllReduceCallback(num_hosts);
  std::vector<bool> results(num_hosts);
  absl::Mutex results_mutex;
  absl::BlockingCounter host_counter(num_hosts);
  absl::Status shared_status = absl::OkStatus();

  for (int host_id = 0; host_id < num_hosts; ++host_id) {
    MultiHostPool()->Schedule([&, host_id]() {
      MinibatchingSyncService<bool> service(num_tables);
      absl::BlockingCounter table_counter(num_tables);
      bool host_result = false;

      for (int table_id = 0; table_id < num_tables; ++table_id) {
        PreprocessingThreadPool()->Schedule([&, host_id, table_id]() {
          auto status_or_required = service.SyncValue(
              test_case.host_table_requires[host_id][table_id],
              /*batch_number=*/0, all_reduce_callback);
          if (!status_or_required.ok()) {
            absl::MutexLock lock(&results_mutex);  // NOLINT (b/438618768)
            shared_status = status_or_required.status();
            return;
          } else {
            if (table_id == 0) {
              host_result = status_or_required.value();
            }
          }
          table_counter.DecrementCount();
        });
      }
      table_counter.Wait();
      {
        absl::MutexLock lock(&results_mutex);  // NOLINT (b/438618768)
        results[host_id] = host_result;
      }
      host_counter.DecrementCount();
    });
  }
  host_counter.Wait();
  EXPECT_TRUE(shared_status.ok());

  for (int host_id = 0; host_id < num_hosts; ++host_id) {
    EXPECT_EQ(results[host_id], test_case.expected_required)
        << "Host " << host_id;
  }
}

INSTANTIATE_TEST_SUITE_P(
    SyncRequiredTests, SyncRequiredTest,
    testing::ValuesIn<SyncRequiredTestCase>(
        {{.test_name = "SingleHost_None",
          .num_hosts = 1,
          .num_tables = 3,
          .host_table_requires = {{false, false, false}},
          .expected_required = false},
         {.test_name = "SingleHost_Some",
          .num_hosts = 1,
          .num_tables = 3,
          .host_table_requires = {{false, true, false}},
          .expected_required = true},
         {.test_name = "SingleHost_All",
          .num_hosts = 1,
          .num_tables = 3,
          .host_table_requires = {{true, true, true}},
          .expected_required = true},
         {.test_name = "MultiHost_None",
          .num_hosts = 2,
          .num_tables = 3,
          .host_table_requires = {{false, false, false}, {false, false, false}},
          .expected_required = false},
         {.test_name = "MultiHost_OneHostSome",
          .num_hosts = 2,
          .num_tables = 3,
          .host_table_requires = {{false, true, false}, {false, false, false}},
          .expected_required = true},
         {.test_name = "MultiHost_AllHostsAll",
          .num_hosts = 2,
          .num_tables = 3,
          .host_table_requires = {{true, true, true}, {true, true, true}},
          .expected_required = true}}),
    [](const testing::TestParamInfo<SyncRequiredTest::ParamType>& info) {
      return info.param.test_name;
    });

struct SyncSplitTestCase {
  std::string test_name;
  int num_hosts;
  int num_tables;
  std::vector<std::vector<MinibatchingSplit>> host_table_splits;
  MinibatchingSplit expected_split;
};

class SyncSplitTest : public ::testing::TestWithParam<SyncSplitTestCase> {};

TEST_P(SyncSplitTest, SyncMinibatchingSplit) {
  const auto& test_case = GetParam();
  const int num_hosts = test_case.num_hosts;
  const int num_tables = test_case.num_tables;

  auto all_reduce_callback =
      testing_utils::CreateFakeAllReduceCallback(num_hosts);
  std::vector<MinibatchingSplit> results(num_hosts);
  absl::Mutex results_mutex;
  absl::BlockingCounter host_counter(num_hosts);
  absl::Status shared_status;

  for (int host_id = 0; host_id < num_hosts; ++host_id) {
    MultiHostPool()->Schedule([&, host_id]() {
      MinibatchingSyncService<MinibatchingSplit> service(num_tables);
      absl::BlockingCounter table_counter(num_tables);
      MinibatchingSplit host_result = 0;

      for (int table_id = 0; table_id < num_tables; ++table_id) {
        PreprocessingThreadPool()->Schedule([&, host_id, table_id]() {
          auto status_or_split =
              service.SyncValue(test_case.host_table_splits[host_id][table_id],
                                /*batch_number=*/1, all_reduce_callback);
          if (!status_or_split.ok()) {
            absl::MutexLock lock(&results_mutex);  // NOLINT (b/438618768)
            shared_status = status_or_split.status();
            return;
          } else {
            if (table_id == 0) {
              host_result = status_or_split.value();
            }
          }
          table_counter.DecrementCount();
        });
      }
      table_counter.Wait();
      {
        absl::MutexLock lock(&results_mutex);  // NOLINT (b/438618768)
        results[host_id] = host_result;
      }
      host_counter.DecrementCount();
    });
  }
  host_counter.Wait();
  EXPECT_TRUE(shared_status.ok());

  for (int host_id = 0; host_id < num_hosts; ++host_id) {
    EXPECT_EQ(results[host_id], test_case.expected_split) << "Host " << host_id;
  }
}

INSTANTIATE_TEST_SUITE_P(
    SyncSplitTests, SyncSplitTest,
    testing::ValuesIn<SyncSplitTestCase>(
        {{.test_name = "SingleHost_Simple",
          .num_hosts = 1,
          .num_tables = 3,
          .host_table_splits = {{0b101, 0b010, 0b100}},
          .expected_split = 0b111},
         {.test_name = "MultiHost_Disjoint",
          .num_hosts = 2,
          .num_tables = 2,
          .host_table_splits = {{0b00001100, 0b00000011},
                                {0b11000000, 0b00110000}},
          .expected_split = 0b11111111},
         {.test_name = "MultiHost_Overlap",
          .num_hosts = 2,
          .num_tables = 3,
          .host_table_splits = {{0b01010100, 0b01000100, 0b00010100},
                                {0b00101010, 0b00100010, 0b00001010}},
          .expected_split = 0b01111110},
         {.test_name = "MultiHost_Varied",
          .num_hosts = 3,
          .num_tables = 2,
          .host_table_splits = {{0b00000011, 0b00000010},
                                {0b00001100, 0b00000100},
                                {0b00110000, 0b00100000}},
          .expected_split = 0b00111111},
         {.test_name = "MultiHost_NotAllBitsSet",
          .num_hosts = 2,
          .num_tables = 2,
          .host_table_splits = {{0b00001010, 0b00001000},
                                {0b00000101, 0b00000001}},
          .expected_split = 0b00001111}}),
    [](const testing::TestParamInfo<SyncSplitTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace jax_sc_embedding
