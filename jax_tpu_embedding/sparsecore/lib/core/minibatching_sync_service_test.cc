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

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/minibatching_node.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "jax_tpu_embedding/sparsecore/lib/core/minibatching_test_utils.h"
#include "tsl/platform/env.h"  // from @tsl
#include "tsl/platform/threadpool.h"  // from @tsl

namespace jax_sc_embedding {
namespace {

template <typename T>
struct SyncTestCase {
  std::string test_name;
  int num_hosts;
  int num_tables;
  std::vector<std::vector<T>> host_table_values;
  T expected_value;
};

template <typename T>
void RunMinibatchingSync(const SyncTestCase<T>& test_case, int sync_key,
                         std::vector<T>& results, absl::Status& shared_status) {
  const int num_hosts = test_case.num_hosts;
  const int num_tables = test_case.num_tables;

  auto nodes = testing_utils::SetUpMinibatchingNodes(num_hosts);
  absl::Mutex results_mutex;

  std::vector<std::unique_ptr<MinibatchingSyncService<T>>> services;
  for (int i = 0; i < num_hosts; ++i) {
    services.push_back(
        std::make_unique<MinibatchingSyncService<T>>(num_tables));
  }

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "TestPool",
                               num_hosts * num_tables);

  for (int host_id = 0; host_id < num_hosts; ++host_id) {
    for (int table_id = 0; table_id < num_tables; ++table_id) {
      pool.Schedule([&, host_id, table_id]() {
        auto status_or_value = services[host_id]->SyncValue(
            test_case.host_table_values[host_id][table_id], sync_key,
            nodes[host_id]->GetAllReduceInterface());
        absl::MutexLock lock(&results_mutex);  // NOLINT (b/438618768)
        if (!status_or_value.ok()) {
          shared_status.Update(status_or_value.status());
        } else if (table_id == 0) {
          results[host_id] = status_or_value.value();
        }
      });
    }
  }
}

class SyncRequiredTest : public ::testing::TestWithParam<SyncTestCase<bool>> {};

TEST_P(SyncRequiredTest, SyncMinibatchingRequired) {
  // Arrange
  const auto& test_case = this->GetParam();
  std::vector<bool> results(test_case.num_hosts);
  absl::Status shared_status = absl::OkStatus();

  // Act
  RunMinibatchingSync(test_case, /*sync_key=*/0, results, shared_status);

  // Assert
  EXPECT_TRUE(shared_status.ok());

  EXPECT_THAT(results, testing::Each(testing::Eq(test_case.expected_value)));
}

INSTANTIATE_TEST_SUITE_P(
    SyncRequiredTests, SyncRequiredTest,
    testing::ValuesIn<SyncTestCase<bool>>(
        {{.test_name = "SingleHost_None",
          .num_hosts = 1,
          .num_tables = 3,
          .host_table_values = {{false, false, false}},
          .expected_value = false},
         {.test_name = "SingleHost_Some",
          .num_hosts = 1,
          .num_tables = 3,
          .host_table_values = {{false, true, false}},
          .expected_value = true},
         {.test_name = "SingleHost_All",
          .num_hosts = 1,
          .num_tables = 3,
          .host_table_values = {{true, true, true}},
          .expected_value = true},
         {.test_name = "MultiHost_None",
          .num_hosts = 2,
          .num_tables = 3,
          .host_table_values = {{false, false, false}, {false, false, false}},
          .expected_value = false},
         {.test_name = "MultiHost_OneHostSome",
          .num_hosts = 2,
          .num_tables = 3,
          .host_table_values = {{false, true, false}, {false, false, false}},
          .expected_value = true},
         {.test_name = "MultiHost_AllHostsAll",
          .num_hosts = 2,
          .num_tables = 3,
          .host_table_values = {{true, true, true}, {true, true, true}},
          .expected_value = true}}),
    [](const testing::TestParamInfo<SyncRequiredTest::ParamType>& info) {
      return info.param.test_name;
    });

class SyncSplitTest
    : public ::testing::TestWithParam<SyncTestCase<MinibatchingSplit>> {};

TEST_P(SyncSplitTest, SyncMinibatchingSplit) {
  // Arrange
  const auto& test_case = this->GetParam();
  std::vector<MinibatchingSplit> results(test_case.num_hosts);
  absl::Status shared_status = absl::OkStatus();

  // Act
  RunMinibatchingSync(test_case, /*sync_key=*/1, results, shared_status);

  // Assert
  EXPECT_TRUE(shared_status.ok());

  for (int host_id = 0; host_id < test_case.num_hosts; ++host_id) {
    EXPECT_EQ(results[host_id], test_case.expected_value) << "Host " << host_id;
  }
}

INSTANTIATE_TEST_SUITE_P(
    SyncSplitTests, SyncSplitTest,
    testing::ValuesIn<SyncTestCase<MinibatchingSplit>>(
        {{.test_name = "SingleHost_Simple",
          .num_hosts = 1,
          .num_tables = 3,
          .host_table_values = {{0b101, 0b010, 0b100}},
          .expected_value = 0b111},
         {.test_name = "MultiHost_Disjoint",
          .num_hosts = 2,
          .num_tables = 2,
          .host_table_values = {{0b00001100, 0b00000011},
                                {0b11000000, 0b00110000}},
          .expected_value = 0b11111111},
         {.test_name = "MultiHost_Overlap",
          .num_hosts = 2,
          .num_tables = 3,
          .host_table_values = {{0b01010100, 0b01000100, 0b00010100},
                                {0b00101010, 0b00100010, 0b00001010}},
          .expected_value = 0b01111110},
         {.test_name = "MultiHost_Varied",
          .num_hosts = 3,
          .num_tables = 2,
          .host_table_values = {{0b00000011, 0b00000010},
                                {0b00001100, 0b00000100},
                                {0b00110000, 0b00100000}},
          .expected_value = 0b00111111},
         {.test_name = "MultiHost_NotAllBitsSet",
          .num_hosts = 2,
          .num_tables = 2,
          .host_table_values = {{0b00001010, 0b00001000},
                                {0b00000101, 0b00000001}},
          .expected_value = 0b00001111}}),
    [](const testing::TestParamInfo<SyncSplitTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace jax_sc_embedding
