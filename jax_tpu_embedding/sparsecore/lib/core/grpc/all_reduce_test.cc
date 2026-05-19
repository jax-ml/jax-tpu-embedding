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
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "fuzztest/fuzztest.h"
#include "absl/base/log_severity.h"  // from @com_google_absl
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "fuzztest/googletest_fixture_adapter.h" // for OSS
#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/scoped_mock_log.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/synchronization/blocking_counter.h"  // from @com_google_absl
#include "include/grpcpp/client_context.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/create_channel.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/support/status.h"  // from @com_github_grpc_grpc
#include "jax_tpu_embedding/sparsecore/lib/core/all_reduce_interface.h"
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce.grpc.pb.h"
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce.pb.h" // from internal
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce_service_impl.h"
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/grpc_credentials.h"
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/minibatching_node.h"
#include "jax_tpu_embedding/sparsecore/lib/core/minibatching_test_utils.h"
#include "xla/tsl/concurrency/async_value_ref.h"  // from @xla
#include "tsl/platform/env.h"  // from @tsl
#include "tsl/platform/test.h"  // from @tsl
#include "tsl/platform/threadpool.h"  // from @tsl

namespace jax_sc_embedding {
namespace rpc {
namespace {



using ::absl_testing::IsOkAndHolds;
using ::testing::Each;

class AllReduceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    num_tasks_ = 4;
    fake_env_ = std::make_unique<FakeEnv>();
    thread_pool_ = std::make_unique<tsl::thread::ThreadPool>(
        tsl::Env::Default(), "AllReduceTest", 4);
    nodes_ =
        testing_utils::SetUpMinibatchingNodes(num_tasks_, 1, fake_env_.get());
  }

  int num_tasks_;
  std::unique_ptr<FakeEnv> fake_env_;
  std::vector<std::unique_ptr<MinibatchingNode>> nodes_;
  std::unique_ptr<tsl::thread::ThreadPool> thread_pool_;

  template <typename T>
  std::vector<absl::StatusOr<T>> RunAllReduceTest(
      int sync_key, const std::vector<T>& inputs) {
    std::vector<absl::StatusOr<T>> results(num_tasks_);
    absl::BlockingCounter barrier(num_tasks_);

    for (int i = 0; i < num_tasks_; ++i) {
      thread_pool_->Schedule([&, i]() {
        results[i] = nodes_[i]->GetAllReduceInterface()->BlockingAllReduce(
            sync_key, inputs[i]);
        barrier.DecrementCount();
      });
    }
    barrier.Wait();
    return results;
  }
};

TEST_F(AllReduceTest, BlockingAllReduceBool) {
  // Arrange
  const std::vector<bool> inputs = {true, false, true, true};
  const bool expected_result =
      absl::c_accumulate(inputs, false, std::logical_or<>());

  // Act
  std::vector<absl::StatusOr<bool>> results =
      RunAllReduceTest(/*sync_key=*/123, inputs);

  // Assert
  EXPECT_THAT(results, Each(IsOkAndHolds(expected_result)));
}

TEST_F(AllReduceTest, BlockingAllReduceUint64) {
  // Arrange
  const std::vector<uint64_t> inputs = {10, 20, 30, 40};
  const uint64_t expected_result =
      absl::c_accumulate(inputs, uint64_t{0}, std::bit_or<>());

  // Act
  std::vector<absl::StatusOr<uint64_t>> results =
      RunAllReduceTest(/*sync_key=*/456, inputs);

  // Assert
  EXPECT_THAT(results, Each(IsOkAndHolds(expected_result)));
}

using AllReduceDeathTest = AllReduceTest;

TEST_F(AllReduceDeathTest, BlockingAllReduceDataTypeMismatch) {
  // Arrange
  int sync_key = 789;
  auto run = [&]() {
    absl::BlockingCounter barrier(num_tasks_);

    // Act
    // Peer 0 will send a bool value.
    thread_pool_->Schedule([&]() {
      auto result =
          nodes_[0]->GetAllReduceInterface()->BlockingAllReduce(sync_key, true);
      barrier.DecrementCount();
    });

    // Other peers will send uint64_t values.
    for (int i = 1; i < num_tasks_; ++i) {
      thread_pool_->Schedule([&, i]() {
        auto result = nodes_[i]->GetAllReduceInterface()->BlockingAllReduce(
            sync_key, static_cast<uint64_t>(i * 10));
        barrier.DecrementCount();
      });
    }

    barrier.Wait();
  };

  // Assert
#ifdef NDEBUG
  run();
#else
  EXPECT_DEATH(run(), "Data type mismatch");
#endif
}
class AllReduceFuzzTest
    : public fuzztest::PerFuzzTestFixtureAdapter<AllReduceTest> {
 public:
  void SetUp() override {
    num_tasks_ = 4;
    nodes_ = testing_utils::SetUpMinibatchingNodes(num_tasks_, 1,
                                                   tsl::Env::Default());
    thread_pool_ = std::make_unique<tsl::thread::ThreadPool>(
        tsl::Env::Default(), "AllReduceFuzzTest", 4);
  }

  void BlockingAllReduceBoolFuzz(int sync_key_delta,
                                 const std::vector<bool>& inputs) {
    int sync_key = GetNextSyncKey(sync_key_delta);
    // Arrange
    const bool expected_result =
        absl::c_accumulate(inputs, false, std::logical_or<>());

    // Act
    std::vector<absl::StatusOr<bool>> results =
        RunAllReduceTest(sync_key, inputs);

    // Assert
    EXPECT_THAT(results, Each(IsOkAndHolds(expected_result)));
  }

  void BlockingAllReduceUint64Fuzz(int sync_key_delta,
                                   const std::vector<uint64_t>& inputs) {
    int sync_key = GetNextSyncKey(sync_key_delta);
    // Arrange
    const uint64_t expected_result =
        absl::c_accumulate(inputs, uint64_t{0}, std::bit_or<>());

    // Act
    std::vector<absl::StatusOr<uint64_t>> results =
        RunAllReduceTest(sync_key, inputs);

    // Assert
    EXPECT_THAT(results, Each(IsOkAndHolds(expected_result)));
  }

 private:
  int GetNextSyncKey(int delta) {
    int64_t delta_64 = delta;
    int next = last_sync_key_ + (std::abs(delta_64) % 1000) + 1;
    last_sync_key_ = next;
    return next;
  }
  int last_sync_key_ = 0;
};

FUZZ_TEST_F(AllReduceFuzzTest, BlockingAllReduceBoolFuzz)
    .WithDomains(fuzztest::Arbitrary<int>(),
                 fuzztest::VectorOf(fuzztest::Arbitrary<bool>()).WithSize(4));

FUZZ_TEST_F(AllReduceFuzzTest, BlockingAllReduceUint64Fuzz)
    .WithDomains(
        fuzztest::Arbitrary<int>(),
        fuzztest::VectorOf(fuzztest::Arbitrary<uint64_t>()).WithSize(4));

class MultipleLocalValuesAllReduceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    num_tasks_ = 2;
    threads_per_task_ = 2;
    total_threads_ = num_tasks_ * threads_per_task_;
    fake_env_ = std::make_unique<FakeEnv>();
    thread_pool_ = std::make_unique<tsl::thread::ThreadPool>(
        tsl::Env::Default(), "MultipleLocalValuesAllReduceTest",
        total_threads_ * 2);

    nodes_ = testing_utils::SetUpMinibatchingNodes(
        num_tasks_, threads_per_task_, fake_env_.get());
  }

  template <typename T>
  std::vector<absl::StatusOr<T>> RunAllReduceTest(
      int sync_key, const std::vector<T>& inputs) {
    std::vector<absl::StatusOr<T>> results(total_threads_);
    absl::BlockingCounter barrier(total_threads_);
    for (int thread_id = 0; thread_id < total_threads_; ++thread_id) {
      thread_pool_->Schedule([&, thread_id]() {
        int task_id = thread_id / threads_per_task_;
        results[thread_id] =
            nodes_[task_id]->GetAllReduceInterface()->BlockingAllReduce(
                sync_key, inputs[thread_id]);
        barrier.DecrementCount();
      });
    }
    barrier.Wait();
    return results;
  }

  int num_tasks_;
  int threads_per_task_;
  int total_threads_;
  std::unique_ptr<FakeEnv> fake_env_;
  std::vector<std::unique_ptr<MinibatchingNode>> nodes_;
  std::unique_ptr<tsl::thread::ThreadPool> thread_pool_;
};

TEST_F(MultipleLocalValuesAllReduceTest, BlockingAllReduceUint64) {
  // Arrange
  // 2 Tasks x 2 Threads per Task.
  const std::vector<uint64_t> inputs = {10, 20, 30, 40};
  const uint64_t expected_result =
      absl::c_accumulate(inputs, uint64_t{0}, std::bit_or<>());

  // Act
  std::vector<absl::StatusOr<uint64_t>> results =
      RunAllReduceTest(/*sync_key=*/456, inputs);

  // Assert
  EXPECT_THAT(results, Each(IsOkAndHolds(expected_result)));
}

TEST_F(MultipleLocalValuesAllReduceTest, BlockingAllReduceBool) {
  // Arrange
  // 2 Tasks x 2 Threads per Task.
  const std::vector<bool> inputs = {true, false, true, true};
  const bool expected_result =
      absl::c_accumulate(inputs, false, std::logical_or<>());

  // Act
  std::vector<absl::StatusOr<bool>> results =
      RunAllReduceTest(/*sync_key=*/123, inputs);

  // Assert
  EXPECT_THAT(results, Each(IsOkAndHolds(expected_result)));
}

TEST(WatchdogTest, WatchdogLogsWarning) {
  FakeEnv fake_env;
  int task_id = 0;
  int num_tasks = 1;
  int threads_per_task = 2;

  AllReduceServiceImpl service(task_id, num_tasks, threads_per_task, &fake_env);

  AllReduceData data;
  data.set_sync_key(123);
  data.set_bool_val(true);

  absl::ScopedMockLog mock_log;
  EXPECT_CALL(mock_log,
              Log(absl::LogSeverity::kWarning, testing::_,
                  testing::HasSubstr("Host is waiting for more than")))
      .Times(1);

  // This should call InitializeState and ScheduleWatchdog
  service.InitializeOrUpdateState(123, data);

  // Verify that closures were scheduled
  EXPECT_EQ(fake_env.num_scheduled_closures(), 1);

  mock_log.StartCapturingLogs();
  // Run them
  fake_env.RunScheduledClosures();
  mock_log.StopCapturingLogs();

  // Verify that it scheduled the NEXT one (recurring)
  EXPECT_EQ(fake_env.num_scheduled_closures(), 1);
}

class AllReduceRaceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    num_tasks_ = 3;
    fake_env_ = std::make_unique<FakeEnv>();
    ports_.reserve(num_tasks_);
    for (int i = 0; i < num_tasks_; ++i) {
      ports_.push_back(tsl::testing::PickUnusedPortOrDie());
    }

    std::vector<std::string> peer_addresses;
    peer_addresses.reserve(num_tasks_);
    for (int i = 0; i < num_tasks_; ++i) {
      peer_addresses.push_back(absl::StrCat("localhost:", ports_[i]));
    }

    nodes_.reserve(num_tasks_);
    for (int i = 0; i < num_tasks_; ++i) {
      std::vector<std::string> other_peer_addresses;
      for (int j = 0; j < num_tasks_; ++j) {
        if (i == j) continue;
        other_peer_addresses.push_back(peer_addresses[j]);
      }
      nodes_.push_back(std::make_unique<MinibatchingNode>(
          /*task_id=*/i, /*num_tasks=*/num_tasks_, other_peer_addresses,
          ports_[i],
          /*threads_per_task=*/1, fake_env_.get()));
    }

    thread_pool_ = std::make_unique<tsl::thread::ThreadPool>(
        tsl::Env::Default(), "AllReduceRaceTest", 10);
  }

  int num_tasks_;
  std::unique_ptr<FakeEnv> fake_env_;
  std::vector<int> ports_;
  std::vector<std::unique_ptr<MinibatchingNode>> nodes_;
  std::unique_ptr<tsl::thread::ThreadPool> thread_pool_;
};

TEST_F(AllReduceRaceTest, DuplicateRpcDoesNotCausePrematureConsensus) {
  int sync_key = 999;
  AllReduceInterface* interface = nodes_[0]->GetAllReduceInterface();

  tsl::AsyncValueRef<bool> result_av =
      interface->AsyncAllReduce(sync_key, /*value=*/false);
  EXPECT_FALSE(result_av.IsAvailable());

  std::string target = absl::StrCat("localhost:", ports_[0]);
  std::shared_ptr<::grpc::Channel> channel =
      ::grpc::CreateChannel(target, GetDefaultChannelCredentials());
  std::unique_ptr<AllReduceGrpcService::Stub> stub =
      AllReduceGrpcService::NewStub(channel);

  // Send contribution from Task 1 (false).
  {
    ::grpc::ClientContext context;
    AllReduceData request;
    request.set_sync_key(sync_key);
    request.set_src_rank(1);
    request.set_bool_val(false);
    AllReduceResponse response;
    ::grpc::Status status = stub->ContributeData(&context, request, &response);
    ASSERT_TRUE(status.ok());
  }

  EXPECT_FALSE(result_av.IsAvailable());

  // Send DUPLICATE contribution from Task 1 (false).
  {
    ::grpc::ClientContext context;
    AllReduceData request;
    request.set_sync_key(sync_key);
    request.set_src_rank(1);
    request.set_bool_val(false);
    AllReduceResponse response;
    ::grpc::Status status = stub->ContributeData(&context, request, &response);
    ASSERT_TRUE(status.ok());
  }

  // If the bug is present, result_av will now be ready (premature consensus).
  // We assert it is NOT ready.
  EXPECT_FALSE(result_av.IsAvailable());

  // Send contribution from Task 2 (true).
  {
    ::grpc::ClientContext context;
    AllReduceData request;
    request.set_sync_key(sync_key);
    request.set_src_rank(2);
    request.set_bool_val(true);
    AllReduceResponse response;
    ::grpc::Status status = stub->ContributeData(&context, request, &response);
    ASSERT_TRUE(status.ok());
  }

  tsl::BlockUntilReady(result_av);
  ASSERT_TRUE(result_av.IsAvailable());
  EXPECT_TRUE(result_av.get());
}

TEST_F(AllReduceRaceTest, LateRpcIsIgnored) {
  int sync_key = 1000;
  std::vector<absl::StatusOr<bool>> results(num_tasks_);
  absl::BlockingCounter barrier(num_tasks_);

  for (int i = 0; i < num_tasks_; ++i) {
    thread_pool_->Schedule([&, i]() {
      results[i] = nodes_[i]->GetAllReduceInterface()->BlockingAllReduce(
          sync_key, /*value=*/false);
      barrier.DecrementCount();
    });
  }
  barrier.Wait();

  for (int i = 0; i < num_tasks_; ++i) {
    ASSERT_TRUE(results[i].ok());
    EXPECT_FALSE(*results[i]);
  }

  EXPECT_EQ(nodes_[0]->GetActiveStatesCount(), 0);

  std::string target = absl::StrCat("localhost:", ports_[0]);
  std::shared_ptr<::grpc::Channel> channel =
      ::grpc::CreateChannel(target, GetDefaultChannelCredentials());
  std::unique_ptr<AllReduceGrpcService::Stub> stub =
      AllReduceGrpcService::NewStub(channel);

  {
    ::grpc::ClientContext context;
    AllReduceData request;
    request.set_sync_key(sync_key);
    request.set_src_rank(1);
    request.set_bool_val(false);
    AllReduceResponse response;
    ::grpc::Status status = stub->ContributeData(&context, request, &response);
    ASSERT_TRUE(status.ok());
  }

  EXPECT_EQ(nodes_[0]->GetActiveStatesCount(), 0);
}

}  // namespace
}  // namespace rpc
}  // namespace jax_sc_embedding
