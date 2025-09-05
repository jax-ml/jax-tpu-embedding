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
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "fuzztest/fuzztest.h"
#include "fuzztest/googletest_fixture_adapter.h" // for OSS
#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/synchronization/blocking_counter.h"  // from @com_google_absl
#include "include/grpcpp/security/server_credentials.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/server.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/server_builder.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/support/status.h"  // from @com_github_grpc_grpc
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce.pb.h" // from internal
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce_interface.h"
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce_service_impl.h"
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/grpc_credentials.h"
#include "tsl/platform/env.h"  // from @tsl
#include "tsl/platform/test.h"  // from @tsl
#include "tsl/platform/threadpool.h"  // from @tsl

namespace jax_sc_embedding {
namespace rpc {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::Each;
using ::testing::HasSubstr;

struct Peer {
  int task_id;
  std::string address;
  std::unique_ptr<AllReduceServiceImpl> service;
  std::unique_ptr<::grpc::Server> server;
  std::unique_ptr<GrpcAllReduceInterface> all_reduce_interface;
};

class AllReduceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    num_tasks_ = 4;
    thread_pool_ = std::make_unique<tsl::thread::ThreadPool>(
        tsl::Env::Default(), "AllReduceTest", 4);
    for (int i = 0; i < num_tasks_; ++i) {
      peers_.emplace_back(CreatePeer(i, num_tasks_));
    }

    for (int i = 0; i < num_tasks_; ++i) {
      std::vector<std::string> peer_addresses;
      peer_addresses.reserve(num_tasks_ - 1);
      for (int j = 0; j < num_tasks_; ++j) {
        if (j != i) {
          peer_addresses.push_back(peers_[j].address);
        }
      }
      peers_[i].all_reduce_interface = std::make_unique<GrpcAllReduceInterface>(
          peer_addresses, i, num_tasks_, 0, peers_[i].service.get());
      peers_[i].all_reduce_interface->SetUp();
    }
  }

  void TearDown() override {
    for (auto& peer : peers_) {
      if (peer.server) {
        peer.server->Shutdown();
      }
    }
    for (auto& peer : peers_) {
      if (peer.server) {
        peer.server->Wait();
      }
    }
  }

  Peer CreatePeer(int task_id, int num_tasks, int threads_per_task = 1) {
    Peer peer;
    peer.task_id = task_id;
    const int port = tsl::testing::PickUnusedPortOrDie();
    peer.address = absl::StrCat("localhost:", port);

    peer.service = std::make_unique<AllReduceServiceImpl>(task_id, num_tasks,
                                                          threads_per_task);

    ::grpc::ServerBuilder builder;
    builder.AddListeningPort(peer.address, GetDefaultServerCredentials());
    builder.RegisterService(peer.service.get());
    peer.server = builder.BuildAndStart();
    return peer;
  }

  int num_tasks_;
  std::vector<Peer> peers_;
  std::unique_ptr<tsl::thread::ThreadPool> thread_pool_;

  template <typename T>
  std::vector<absl::StatusOr<T>> RunAllReduceTest(
      int sync_key, const std::vector<T>& inputs) {
    std::vector<absl::StatusOr<T>> results(num_tasks_);
    absl::BlockingCounter barrier(num_tasks_);

    for (int i = 0; i < num_tasks_; ++i) {
      thread_pool_->Schedule([&, i]() {
        results[i] = peers_[i].all_reduce_interface->BlockingAllReduce(
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
  std::vector<bool> inputs = {true, false, true, true};
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
  std::vector<uint64_t> inputs = {10, 20, 30, 40};
  const uint64_t expected_result =
      absl::c_accumulate(inputs, uint64_t{0}, std::bit_or<>());

  // Act
  std::vector<absl::StatusOr<uint64_t>> results =
      RunAllReduceTest(/*sync_key=*/456, inputs);

  // Assert
  EXPECT_THAT(results, Each(IsOkAndHolds(expected_result)));
}

TEST_F(AllReduceTest, BlockingAllReduceDataTypeMismatch) {
  // Arrange
  int sync_key = 789;
  std::vector<absl::Status> statuses(num_tasks_);
  absl::BlockingCounter barrier(num_tasks_);

  // Act
  // Peer 0 will send a bool value.
  thread_pool_->Schedule([&]() {
    statuses[0] = peers_[0]
                      .all_reduce_interface->BlockingAllReduce(sync_key, true)
                      .status();
    barrier.DecrementCount();
  });

  // Other peers will send uint64_t values.
  for (int i = 1; i < num_tasks_; ++i) {
    thread_pool_->Schedule([&, i]() {
      statuses[i] =
          peers_[i]
              .all_reduce_interface
              ->BlockingAllReduce(sync_key, static_cast<uint64_t>(i * 10))
              .status();
      barrier.DecrementCount();
    });
  }

  barrier.Wait();

  // Assert
  // All peers should receive an InvalidArgumentError because of the type
  // mismatch.
  EXPECT_THAT(statuses, Each(StatusIs(grpc::StatusCode::INVALID_ARGUMENT)));
}

TEST_F(AllReduceTest, BlockingAllReduceNoStubs) {
  // Create an interface with no peer addresses (num_tasks must be 1).
  AllReduceServiceImpl local_service(/*task_id=*/0, /*num_tasks=*/1);
  GrpcAllReduceInterface interface(/*peer_addresses=*/{}, /*task_id=*/0,
                                   /*num_tasks=*/1, /*all_reduce_port=*/0,
                                   &local_service);
  interface.SetUp();  // This will result in an empty `stubs_` vector.

  // Expect a FailedPreconditionError when calling BlockingAllReduce.
  EXPECT_THAT(interface.BlockingAllReduce(/*sync_key=*/1,
                                          /*minibatching_required=*/true),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("No gRPC stubs available.")));
  EXPECT_THAT(interface.BlockingAllReduce(/*sync_key=*/2,
                                          /*minibatching_split=*/uint64_t{100}),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("No gRPC stubs available.")));
}

class AllReduceFuzzTest
    : public fuzztest::PerFuzzTestFixtureAdapter<AllReduceTest> {
 public:
  void BlockingAllReduceBoolFuzz(int sync_key,
                                 const std::vector<bool>& inputs) {
    // Arrange
    const bool expected_result =
        absl::c_accumulate(inputs, false, std::logical_or<>());

    // Act
    std::vector<absl::StatusOr<bool>> results =
        RunAllReduceTest(sync_key, inputs);

    // Assert
    EXPECT_THAT(results, Each(IsOkAndHolds(expected_result)));
  }

  void BlockingAllReduceUint64Fuzz(int sync_key,
                                   const std::vector<uint64_t>& inputs) {
    // Arrange
    const uint64_t expected_result =
        absl::c_accumulate(inputs, uint64_t{0}, std::bit_or<>());

    // Act
    std::vector<absl::StatusOr<uint64_t>> results =
        RunAllReduceTest(sync_key, inputs);

    // Assert
    EXPECT_THAT(results, Each(IsOkAndHolds(expected_result)));
  }
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
    thread_pool_ = std::make_unique<tsl::thread::ThreadPool>(
        tsl::Env::Default(), "MultipleLocalValuesAllReduceTest",
        total_threads_ * 2);

    // Setup task services and servers.
    task_services_.resize(num_tasks_);
    task_servers_.resize(num_tasks_);
    std::vector<std::string> task_address(num_tasks_);
    for (int i = 0; i < num_tasks_; ++i) {
      const int port = tsl::testing::PickUnusedPortOrDie();
      task_address[i] = absl::StrCat("localhost:", port);
      task_services_[i] = std::make_unique<AllReduceServiceImpl>(
          i, num_tasks_, threads_per_task_);
      ::grpc::ServerBuilder builder;
      builder.AddListeningPort(task_address[i], GetDefaultServerCredentials());
      builder.RegisterService(task_services_[i].get());
      task_servers_[i] = builder.BuildAndStart();
    }

    // Setup task interfaces.
    task_interfaces_.resize(num_tasks_);
    for (int task_id = 0; task_id < num_tasks_; ++task_id) {
      std::vector<std::string> peer_task_addresses;
      for (int j = 0; j < num_tasks_; ++j) {
        if (j != task_id) {
          peer_task_addresses.push_back(task_address[j]);
        }
      }
      task_interfaces_[task_id] = std::make_unique<GrpcAllReduceInterface>(
          peer_task_addresses, task_id, num_tasks_, /*all_reduce_port=*/0,
          task_services_[task_id].get(), threads_per_task_);
      task_interfaces_[task_id]->SetUp();
    }
  }

  void TearDown() override {
    for (auto& server : task_servers_) {
      if (server) {
        server->Shutdown();
      }
    }
    for (auto& server : task_servers_) {
      if (server) {
        server->Wait();
      }
    }
  }

  template <typename T>
  std::vector<absl::StatusOr<T>> RunAllReduceTest(
      int sync_key, const std::vector<T>& inputs) {
    std::vector<absl::StatusOr<T>> results(total_threads_);
    absl::BlockingCounter barrier(total_threads_);
    for (int thread_id = 0; thread_id < total_threads_; ++thread_id) {
      thread_pool_->Schedule([&, thread_id]() {
        int task_id = thread_id / threads_per_task_;
        results[thread_id] = task_interfaces_[task_id]->BlockingAllReduce(
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
  std::vector<std::unique_ptr<AllReduceServiceImpl>> task_services_;
  std::vector<std::unique_ptr<::grpc::Server>> task_servers_;
  std::vector<std::unique_ptr<GrpcAllReduceInterface>> task_interfaces_;
  std::unique_ptr<tsl::thread::ThreadPool> thread_pool_;
};

TEST_F(MultipleLocalValuesAllReduceTest, BlockingAllReduceUint64) {
  // Arrange
  // 2 Tasks x 2 Threads per Task.
  std::vector<uint64_t> inputs = {10, 20, 30, 40};
  const uint64_t expected_result =
      absl::c_accumulate(inputs, uint64_t{0}, std::bit_or<>());

  // Act
  std::vector<absl::StatusOr<uint64_t>> results =
      RunAllReduceTest(/*sync_key=*/456, inputs);

  // Assert
  EXPECT_THAT(results, Each(IsOkAndHolds(expected_result)));
}

}  // namespace
}  // namespace rpc
}  // namespace jax_sc_embedding
