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
#include <memory>
#include <string>
#include <vector>

#include "net/grpc/public/include/grpcpp/channel_credentials_google.h"
#include "net/grpc/public/include/grpcpp/server_credentials_google.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "fuzztest/fuzztest.h"
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/synchronization/blocking_counter.h"  // from @com_google_absl
#include "third_party/grpc/include/grpcpp/security/server_credentials.h"
#include "third_party/grpc/include/grpcpp/server.h"
#include "third_party/grpc/include/grpcpp/server_builder.h"
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce.proto.h"
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce_interface.h"
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce_service_impl.h"
#include "tsl/platform/env.h"  // from @tsl
#include "tsl/platform/test.h"  // from @tsl
#include "tsl/platform/threadpool.h"  // from @tsl

namespace jax_sc_embedding {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::Each;

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

  Peer CreatePeer(int task_id, int num_tasks) {
    Peer peer;
    peer.task_id = task_id;
    int port = 10000 + task_id;
    peer.address = absl::StrCat("localhost:", port);

    peer.service = std::make_unique<AllReduceServiceImpl>(task_id, num_tasks);

    ::grpc::ServerBuilder builder;
    builder.AddListeningPort(peer.address,
                             ::grpc::Loas2ServerCredentials(
                                 ::grpc::Loas2ServerCredentialsOptions()));
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
    auto barrier = std::make_shared<absl::BlockingCounter>(num_tasks_);

    for (int i = 0; i < num_tasks_; ++i) {
      thread_pool_->Schedule([&, i]() {
        results[i] = peers_[i].all_reduce_interface->BlockingAllReduce(
            sync_key, inputs[i]);
        barrier->DecrementCount();
      });
    }
    barrier->Wait();
    return results;
  }
};

TEST_F(AllReduceTest, BlockingAllReduceBool) {
  // Arrange
  std::vector<bool> inputs = {true, false, true, true};
  bool expected_result = false;
  for (bool val : inputs) {
    expected_result |= val;
  }

  // Act
  std::vector<absl::StatusOr<bool>> results =
      RunAllReduceTest(/*sync_key=*/123, inputs);

  // Assert
  EXPECT_THAT(results, Each(IsOkAndHolds(expected_result)));
}

TEST_F(AllReduceTest, BlockingAllReduceUint64) {
  // Arrange
  std::vector<uint64_t> inputs = {10, 20, 30, 40};
  uint64_t expected_result = 0;
  for (uint64_t val : inputs) {
    expected_result |= val;
  }

  // Act
  std::vector<absl::StatusOr<uint64_t>> results =
      RunAllReduceTest(/*sync_key=*/456, inputs);

  // Assert
  EXPECT_THAT(results, Each(IsOkAndHolds(expected_result)));
}

class AllReduceFuzzTest
    : public fuzztest::PerFuzzTestFixtureAdapter<AllReduceTest> {
 public:
  void BlockingAllReduceBoolFuzz(int sync_key,
                                 const std::vector<bool>& inputs) {
    // Arrange
    bool expected_result = false;
    for (bool val : inputs) {
      expected_result |= val;
    }

    // Act
    std::vector<absl::StatusOr<bool>> results =
        RunAllReduceTest(sync_key, inputs);

    // Assert
    EXPECT_THAT(results, Each(IsOkAndHolds(expected_result)));
  }

  void BlockingAllReduceUint64Fuzz(int sync_key,
                                   const std::vector<uint64_t>& inputs) {
    // Arrange
    uint64_t expected_result = 0;
    for (uint64_t val : inputs) {
      expected_result |= val;
    }

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

}  // namespace
}  // namespace jax_sc_embedding
