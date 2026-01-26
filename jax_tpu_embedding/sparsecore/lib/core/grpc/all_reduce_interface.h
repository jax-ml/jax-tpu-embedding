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
#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_GRPC_ALL_REDUCE_INTERFACE_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_GRPC_ALL_REDUCE_INTERFACE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/all_reduce_interface.h"
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce.grpc.pb.h"
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce_service_impl.h"
#include "xla/tsl/concurrency/async_value_ref.h"  // from @xla

namespace jax_sc_embedding {
namespace rpc {

class GrpcAllReduceInterface final : public AllReduceInterface {
 public:
  GrpcAllReduceInterface(std::vector<std::string> peer_addresses, int task_id,
                         int num_tasks, AllReduceServiceImpl* local_service,
                         int threads_per_task = 1)
      : peer_addresses_(peer_addresses),
        task_id_(task_id),
        num_tasks_(num_tasks),
        threads_per_task_(threads_per_task),
        local_service_(local_service) {
    VLOG(2) << "GrpcAllReduceInterface created with task_id: " << task_id
              << " num_tasks: " << num_tasks
              << " peer_addresses: " << absl::StrJoin(peer_addresses, ",");
    CHECK_EQ(peer_addresses_.size(), num_tasks_ - 1);
    SetUp();
  }

  // Performs a non-blocking All-Reduce operation.
  // `sync_key`: A unique key for this all-reduce operation.
  // `minibatching_required`: The local value to be reduced.
  tsl::AsyncValueRef<bool> AsyncAllReduce(int sync_key,
                                          bool minibatching_required) override;

  // Performs a non-blocking All-Reduce operation.
  // `sync_key`: A unique key for this all-reduce operation.
  // `minibatching_split`: The local value to be reduced.
  tsl::AsyncValueRef<uint64_t> AsyncAllReduce(
      int sync_key, uint64_t minibatching_split) override;

 private:
  // Internal helper to perform the gRPC-based All-Reduce.
  // `request`: Contains the sync_key, src_rank, and the value to be reduced.
  // Returns the reduced value.
  tsl::AsyncValueRef<AllReduceData> AsyncAllReduceInternal(
      const AllReduceData& request);

  std::vector<std::string> peer_addresses_;
  int task_id_;
  int num_tasks_;
  int threads_per_task_;
  AllReduceServiceImpl* local_service_;  // Not owned.

  absl::flat_hash_map<std::string, std::unique_ptr<AllReduceGrpcService::Stub>>
      stubs_;

 private:
  // Create gRPC channels to other peers.
  void SetUp();
};

}  // namespace rpc
}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_GRPC_ALL_REDUCE_INTERFACE_H_
