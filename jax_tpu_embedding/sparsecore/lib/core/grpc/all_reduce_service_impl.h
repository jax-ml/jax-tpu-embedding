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
#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_GRPC_ALL_REDUCE_SERVICE_IMPL_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_GRPC_ALL_REDUCE_SERVICE_IMPL_H_

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "include/grpcpp/server_context.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/support/server_callback.h"  // from @com_github_grpc_grpc
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce.grpc.pb.h"
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce.pb.h" // from internal
#include "xla/tsl/concurrency/async_value_ref.h"  // from @xla
#include "xla/tsl/concurrency/chain.h"  // from @xla

namespace jax_sc_embedding {
namespace rpc {

// Implementation of the gRPC AllReduce service. This class manages the state
// for multiple concurrent all-reduce operations, identified by a `sync_key`.
class AllReduceServiceImpl : public AllReduceGrpcService::CallbackService {
  struct AllReduceState {
    AllReduceState() = default;
    AllReduceState(AllReduceState&&) = default;
    AllReduceState& operator=(AllReduceState&&) = default;

    AllReduceData local_data;
    // Counter for all local threads to retrieve the results and delete
    // this state from the map when done.
    int results_counter;
    tsl::CountDownAsyncValueRef<tsl::Chain> local_reduction_countdown;
    tsl::CountDownAsyncValueRef<tsl::Chain> global_values_countdown;
  };

 public:
  explicit AllReduceServiceImpl(int task_id, int num_tasks,
                                int threads_per_task = 1)
      : task_id_(task_id),
        num_tasks_(num_tasks),
        threads_per_task_(threads_per_task) {}

  // Called by remote peers. Returns this server's local value for the sync_key.
  ::grpc::ServerUnaryReactor* ContributeData(
      ::grpc::CallbackServerContext* context, const AllReduceData* request,
      AllReduceResponse* response) override;

  // Method to register the local data for a given sync_key. Called by the local
  // client. Returns true if this is the last local thread to contribute.
  bool InitializeOrUpdateState(int sync_key, const AllReduceData& data);

  // Gets locally reduced value.
  tsl::AsyncValueRef<AllReduceData> GetLocalReducedValue(int sync_key);

  // Gets locally and globally reduced result.
  tsl::AsyncValueRef<AllReduceData> GetResult(int sync_key);

 private:
  // Helper to initialize AllReduceState.
  void InitializeState(AllReduceState& state, const AllReduceData& data);

  int task_id_;
  int num_tasks_;
  // Number of threads (within the same process) that will participate in the
  // all-reduce operation.
  int threads_per_task_;

  absl::Mutex mutex_;
  absl::flat_hash_map<int, AllReduceState> all_reduce_state_map_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace rpc
}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_GRPC_ALL_REDUCE_SERVICE_IMPL_H_
