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

#include <memory>
#include <optional>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/barrier.h"  // from @com_google_absl
#include "absl/synchronization/blocking_counter.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "include/grpcpp/server_context.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/support/server_callback.h"  // from @com_github_grpc_grpc
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce.grpc.pb.h"
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce.pb.h" // from internal

namespace jax_sc_embedding {
namespace rpc {

// Implementation of the gRPC AllReduce service. This class manages the state
// for multiple concurrent all-reduce operations, identified by a `sync_key`.
class AllReduceServiceImpl : public AllReduceGrpcService::CallbackService {
  struct AllReduceState {
    AllReduceData local_data;
    // Counter to wait for all other local threads to make their
    // contributions.
    std::unique_ptr<absl::BlockingCounter> local_contributions_counter;
    // Counter for all local threads to retrieve the results and delete
    // this state from the map when done.
    std::unique_ptr<absl::BlockingCounter> results_counter;
    // Barrier to synchronize all local threads before they can retrieve the
    // final result.
    std::unique_ptr<absl::Barrier> global_results_barrier;
    // Counter for the local threads that performs the RPC to wait for all
    // other tasks.
    std::unique_ptr<absl::BlockingCounter> incoming_rpc_counter;
  };

 public:
  explicit AllReduceServiceImpl(int task_id, int num_tasks,
                                int threads_per_task)
      : task_id_(task_id),
        num_tasks_(num_tasks),
        threads_per_task_(threads_per_task) {}

  // Called by remote peers. Returns this server's local value for the sync_key.
  ::grpc::ServerUnaryReactor* ContributeData(
      ::grpc::CallbackServerContext* context, const AllReduceData* request,
      AllReduceResponse* response) override;

  // Method to register the local data for a given sync_key. Called by the local
  // client. Returns the locally-reduced data for the initializer thread, or
  // nullopt for other threads.
  absl::StatusOr<std::optional<AllReduceData>> InitializeOrUpdateState(
      int sync_key, const AllReduceData& data);

  // Waits for incoming RPCs from all other tasks. Should be called from only
  // the initializer thread.
  void WaitIncomingRPCs(int sync_key);

  // A barrier for all local threads to wait on before retrieving the result.
  void WaitResults(int sync_key);

  // Gets locally and globally reduced result.
  absl::StatusOr<AllReduceData> GetResult(int sync_key);

 private:
  int task_id_;
  int num_tasks_;
  // Number of threads (within the same task) that will participate in the
  // all-reduce operation.
  int threads_per_task_;

  absl::Mutex mutex_;
  absl::flat_hash_map<int, AllReduceState> all_reduce_state_map_
      ABSL_GUARDED_BY(mutex_);
  // CV to wait for state to be updated by all local thread.
  absl::CondVar local_reduced_cv_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace rpc
}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_GRPC_ALL_REDUCE_SERVICE_IMPL_H_
