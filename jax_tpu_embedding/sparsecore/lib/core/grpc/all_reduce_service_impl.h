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

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
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
//
// Synchronization Strategy:
// - `mutex_`: Protects access to `all_reduce_state_map_` and `local_data_cv_`.
//   All operations modifying or reading the map must hold this mutex.
// - `local_data_cv_`: A condition variable used to signal when a new
//   `sync_key` has been initialized via `InitializeState`. The `ContributeData`
//   RPC handlers wait on this if they receive a request for a `sync_key`
//   before the local task has called `InitializeState` for that key.
// - `AllReduceState::rpc_counter`: Each all-reduce operation (per `sync_key`)
//   has a `BlockingCounter`. This counter is initialized to `num_tasks_ - 1`
//   in `InitializeState`. Each successful `ContributeData` RPC decrements
//   this counter. The local task's `Wait` method blocks until this counter
//   reaches zero, ensuring all remote contributions have been received.
//
// Deadlock Prevention:
// The `Wait` method retrieves the `rpc_counter` pointer under `mutex_`, but
// then calls `rpc_counter->Wait()` *outside* of the mutex lock. This is crucial
// because `ContributeData` needs to acquire `mutex_` to decrement the counter.
// If `Wait` held `mutex_` while blocking on `rpc_counter`, it would prevent
// `ContributeData` from acquiring the lock and decrementing the counter,
// leading to a deadlock.
class AllReduceServiceImpl : public AllReduceGrpcService::CallbackService {
  struct AllReduceState {
    AllReduceData local_data;
    std::unique_ptr<absl::BlockingCounter> rpc_counter;
  };

 public:
  explicit AllReduceServiceImpl(int task_id, int num_tasks)
      : task_id_(task_id), num_tasks_(num_tasks) {}

  // Called by remote peers. Returns this server's local value for the sync_key.
  ::grpc::ServerUnaryReactor* ContributeData(
      ::grpc::CallbackServerContext* context, const AllReduceData* request,
      AllReduceResponse* response) override;

  // Method to register the local data for a given sync_key. Called by the local
  // client.
  void InitializeState(int sync_key, const AllReduceData& data);

  // Wait for reduction on a given sync_key.
  absl::StatusOr<AllReduceData> Wait(int sync_key);

 private:
  int task_id_;
  int num_tasks_;
  absl::Mutex mutex_;
  absl::flat_hash_map<int, AllReduceState> all_reduce_state_map_
      ABSL_GUARDED_BY(mutex_);
  absl::CondVar local_data_cv_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace rpc
}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_GRPC_ALL_REDUCE_SERVICE_IMPL_H_
