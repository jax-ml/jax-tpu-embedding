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

#include "google/protobuf/empty.proto.h"
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/blocking_counter.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "third_party/grpc/include/grpcpp/server_context.h"
#include "third_party/grpc/include/grpcpp/support/server_callback.h"
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce.grpc.pb.h"
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce.proto.h"

namespace jax_sc_embedding {

class AllReduceServiceImpl : public AllReduceGrpcService::CallbackService {
  struct AllReduceState {
    AllReduceData local_data;
    std::unique_ptr<absl::BlockingCounter> rpc_counter;
    absl::Status overall_status;
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

}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_GRPC_ALL_REDUCE_SERVICE_IMPL_H_
