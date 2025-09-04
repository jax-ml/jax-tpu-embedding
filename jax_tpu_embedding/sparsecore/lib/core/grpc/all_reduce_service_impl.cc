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
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce_service_impl.h"

#include <memory>

#include "absl/log/check.h"  // from @com_google_absl
#include "absl/synchronization/blocking_counter.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "include/grpcpp/server_context.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/support/server_callback.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/support/status.h"  // from @com_github_grpc_grpc
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce.pb.h" // from internal

namespace jax_sc_embedding {
namespace rpc {

::grpc::ServerUnaryReactor* AllReduceServiceImpl::ContributeData(
    ::grpc::CallbackServerContext* context, const AllReduceData* request,
    AllReduceResponse* response) {
  auto* reactor = context->DefaultReactor();
  absl::MutexLock lock(&mutex_);  // NOLINT (b/438618768)

  // Wait for local value.
  while (all_reduce_state_map_.find(request->sync_key()) ==
         all_reduce_state_map_.end()) {
    bool timeout = local_data_cv_.WaitWithTimeout(&mutex_, absl::Seconds(60));
    if (timeout) {
      reactor->Finish(grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED,
                                   "Timed out waiting for local value."));
      return reactor;
    }
  }

  auto it = all_reduce_state_map_.find(request->sync_key());
  CHECK(it != all_reduce_state_map_.end());
  AllReduceData& local_data = it->second.local_data;

  if (local_data.value_case() != request->value_case()) {
    it->second.rpc_counter->DecrementCount();
    reactor->Finish(
        grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                     "Request data type does not match local data type."));
    return reactor;
  }

  // Reduce local value.
  if (local_data.has_bool_val()) {
    local_data.set_bool_val(request->bool_val() || local_data.bool_val());
  } else if (local_data.has_uint64_val()) {
    local_data.set_uint64_val(request->uint64_val() | local_data.uint64_val());
  }

  all_reduce_state_map_[request->sync_key()].rpc_counter->DecrementCount();

  reactor->Finish(::grpc::Status::OK);
  return reactor;
}

void AllReduceServiceImpl::InitializeState(int sync_key,
                                           const AllReduceData& data) {
  absl::MutexLock lock(&mutex_);  // NOLINT (b/438618768)
  all_reduce_state_map_[sync_key] = {
      .local_data = data,
      .rpc_counter = std::make_unique<absl::BlockingCounter>(num_tasks_ - 1),
  };
  local_data_cv_.SignalAll();
}

absl::StatusOr<AllReduceData> AllReduceServiceImpl::Wait(int sync_key) {
  absl::BlockingCounter* rpc_counter = nullptr;
  {
    absl::MutexLock lock(&mutex_);  // NOLINT (b/438618768)
    rpc_counter = all_reduce_state_map_.at(sync_key).rpc_counter.get();
  }

  // Wait for the counter outside of the mutex lock to prevent deadlock.
  CHECK_NE(rpc_counter, nullptr);
  // TODO: b/428790659 - Add timeout (absl::BlockingCounter does not support
  // timeout, maybe use absl::CondVar::WaitWithTimeout instead?)
  rpc_counter->Wait();

  absl::MutexLock lock(&mutex_);  // NOLINT (b/438618768)
  AllReduceData result = all_reduce_state_map_[sync_key].local_data;
  all_reduce_state_map_.erase(sync_key);
  return result;
}

}  // namespace rpc
}  // namespace jax_sc_embedding
